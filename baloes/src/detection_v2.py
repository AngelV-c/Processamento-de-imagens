"""Pipeline v2 — arquitetura reestruturada de detecção de balões.

Implementa os quatro princípios da reestruturação:

1. CALIBRAÇÃO POR LOCAL — os modelos de cor vêm de calibracao_local.json
   (gerado por tools/calibrar_cores.py clicando em 1 balão de cada cor),
   não de faixas HSV fixas. Cores acromáticas (branco/cinza) são modeladas
   por LAB + saturação baixa, impossível em segmentação por matiz.

2. LIMIARES RELATIVOS — a iluminação é normalizada na entrada (gray-world +
   CLAHE) e os pisos de S/V derivam dos percentis da própria amostra
   calibrada; a via de saturação usa percentil da distribuição da CENA.

3. SCORE COMBINADO — em vez da cascata de ANDs, cada candidato recebe
       score = w1·circularidade + w2·solidity + w3·fourier + w4·conf_cor
   e é aceito se score ≥ limiar único. Um balão ocluído com circularidade
   baixa mas cor e espectro perfeitos sobrevive.

4. CANDIDATOS MULTI-VIA — três detectores fracos independentes:
       A. máscaras de cor calibradas + watershed   (preciso, cor conhecida)
       B. HoughCircles                             (acha círculos por ARCOS —
          funciona com oclusão parcial, onde nenhum blob fechado existe)
       C. saturação adaptativa (percentil) + watershed (pega cromáticos
          que escaparam dos modelos)
   A união é deduplicada por distância entre centros; vence o maior score.
"""

import math

import cv2
import numpy as np

from tipos import Deteccao
from detection import _metricas_forma
from detection_fourier import _fourier_energia_baixa
from detection_watershed import _watershed_mascara
from preprocess import normalizar_iluminacao


# ---------------------------------------------------------------------------
# Modelos de cor calibrados
# ---------------------------------------------------------------------------

def _dist_h_circular(h: np.ndarray, h_ref: float) -> np.ndarray:
    """Distância de matiz respeitando a circularidade (0–179)."""
    d = np.abs(h.astype(np.float64) - h_ref)
    return np.minimum(d, 180.0 - d)


def _match_modelo(
    hsv: np.ndarray,
    lab: np.ndarray,
    modelo: dict,
    mahalanobis_max: float,
) -> np.ndarray:
    """Máscara booleana dos pixels compatíveis com um modelo de cor calibrado."""
    h = hsv[:, :, 0].astype(np.float64)
    s = hsv[:, :, 1].astype(np.float64)
    v = hsv[:, :, 2].astype(np.float64)

    if modelo["acromatica"]:
        # Sem matiz confiável: croma a-b APERTADO + saturação baixa + brilho.
        # Mahalanobis com covariância multi-clique fica largo demais e engole
        # paredes/tetos — portões explícitos são mais controláveis aqui.
        media = modelo["lab_media"]
        cov = modelo["lab_cov"]
        ab_tol = max(8.0, min(15.0, 2.5 * math.sqrt(cov[1][1] + cov[2][2])))
        dist_ab = np.sqrt((lab[:, :, 1] - media[1]) ** 2 + (lab[:, :, 2] - media[2]) ** 2)
        s_teto = max(50.0, modelo["s_p90"] * 1.2)
        v_piso = modelo["v_p10"] * 0.75
        # O brilho L separa balão branco (alto) de teto/camiseta (mais escuros)
        # E de luminárias: glare é mais claro que qualquer balão (teto de L
        # apertado + corte absoluto de pixels estourados em V).
        l_piso = modelo.get("l_p10", 0.0) * 0.9
        l_teto = modelo.get("l_p90", 255.0) * 1.08
        L = lab[:, :, 0]
        return ((dist_ab <= ab_tol) & (s <= s_teto) & (v >= v_piso)
                & (v <= 250) & (L >= l_piso) & (L <= l_teto))

    # Cromática: janela circular de matiz + pisos RELATIVOS à amostra
    # + portão de cromaticidade no plano a-b do LAB — o matiz sozinho não
    # separa balão amarelo de parede creme; a distância a-b separa.
    dh = _dist_h_circular(h, modelo["h_mediana"])
    s_piso = max(30.0, modelo["s_p10"] * 0.5)
    v_piso = max(30.0, modelo["v_p10"] * 0.5)

    media = modelo["lab_media"]
    cov = modelo["lab_cov"]
    ab_tol = max(15.0, min(30.0, 2.5 * math.sqrt(cov[1][1] + cov[2][2])))
    dist_ab = np.sqrt((lab[:, :, 1] - media[1]) ** 2 + (lab[:, :, 2] - media[2]) ** 2)

    return (dh <= modelo["h_tolerancia"]) & (s >= s_piso) & (v >= v_piso) & (dist_ab <= ab_tol)


def _mascaras_calibradas(
    hsv: np.ndarray,
    lab: np.ndarray,
    modelos: dict,
    mahalanobis_max: float,
    kernel_tam: int,
) -> dict:
    """Uma máscara binária limpa por cor calibrada."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_tam, kernel_tam))
    mascaras = {}
    for nome, modelo in modelos.items():
        m = (_match_modelo(hsv, lab, modelo, mahalanobis_max) * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        mascaras[nome] = m
    return mascaras


def _classificar_regiao(
    regiao_sel: np.ndarray,
    hsv: np.ndarray,
    lab: np.ndarray,
    modelos: dict,
    mahalanobis_max: float,
) -> tuple[str | None, float]:
    """(cor, confiança) da região: cor com maior fração de pixels compatíveis."""
    n_total = int(regiao_sel.sum())
    if n_total == 0:
        return None, 0.0

    melhor_nome, melhor_frac = None, 0.0
    for nome, modelo in modelos.items():
        match = _match_modelo(hsv, lab, modelo, mahalanobis_max)
        frac = float((match & regiao_sel).sum()) / n_total
        if frac > melhor_frac:
            melhor_nome, melhor_frac = nome, frac
    return melhor_nome, melhor_frac


# ---------------------------------------------------------------------------
# Geração de candidatos (multi-via)
# ---------------------------------------------------------------------------

def _candidatos_de_mascara(
    mascara: np.ndarray, ws_cfg: dict
) -> list[tuple[np.ndarray, float]]:
    """Watershed numa máscara → lista de (contorno, área_do_componente_original).

    A área do componente conexo ANTES do watershed é o contexto do candidato:
    um balão (mesmo num cacho de 3–4) vem de um componente pequeno; um pedaço
    de teto/parede vem do retalho de um componente gigante. Esse contexto
    permite rejeitar recortes de superfícies que são idênticos a balões em
    cor e forma.
    """
    if cv2.countNonZero(mascara) == 0:
        return []

    n_comp, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)

    labels = _watershed_mascara(
        mascara, ws_cfg.get("kernel_maximos", 13), ws_cfg.get("limiar_distancia", 0.25)
    )
    candidatos = []
    for label in range(1, labels.max() + 1):
        reg = np.zeros(mascara.shape, dtype=np.uint8)
        reg[(labels == label) & (mascara > 0)] = 255
        cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contorno = max(cnts, key=cv2.contourArea)

        ys, xs = np.where(reg > 0)
        comp_id = comp_labels[ys[0], xs[0]]
        comp_area = float(stats[comp_id, cv2.CC_STAT_AREA])
        candidatos.append((contorno, comp_area))
    return candidatos


def _candidatos_hough(bgr_norm: np.ndarray, cfg_hough: dict, area_min: float,
                      area_max: float) -> list[tuple[float, float, float]]:
    """HoughCircles sobre a imagem normalizada → lista de (cx, cy, r).

    O acumulador do Hough vota com ARCOS de borda — um balão 40% ocluído
    ainda gera um pico, onde nenhum blob fechado de cor existe.
    """
    cinza = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2GRAY)
    cinza = cv2.medianBlur(cinza, 5)
    largura = bgr_norm.shape[1]

    r_min = max(4, int(math.sqrt(area_min / math.pi)))
    r_max = int(min(math.sqrt(area_max / math.pi), largura * 0.08))

    circulos = cv2.HoughCircles(
        cinza, cv2.HOUGH_GRADIENT,
        dp=cfg_hough.get("dp", 1.2),
        minDist=max(10, int(cfg_hough.get("min_dist_relativa", 0.06) * largura * 0.5)),
        param1=cfg_hough.get("param1", 120),
        param2=cfg_hough.get("param2", 45),
        minRadius=r_min, maxRadius=r_max,
    )
    if circulos is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circulos[0]]


def _mascara_saturacao_adaptativa(hsv: np.ndarray, percentil: float,
                                  kernel_tam: int) -> np.ndarray:
    """Pixels com S acima do percentil da CENA (não um valor absoluto).

    O fundo neutro domina o histograma de saturação; os balões cromáticos
    são os outliers do topo — o piso se adapta sozinho a cada imagem.
    """
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    s_piso = np.percentile(s, percentil)
    m = ((s >= s_piso) & (v >= 40)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_tam, kernel_tam))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m


# ---------------------------------------------------------------------------
# Score combinado e deduplicação
# ---------------------------------------------------------------------------

def _fracao_anel_mesma_cor(
    regiao: np.ndarray,
    mascara_cor: np.ndarray,
    raio: float,
) -> float:
    """Fração do anel externo da região que também pertence à mesma cor.

    Um balão é um objeto ISOLADO: o anel ao seu redor é fundo (outra cor).
    Um pedaço de teto/parede retalhado pelo watershed está imerso na própria
    cor — o anel devolve fração alta. Este critério elimina recortes de
    superfícies grandes que passariam em forma e cor.
    """
    espessura = max(3, int(raio * 0.35))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (espessura * 2 + 1,) * 2)
    dilatada = cv2.dilate(regiao, kernel)
    anel = cv2.subtract(dilatada, regiao)
    n_anel = cv2.countNonZero(anel)
    if n_anel == 0:
        return 1.0
    mesma = cv2.bitwise_and(anel, mascara_cor)
    return cv2.countNonZero(mesma) / n_anel


def _penalidade_aspecto(aspect_ratio: float) -> float:
    """1.0 dentro de [0.4, 1.9]; decai suavemente fora — nunca é uma porta."""
    if 0.4 <= aspect_ratio <= 1.9:
        return 1.0
    desvio = min(abs(aspect_ratio - 0.4), abs(aspect_ratio - 1.9))
    return max(0.0, 1.0 - desvio)


def detectar_baloes_v2(
    imagem_bgr: np.ndarray,
    config: dict,
    calibracao: dict,
) -> tuple[list[Deteccao], dict]:
    """Detecta balões com a arquitetura v2 (calibração local + multi-via + score).

    Args:
        imagem_bgr: Imagem original BGR (já redimensionada como na calibração).
        config: cores.json + params.json unidos (usa config["v2"], ["watershed"],
            ["deteccao"] para áreas e ["hough"]).
        calibracao: calibracao_local.json carregado (modelos por cor).

    Returns:
        (lista de Deteccao, {nome_cor: máscara}) — as máscaras para debug/UI.
    """
    v2 = config.get("v2", {})
    pesos = v2.get("pesos", {"circularidade": 0.2, "solidity": 0.15,
                             "fourier": 0.25, "cor": 0.4})
    score_min = v2.get("score_minimo", 0.75)
    conf_min = v2.get("conf_cor_minima", 0.30)
    conf_min_hough = v2.get("conf_cor_minima_hough", 0.55)
    anel_max = v2.get("anel_mesma_cor_max", 0.40)
    componente_razao = v2.get("componente_max_razao", 6.0)
    mah_max = v2.get("mahalanobis_max", 12.0)
    sat_pct = v2.get("sat_percentil", 85)

    cfg_det = config["deteccao"]
    kernel_tam = cfg_det.get("kernel_morfologia", 5)
    ws_cfg = config.get("watershed", {})

    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura
    area_min = cfg_det["area_minima_relativa"] * area_total
    area_max = cfg_det["area_maxima_relativa"] * area_total

    # Normalização obrigatória + suavização leve
    bgr_norm = normalizar_iluminacao(imagem_bgr)
    bgr_ms = cv2.pyrMeanShiftFiltering(
        bgr_norm, sp=v2.get("meanshift_sp", 10), sr=v2.get("meanshift_sr", 20)
    )
    hsv = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2LAB).astype(np.float64)

    modelos = calibracao["cores"]
    mascaras = _mascaras_calibradas(hsv, lab, modelos, mah_max, kernel_tam)

    # --- Geração de candidatos pelas três vias ---
    candidatos: list[dict] = []

    # Via A: máscaras calibradas + watershed
    for nome, mascara in mascaras.items():
        for contorno, comp_area in _candidatos_de_mascara(mascara, ws_cfg):
            candidatos.append({"contorno": contorno, "via": "A", "comp_area": comp_area})

    # Via C: saturação adaptativa + watershed
    m_sat = _mascara_saturacao_adaptativa(hsv, sat_pct, kernel_tam)
    for contorno, comp_area in _candidatos_de_mascara(m_sat, ws_cfg):
        candidatos.append({"contorno": contorno, "via": "C", "comp_area": comp_area})

    # Via B: círculos por arcos de borda (oclusão parcial)
    # As métricas de forma de um círculo sintético são perfeitas por
    # construção — a cor é o ÚNICO portão real desta via, por isso ela
    # exige confiança bem mais alta (conf_min_hough).
    for cx, cy, r in _candidatos_hough(bgr_norm, config.get("hough", {}),
                                       area_min, area_max):
        angulos = np.linspace(0, 2 * math.pi, 32, endpoint=False)
        pontos = np.stack([cx + r * np.cos(angulos), cy + r * np.sin(angulos)], axis=1)
        contorno = pontos.reshape(-1, 1, 2).astype(np.int32)
        candidatos.append({"contorno": contorno, "via": "B", "comp_area": None})

    # --- Avaliação: score combinado por candidato ---
    avaliados: list[dict] = []
    # Prior de domínio: balões flutuam presos às mesas — sempre na faixa
    # média do enquadramento. Abaixo da faixa: pessoas, camisetas e mesas;
    # acima dela: o teto e suas luminárias (um balão branco, uma camiseta
    # branca e um teto branco são indistinguíveis por cor; a POSIÇÃO separa).
    cy_max = v2.get("fracao_altura_max", 0.75) * altura
    cy_min = v2.get("fracao_altura_min", 0.15) * altura

    for cand in candidatos:
        contorno = cand["contorno"]
        area = cv2.contourArea(contorno)
        if area < area_min or area > area_max:
            continue

        # Contexto: candidato vindo do retalho de um componente gigante
        # (teto, parede) não é balão — cacho real tem componente ≤ ~6 balões
        comp_area = cand.get("comp_area")
        if comp_area is not None and comp_area > componente_razao * area:
            continue

        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue
        cx = momentos["m10"] / momentos["m00"]
        cy = momentos["m01"] / momentos["m00"]
        if cy > cy_max or cy < cy_min:
            continue

        circ, sol, ar = _metricas_forma(contorno)
        fourier = _fourier_energia_baixa(contorno)

        regiao = np.zeros((altura, largura), dtype=np.uint8)
        cv2.drawContours(regiao, [contorno], -1, 255, -1)
        # Interior a 80% do raio: evita contaminar a confiança com a borda
        raio = math.sqrt(area / math.pi)
        interno = cv2.erode(regiao, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, int(raio * 0.2)),) * 2))
        sel = (interno if cv2.countNonZero(interno) > 20 else regiao) > 0

        cor, conf = _classificar_regiao(sel, hsv, lab, modelos, mah_max)
        conf_necessaria = conf_min_hough if cand["via"] == "B" else conf_min
        if cor is None or conf < conf_necessaria:
            continue

        # Isolamento: rejeita recortes de tetos/paredes imersos na própria cor
        if _fracao_anel_mesma_cor(regiao, mascaras[cor], raio) > anel_max:
            continue

        score = (
            pesos["circularidade"] * circ
            + pesos["solidity"] * sol
            + pesos["fourier"] * fourier
            + pesos["cor"] * min(1.0, conf)
        ) * _penalidade_aspecto(ar)

        if score < score_min:
            continue

        avaliados.append({
            "cor": cor, "cx": cx, "cy": cy, "area": area,
            "circularidade": circ, "raio": raio, "score": score,
        })

    # --- Deduplicação: maior score vence; vizinho próximo demais é descartado ---
    avaliados.sort(key=lambda c: -c["score"])
    aceitos: list[dict] = []
    for cand in avaliados:
        duplicado = False
        for ac in aceitos:
            dist = math.hypot(cand["cx"] - ac["cx"], cand["cy"] - ac["cy"])
            if dist < 0.6 * (cand["raio"] + ac["raio"]):
                duplicado = True
                break
        if not duplicado:
            aceitos.append(cand)

    deteccoes = [
        Deteccao(cor=c["cor"], cx=c["cx"], cy=c["cy"], area=c["area"],
                 circularidade=c["circularidade"], raio=c["raio"])
        for c in aceitos
    ]
    deteccoes.sort(key=lambda d: (d.cor, -d.area))
    return deteccoes, mascaras
