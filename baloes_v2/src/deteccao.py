"""Detecção de balões: candidatos multi-via + score combinado (recs. 3 e 4).

Fluxo:
  1. Normaliza iluminação e suaviza (mean-shift leve).
  2. Gera candidatos por três vias independentes:
       A. máscaras de cor calibradas + watershed
       B. HoughCircles (vota com arcos — funciona sob oclusão parcial)
       C. saturação adaptativa (percentil da cena) + watershed
  3. Cada candidato recebe score = w·(circ, solidity, fourier, conf_cor),
     com aspect_ratio como penalidade suave, e passa pelos priors de
     domínio (faixa de altura, isolamento, contexto de componente).
  4. União deduplicada por distância entre centros; maior score vence.
"""

import math

import cv2
import numpy as np

from tipos import Deteccao
from preprocess import normalizar_iluminacao
from calibracao import mascaras_calibradas, classificar_regiao
from formas import metricas_forma, fourier_energia_baixa, candidatos_de_mascara


def _mascara_saturacao_adaptativa(hsv: np.ndarray, percentil: float,
                                  kernel_tam: int) -> np.ndarray:
    """Pixels com S acima do percentil da CENA (não um valor absoluto).

    O fundo neutro domina o histograma de saturação; os balões cromáticos
    são os outliers do topo — o piso se adapta sozinho a cada imagem.
    """
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    m = ((s >= np.percentile(s, percentil)) & (v >= 40)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_tam, kernel_tam))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m


def _candidatos_hough(bgr_norm: np.ndarray, cfg: dict, area_min: float,
                      area_max: float) -> list[tuple[float, float, float]]:
    """HoughCircles sobre a imagem normalizada → (cx, cy, r) por círculo."""
    cinza = cv2.medianBlur(cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2GRAY), 5)
    largura = bgr_norm.shape[1]
    circulos = cv2.HoughCircles(
        cinza, cv2.HOUGH_GRADIENT,
        dp=cfg.get("dp", 1.2),
        minDist=max(10, int(cfg.get("min_dist_relativa", 0.06) * largura * 0.5)),
        param1=cfg.get("param1", 120),
        param2=cfg.get("param2", 45),
        minRadius=max(4, int(math.sqrt(area_min / math.pi))),
        maxRadius=int(min(math.sqrt(area_max / math.pi), largura * 0.08)),
    )
    if circulos is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circulos[0]]


def _fracao_anel_mesma_cor(regiao: np.ndarray, mascara_cor: np.ndarray,
                           raio: float) -> float:
    """Fração do anel externo da região que também pertence à mesma cor.

    Balão = objeto isolado (anel é fundo). Pedaço de teto retalhado pelo
    watershed = imerso na própria cor (anel devolve fração alta).
    """
    espessura = max(3, int(raio * 0.35))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (espessura * 2 + 1,) * 2)
    anel = cv2.subtract(cv2.dilate(regiao, kernel), regiao)
    n_anel = cv2.countNonZero(anel)
    if n_anel == 0:
        return 1.0
    return cv2.countNonZero(cv2.bitwise_and(anel, mascara_cor)) / n_anel


def _penalidade_aspecto(aspect_ratio: float) -> float:
    """1.0 dentro de [0.4, 1.9]; decai suavemente fora — nunca é uma porta."""
    if 0.4 <= aspect_ratio <= 1.9:
        return 1.0
    desvio = min(abs(aspect_ratio - 0.4), abs(aspect_ratio - 1.9))
    return max(0.0, 1.0 - desvio)


def detectar_baloes(imagem_bgr: np.ndarray, params: dict,
                    calibracao: dict) -> tuple[list[Deteccao], dict]:
    """Detecta balões com calibração local + multi-via + score combinado.

    Args:
        imagem_bgr: Imagem BGR (mesma largura usada na calibração!).
        params: params.json carregado.
        calibracao: calibracao_local.json carregado (modelos por cor).

    Returns:
        (lista de Deteccao, {nome_cor: máscara} para debug/UI).
    """
    det = params["deteccao"]
    pesos = det["pesos"]
    ws = params["watershed"]

    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura
    area_min = det["area_minima_relativa"] * area_total
    area_max = det["area_maxima_relativa"] * area_total

    bgr_norm = normalizar_iluminacao(imagem_bgr)
    bgr_ms = cv2.pyrMeanShiftFiltering(bgr_norm, sp=det["meanshift_sp"], sr=det["meanshift_sr"])
    hsv = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2LAB).astype(np.float64)

    modelos = calibracao["cores"]
    mascaras = mascaras_calibradas(hsv, lab, modelos, det["kernel_morfologia"])

    # --- Candidatos pelas três vias ---
    candidatos: list[dict] = []
    for nome, mascara in mascaras.items():
        for contorno, comp_area in candidatos_de_mascara(
                mascara, ws["kernel_maximos"], ws["limiar_distancia"]):
            candidatos.append({"contorno": contorno, "via": "A", "comp_area": comp_area})

    m_sat = _mascara_saturacao_adaptativa(hsv, det["sat_percentil"], det["kernel_morfologia"])
    for contorno, comp_area in candidatos_de_mascara(
            m_sat, ws["kernel_maximos"], ws["limiar_distancia"]):
        candidatos.append({"contorno": contorno, "via": "C", "comp_area": comp_area})

    for cx, cy, r in _candidatos_hough(bgr_norm, params["hough"], area_min, area_max):
        angulos = np.linspace(0, 2 * math.pi, 32, endpoint=False)
        pts = np.stack([cx + r * np.cos(angulos), cy + r * np.sin(angulos)], axis=1)
        candidatos.append({"contorno": pts.reshape(-1, 1, 2).astype(np.int32),
                           "via": "B", "comp_area": None})

    # --- Priors de domínio + score combinado ---
    cy_min = det["fracao_altura_min"] * altura
    cy_max = det["fracao_altura_max"] * altura

    avaliados: list[dict] = []
    for cand in candidatos:
        contorno = cand["contorno"]
        area = cv2.contourArea(contorno)
        if area < area_min or area > area_max:
            continue

        comp_area = cand["comp_area"]
        if comp_area is not None and comp_area > det["componente_max_razao"] * area:
            continue  # retalho de teto/parede, não balão

        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue
        cx = momentos["m10"] / momentos["m00"]
        cy = momentos["m01"] / momentos["m00"]
        if cy < cy_min or cy > cy_max:
            continue  # fora da faixa onde balões podem flutuar

        circ, sol, ar = metricas_forma(contorno)
        fourier = fourier_energia_baixa(contorno)
        raio = math.sqrt(area / math.pi)

        regiao = np.zeros((altura, largura), dtype=np.uint8)
        cv2.drawContours(regiao, [contorno], -1, 255, -1)
        # Interior erodido: não contamina a confiança com pixels de borda
        interno = cv2.erode(regiao, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, int(raio * 0.2)),) * 2))
        sel = (interno if cv2.countNonZero(interno) > 20 else regiao) > 0

        cor, conf = classificar_regiao(sel, hsv, lab, modelos)
        # Círculo sintético do Hough tem forma perfeita por construção —
        # a cor é o único portão real da via B, e por isso exige mais.
        conf_necessaria = (det["conf_cor_minima_hough"] if cand["via"] == "B"
                           else det["conf_cor_minima"])
        if cor is None or conf < conf_necessaria:
            continue

        if _fracao_anel_mesma_cor(regiao, mascaras[cor], raio) > det["anel_mesma_cor_max"]:
            continue  # imerso na própria cor (teto/parede)

        score = (pesos["circularidade"] * circ
                 + pesos["solidity"] * sol
                 + pesos["fourier"] * fourier
                 + pesos["cor"] * min(1.0, conf)) * _penalidade_aspecto(ar)
        if score < det["score_minimo"]:
            continue

        avaliados.append({"cor": cor, "cx": cx, "cy": cy, "area": area,
                          "circ": circ, "raio": raio, "score": score,
                          "via": cand["via"]})

    # --- Deduplicação: maior score vence ---
    avaliados.sort(key=lambda c: -c["score"])
    aceitos: list[dict] = []
    for cand in avaliados:
        if all(math.hypot(cand["cx"] - a["cx"], cand["cy"] - a["cy"])
               >= 0.6 * (cand["raio"] + a["raio"]) for a in aceitos):
            aceitos.append(cand)

    deteccoes = [Deteccao(cor=c["cor"], cx=c["cx"], cy=c["cy"], area=c["area"],
                          circularidade=c["circ"], raio=c["raio"],
                          score=c["score"], via=c["via"])
                 for c in aceitos]
    deteccoes.sort(key=lambda d: (d.cor, -d.area))
    return deteccoes, mascaras
