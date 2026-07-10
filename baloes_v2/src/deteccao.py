"""Detecção de balões: candidatos multi-via + score combinado (recs. 3 e 4).

Fluxo:
  1. Normaliza iluminação (Shades-of-Gray + CLAHE) e suaviza (mean-shift).
  2. Gera candidatos por três vias independentes, cada uma vinda da
     literatura de PDI para uma dificuldade específica:
       A. máscaras de cor calibradas + watershed      (cor conhecida)
       B. círculos por SEGMENTOS DE ARCO de borda     (oclusão parcial —
          família EDCircles: Canny → ajuste de Kåsa → validação)
       C. MSER nos canais S e L                       (blobs estáveis,
          invariante a iluminação — inclui balões brancos)
  3. Cada candidato recebe score = w·(circ, solidity, fourier, conf_cor,
     brilho_especular), com aspect_ratio como penalidade suave, e passa
     pelos priors de domínio (faixa de altura, isolamento, componente).
  4. União deduplicada por distância entre centros; maior score vence.

O termo de brilho especular vem da física do material: balão de látex é
brilhante e mostra um pequeno reflexo das luzes do teto; paredes, tetos e
camisetas são foscos. É um bônus (peso pequeno), nunca uma porta.
"""

import math

import cv2
import numpy as np

from tipos import Deteccao
from preprocess import normalizar_iluminacao
from calibracao import casar_modelo, classificar_por_matches
from candidatos import candidatos_arcos, candidatos_mser
from formas import metricas_forma, fourier_energia_baixa, candidatos_de_mascara


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


def _tem_brilho_especular(regiao_sel: np.ndarray, hsv: np.ndarray) -> float:
    """1.0 se a região tem um pequeno reflexo especular, senão 0.0.

    Reflexo = mancha pequena de pixels muito claros e pouco saturados
    (0.3%–20% da área). Área grande demais indica superfície estourada
    (luminária), não reflexo pontual.
    """
    s = hsv[:, :, 1][regiao_sel]
    v = hsv[:, :, 2][regiao_sel]
    n = len(v)
    if n == 0:
        return 0.0
    frac = float(((v >= 235) & (s <= 80)).sum()) / n
    return 1.0 if 0.003 <= frac <= 0.20 else 0.0


def _penalidade_aspecto(aspect_ratio: float) -> float:
    """1.0 dentro de [0.4, 1.9]; decai suavemente fora — nunca é uma porta."""
    if 0.4 <= aspect_ratio <= 1.9:
        return 1.0
    desvio = min(abs(aspect_ratio - 0.4), abs(aspect_ratio - 1.9))
    return max(0.0, 1.0 - desvio)


def _circulo_para_contorno(cx: float, cy: float, r: float) -> np.ndarray:
    angulos = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    pts = np.stack([cx + r * np.cos(angulos), cy + r * np.sin(angulos)], axis=1)
    return pts.reshape(-1, 1, 2).astype(np.int32)


def detectar_baloes(imagem_bgr: np.ndarray, params: dict,
                    calibracao: dict) -> tuple[list[Deteccao], dict]:
    """Detecta balões com calibração local + multi-via + score combinado.

    Cores com "ativa": false na calibração são ignoradas — a interface
    permite escolher quais cores da cena participam da detecção.

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
    ajustes = params.get("modelo")
    vias = det.get("vias", {})
    usar_a = vias.get("mascaras", True)
    usar_b = vias.get("arcos", vias.get("hough", True))
    usar_c = vias.get("mser", vias.get("saturacao", True))

    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura
    area_min = det["area_minima_relativa"] * area_total
    area_max = det["area_maxima_relativa"] * area_total
    r_min = max(4.0, math.sqrt(area_min / math.pi))
    r_max = min(math.sqrt(area_max / math.pi), largura * 0.08)

    bgr_norm = normalizar_iluminacao(imagem_bgr)
    bgr_ms = cv2.pyrMeanShiftFiltering(bgr_norm, sp=det["meanshift_sp"], sr=det["meanshift_sr"])
    hsv = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2LAB).astype(np.float64)

    modelos = {nome: m for nome, m in calibracao["cores"].items()
               if m.get("ativa", True)}

    # Casamentos pré-computados UMA vez (imagem inteira × cor) — cada
    # candidato só indexa; essencial com as dezenas de regiões do MSER.
    matches = {nome: casar_modelo(hsv, lab, modelo, ajustes)
               for nome, modelo in modelos.items()}

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (det["kernel_morfologia"],) * 2)
    mascaras: dict = {}
    for nome, m in matches.items():
        binaria = (m * 255).astype(np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
        mascaras[nome] = binaria

    # --- Candidatos pelas vias habilitadas ---
    candidatos: list[dict] = []
    if usar_a:
        for nome, mascara in mascaras.items():
            for contorno, comp_area in candidatos_de_mascara(
                    mascara, ws["kernel_maximos"], ws["limiar_distancia"]):
                candidatos.append({"contorno": contorno, "via": "A", "comp_area": comp_area})

    if usar_b:
        for cx, cy, r in candidatos_arcos(bgr_norm, params.get("arcos", {}), r_min, r_max):
            candidatos.append({"contorno": _circulo_para_contorno(cx, cy, r),
                               "via": "B", "comp_area": None})

    if usar_c:
        for contorno, comp_area in candidatos_mser(
                bgr_norm, hsv, params.get("mser", {}), area_min, area_max):
            candidatos.append({"contorno": contorno, "via": "C", "comp_area": comp_area})

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

        cor, conf = classificar_por_matches(sel, matches, lab, modelos)
        # Círculo sintético da via B tem forma perfeita por construção —
        # a cor é o único portão real dessa via, e por isso exige mais.
        conf_necessaria = (det["conf_cor_minima_hough"] if cand["via"] == "B"
                           else det["conf_cor_minima"])
        if cor is None or conf < conf_necessaria:
            continue

        # A via C (MSER no canal S) existe para balões CROMÁTICOS; o MSER
        # também devolve regiões de S baixa (camisetas/rostos neutros) que
        # casariam com cores acromáticas — balões brancos são cobertos
        # pelas vias A e B, então aqui rejeitamos.
        if cand["via"] == "C" and modelos[cor].get("acromatica"):
            continue

        if _fracao_anel_mesma_cor(regiao, mascaras[cor], raio) > det["anel_mesma_cor_max"]:
            continue  # imerso na própria cor (teto/parede)

        brilho = _tem_brilho_especular(sel, hsv)
        score = (pesos["circularidade"] * circ
                 + pesos["solidity"] * sol
                 + pesos["fourier"] * fourier
                 + pesos["cor"] * min(1.0, conf)
                 + pesos.get("brilho", 0.0) * brilho) * _penalidade_aspecto(ar)
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
