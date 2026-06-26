"""Detecção de balões via HoughCircles + classificação de cor por amostragem central.

Estratégia:
  1. Detecta círculos na imagem em escala de cinza com cv2.HoughCircles.
  2. Para cada círculo, amostra os pixels HSV na região central (raio interno).
  3. Classifica a cor contando quantos pixels caem em cada faixa do config.
  4. Descarta círculos sem cor reconhecível (fundo, objetos não-balão).

Vantagens sobre o método de contorno:
  - Detecta balões sobrepostos individualmente.
  - Não depende de faixas HSV perfeitas para segmentar a forma; apenas para classificar.
"""

import math
import cv2
import numpy as np
from typing import List

from detection import Deteccao


def _pixels_na_faixa(hsv_roi: np.ndarray, faixas: list) -> int:
    """Conta pixels da ROI que caem em qualquer uma das faixas HSV da cor."""
    mascara = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for faixa in faixas:
        lower = np.array(faixa["lower"], dtype=np.uint8)
        upper = np.array(faixa["upper"], dtype=np.uint8)
        mascara |= cv2.inRange(hsv_roi, lower, upper)
    return int(np.count_nonzero(mascara))


def _classificar_cor(
    hsv: np.ndarray,
    cx: int,
    cy: int,
    raio: int,
    config: dict,
    fracao_interna: float = 0.6,
    limiar_cobertura: float = 0.15,
) -> str | None:
    """Retorna o nome da cor dominante no interior do círculo, ou None.

    Args:
        hsv: Imagem HSV completa.
        cx, cy: Centro do círculo em pixels.
        raio: Raio do círculo em pixels.
        config: Dicionário de cores.json.
        fracao_interna: Fração do raio usada para amostragem (evita bordas).
        limiar_cobertura: Fração mínima dos pixels que devem bater com a cor vencedora.
    """
    raio_interno = max(1, int(raio * fracao_interna))

    # Cria máscara circular na imagem completa
    mascara_circulo = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.circle(mascara_circulo, (cx, cy), raio_interno, 255, -1)

    total_pixels = int(np.count_nonzero(mascara_circulo))
    if total_pixels == 0:
        return None

    melhor_cor = None
    melhor_contagem = 0

    for cor in config["cores"]:
        # Aplica as faixas da cor apenas onde a máscara circular está ativa
        mascara_cor = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for faixa in cor["faixas"]:
            lower = np.array(faixa["lower"], dtype=np.uint8)
            upper = np.array(faixa["upper"], dtype=np.uint8)
            mascara_cor |= cv2.inRange(hsv, lower, upper)

        contagem = int(np.count_nonzero(cv2.bitwise_and(mascara_cor, mascara_circulo)))
        if contagem > melhor_contagem:
            melhor_contagem = contagem
            melhor_cor = cor["nome"]

    cobertura = melhor_contagem / total_pixels
    if cobertura < limiar_cobertura:
        return None

    return melhor_cor


def detectar_baloes_hough(
    imagem_bgr: np.ndarray,
    imagem_hsv: np.ndarray,
    config: dict,
) -> List[Deteccao]:
    """Detecta balões com HoughCircles e classifica cor por amostragem central.

    Args:
        imagem_bgr: Imagem original BGR (para dimensões e debug).
        imagem_hsv: Imagem HSV pré-processada.
        config: Dicionário carregado de cores.json.

    Returns:
        Lista de Deteccao com cor classificada, ordenada por cor e área.
    """
    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura

    cfg = config["deteccao"]
    area_min = cfg["area_minima_relativa"] * area_total
    area_max = cfg["area_maxima_relativa"] * area_total
    raio_min = max(5, int(math.sqrt(area_min / math.pi)))
    raio_max = int(math.sqrt(area_max / math.pi))

    hough_cfg = config.get("hough", {})
    dp         = hough_cfg.get("dp", 1.2)
    param1     = hough_cfg.get("param1", 100)
    param2     = hough_cfg.get("param2", 28)
    min_dist   = hough_cfg.get("min_dist_relativa", 0.03) * min(altura, largura)

    # HoughCircles opera em escala de cinza
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    cinza = cv2.medianBlur(cinza, 5)

    circulos = cv2.HoughCircles(
        cinza,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=raio_min,
        maxRadius=raio_max,
    )

    if circulos is None:
        return []

    circulos = np.round(circulos[0]).astype(int)
    deteccoes: List[Deteccao] = []

    limiar_cobertura = hough_cfg.get("limiar_cobertura_cor", 0.15)
    fracao_interna   = hough_cfg.get("fracao_interna", 0.6)

    for (cx, cy, raio) in circulos:
        # Descarta círculos com centro fora da imagem
        if not (0 <= cx < largura and 0 <= cy < altura):
            continue

        cor = _classificar_cor(
            imagem_hsv, cx, cy, raio, config,
            fracao_interna=fracao_interna,
            limiar_cobertura=limiar_cobertura,
        )
        if cor is None:
            continue

        area = math.pi * raio ** 2
        # Circularidade sempre 1.0 para Hough (são círculos por definição)
        deteccoes.append(
            Deteccao(cor=cor, cx=float(cx), cy=float(cy),
                     area=area, circularidade=1.0, raio=float(raio))
        )

    deteccoes.sort(key=lambda d: (d.cor, -d.area))
    return deteccoes
