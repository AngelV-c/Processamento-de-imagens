"""Detecção de balões por watershed dentro de cada máscara de cor.

Estratégia:
  1. Para cada máscara de cor (já filtrada por HSV), aplica distance transform.
  2. Encontra máximos locais na distance transform → cada máximo é um balão.
  3. Usa esses máximos como marcadores para watershed, separando balões que se tocam.
  4. Extrai contornos de cada região separada e filtra por área e circularidade.

Vantagem sobre o método de contorno puro: separa balões sobrepostos da mesma cor.
Vantagem sobre HoughCircles: só olha onde a cor é correta — não marca objetos aleatórios.
"""

import math
import cv2
import numpy as np
from typing import Dict, List

from detection import Deteccao, _calcular_circularidade


def _watershed_mascara(mascara: np.ndarray) -> np.ndarray:
    """Aplica watershed numa máscara binária e retorna imagem de labels.

    Cada região separada recebe um label inteiro positivo distinto.
    """
    # Distance transform: cada pixel recebe distância ao fundo mais próximo
    dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)

    # Normaliza para visualização e para encontrar máximos
    _, dist_norm = cv2.threshold(dist, 0, 1.0, cv2.THRESH_TOZERO)

    # Pico local: pixels que são maiores que todos os vizinhos num raio
    # Usamos dilatação: se pixel == dilatação(pixel), é máximo local
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dist_dilatada = cv2.dilate(dist_norm, kernel)
    maximos_locais = (dist_norm == dist_dilatada) & (dist_norm > 0.3)

    # Cada máximo local vira um marcador único
    _, marcadores = cv2.connectedComponents(
        maximos_locais.astype(np.uint8), connectivity=8
    )
    marcadores = marcadores.astype(np.int32)
    # Fundo da máscara vira marcador -1 para o watershed
    marcadores[mascara == 0] = -1

    # Converte máscara para BGR (watershed exige 3 canais)
    mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mascara_bgr, marcadores)

    return marcadores


def detectar_baloes_watershed(
    imagem_bgr: np.ndarray,
    mascaras: Dict[str, np.ndarray],
    config: dict,
) -> List[Deteccao]:
    """Detecta balões separando regiões sobrepostas com watershed por cor.

    Args:
        imagem_bgr: Imagem original BGR (para dimensões).
        mascaras: Dicionário {nome_cor: máscara_binária} de segmentar_cores().
        config: Dicionário carregado de cores.json.

    Returns:
        Lista de Deteccao ordenada por cor e área decrescente.
    """
    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura

    cfg = config["deteccao"]
    area_min = cfg["area_minima_relativa"] * area_total
    area_max = cfg["area_maxima_relativa"] * area_total
    circ_min = cfg["circularidade_minima"]

    deteccoes: List[Deteccao] = []

    for nome_cor, mascara in mascaras.items():
        if cv2.countNonZero(mascara) == 0:
            continue

        labels = _watershed_mascara(mascara)
        n_labels = labels.max()

        for label in range(1, n_labels + 1):
            regiao = np.zeros(mascara.shape, dtype=np.uint8)
            regiao[labels == label] = 255

            contornos, _ = cv2.findContours(
                regiao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contornos:
                continue

            contorno = max(contornos, key=cv2.contourArea)
            area = cv2.contourArea(contorno)
            if area < area_min or area > area_max:
                continue

            perimetro = cv2.arcLength(contorno, True)
            circularidade = _calcular_circularidade(area, perimetro)
            if circularidade < circ_min:
                continue

            momentos = cv2.moments(contorno)
            if momentos["m00"] == 0:
                continue
            cx = momentos["m10"] / momentos["m00"]
            cy = momentos["m01"] / momentos["m00"]
            raio = math.sqrt(area / math.pi)

            deteccoes.append(
                Deteccao(
                    cor=nome_cor,
                    cx=cx, cy=cy,
                    area=area,
                    circularidade=circularidade,
                    raio=raio,
                )
            )

    deteccoes.sort(key=lambda d: (d.cor, -d.area))
    return deteccoes
