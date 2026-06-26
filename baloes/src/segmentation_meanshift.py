"""Segmentação de cores com mean-shift como pré-processamento.

O mean-shift (pyrMeanShiftFiltering) nivela a imagem em regiões de cor
uniforme antes da limiarização HSV. Isso resolve dois problemas:

  - Reflexos especulares (ponto branco no balão): o mean-shift os absorve
    na cor média da região, eliminando o buraco na máscara.
  - Variação de iluminação (sombras, gradientes): cores ficam mais uniformes
    dentro de cada balão, reduzindo pixels que escapam da faixa HSV.

O restante do pipeline (morfologia, detecção) é idêntico ao método HSV puro.
"""

import cv2
import numpy as np
from typing import Dict


def segmentar_cores_meanshift(
    bgr: np.ndarray,
    config: dict,
    spatial_radius: int = 21,
    color_radius: int = 25,
) -> Dict[str, np.ndarray]:
    """Segmenta cores aplicando mean-shift antes da limiarização HSV.

    Args:
        bgr: Imagem original em BGR.
        config: Dicionário carregado de cores.json.
        spatial_radius: Raio espacial do mean-shift (deve ser ímpar). Maior =
            regiões maiores e mais suaves.
        color_radius: Raio de cor do mean-shift. Maior = mais cores agrupadas,
            cores mais uniformes por região.

    Returns:
        Dicionário {nome_cor: máscara_binária} equivalente ao de segmentar_cores(),
        podendo ser usado diretamente nos métodos de detecção.
    """
    # Garante raio espacial ímpar (requisito do pyrMeanShiftFiltering)
    sp = spatial_radius if spatial_radius % 2 == 1 else spatial_radius + 1

    # Mean-shift: nivela cores em regiões homogêneas
    bgr_ms = cv2.pyrMeanShiftFiltering(bgr, sp=sp, sr=color_radius)

    # Limiarização HSV sobre a imagem nivelada (mesma lógica de segmentation.py)
    hsv_ms = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2HSV)

    kernel_tam = config["deteccao"]["kernel_morfologia"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_tam, kernel_tam))

    mascaras: Dict[str, np.ndarray] = {}
    for cor in config["cores"]:
        mascara = np.zeros(hsv_ms.shape[:2], dtype=np.uint8)
        for faixa in cor["faixas"]:
            lower = np.array(faixa["lower"], dtype=np.uint8)
            upper = np.array(faixa["upper"], dtype=np.uint8)
            mascara |= cv2.inRange(hsv_ms, lower, upper)

        # Abertura remove ruído residual, fechamento preenche buracos de reflexo
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascaras[cor["nome"]] = mascara

    return mascaras
