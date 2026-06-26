"""Detecção de balões por contorno e circularidade."""

import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Deteccao:
    """Representa um balão detectado na imagem.

    Atributos:
        cor: Nome da cor conforme definido em cores.json.
        cx: Coordenada x do centroide em pixels.
        cy: Coordenada y do centroide em pixels.
        area: Área do contorno em pixels².
        circularidade: Métrica 4π·A/P² no intervalo (0, 1].
        raio: Raio aproximado do balão em pixels.
    """

    cor: str
    cx: float
    cy: float
    area: float
    circularidade: float
    raio: float


def _calcular_circularidade(area: float, perimetro: float) -> float:
    """Retorna 4π·A/P² ou 0 se o perímetro for zero."""
    if perimetro <= 0:
        return 0.0
    return 4 * math.pi * area / (perimetro ** 2)


def detectar_baloes(
    imagem_bgr: np.ndarray,
    config: dict,
    mascaras: Dict[str, np.ndarray],
) -> List[Deteccao]:
    """Detecta balões em cada máscara e retorna lista de Deteccao.

    Filtra contornos por área relativa ao tamanho da imagem e por circularidade.

    Args:
        imagem_bgr: Imagem original em BGR (usada apenas para obter dimensões).
        config: Dicionário carregado de cores.json.
        mascaras: Dicionário {nome_cor: máscara_binária} vindo de segmentar_cores().

    Returns:
        Lista de Deteccao ordenada por cor e depois por área decrescente.
    """
    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura

    cfg = config["deteccao"]
    area_min = cfg["area_minima_relativa"] * area_total
    area_max = cfg["area_maxima_relativa"] * area_total
    circ_min = cfg["circularidade_minima"]

    deteccoes: List[Deteccao] = []

    for nome_cor, mascara in mascaras.items():
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contorno in contornos:
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
                    cx=cx,
                    cy=cy,
                    area=area,
                    circularidade=circularidade,
                    raio=raio,
                )
            )

    deteccoes.sort(key=lambda d: (d.cor, -d.area))
    return deteccoes
