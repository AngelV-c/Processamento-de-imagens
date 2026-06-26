"""Detecção de balões por contorno com filtro de forma de balão.

O filtro combina três métricas de forma para discriminar balões de outros objetos:

  circularidade = 4π·A/P²      — quão próximo de um círculo perfeito (0→1)
  solidity      = A / A_hull   — quão convexo é o contorno (0→1)
  aspect_ratio  = w / h        — relação largura/altura do retângulo envolvente

Um balão flutuante típico tem:
  - Solidity alta (> 0.85): corpo convexo, poucas concavidades
  - Aspect ratio moderado (0.5–1.3): levemente mais alto que largo
  - Circularidade média (> 0.5): o nó na base reduz a circularidade

Essa combinação rejeita camisetas, banners, monitores e ventiladores
que passariam pelo filtro de circularidade isolado.
"""

import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


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


def _metricas_forma(contorno: np.ndarray) -> Tuple[float, float, float]:
    """Calcula (circularidade, solidity, aspect_ratio) de um contorno."""
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    circularidade = _calcular_circularidade(area, perimetro)

    hull = cv2.convexHull(contorno)
    area_hull = cv2.contourArea(hull)
    solidity = area / area_hull if area_hull > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contorno)
    aspect_ratio = w / h if h > 0 else 0.0

    return circularidade, solidity, aspect_ratio


def _forma_de_balao(
    circularidade: float,
    solidity: float,
    aspect_ratio: float,
    cfg: dict,
) -> bool:
    """Retorna True se as métricas são compatíveis com a forma de um balão."""
    return (
        circularidade >= cfg["circularidade_minima"]
        and solidity >= cfg["solidity_minima"]
        and cfg["aspect_ratio_min"] <= aspect_ratio <= cfg["aspect_ratio_max"]
    )


def detectar_baloes(
    imagem_bgr: np.ndarray,
    config: dict,
    mascaras: Dict[str, np.ndarray],
) -> List[Deteccao]:
    """Detecta balões filtrando por área e forma característica de balão.

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

    deteccoes: List[Deteccao] = []

    for nome_cor, mascara in mascaras.items():
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area < area_min or area > area_max:
                continue

            circularidade, solidity, aspect_ratio = _metricas_forma(contorno)
            if not _forma_de_balao(circularidade, solidity, aspect_ratio, cfg):
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
