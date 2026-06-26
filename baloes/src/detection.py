"""Detecção de balões por contorno com filtro de forma de balão.

O filtro combina cinco métricas para discriminar balões de outros objetos:

  circularidade   = 4π·A/P²            — quão próximo de um círculo (0→1)
  solidity        = A / A_hull          — quão convexo é o contorno (0→1)
  aspect_ratio    = w / h               — relação largura/altura
  variancia_cor   = std(H) na região    — uniformidade de cor interna
  defect_ratio    = Σdefects / A        — profundidade relativa de concavidades

Um balão flutuante típico tem:
  - circularidade > 0.5 (o nó na base reduz um pouco)
  - solidity > 0.85 (corpo convexo)
  - aspect_ratio 0.5–1.5
  - std(H) baixo (cor uniforme — sem logos, texturas, vincos)
  - defects pequenos relativos à área

Camisetas, banners e mesas costumam falhar em variancia_cor ou defect_ratio
mesmo quando o contorno externo parece circular.
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


def _variancia_cor(
    imagem_hsv: np.ndarray,
    contorno: np.ndarray,
) -> float:
    """Desvio padrão do canal H dentro do contorno.

    Balões têm cor uniforme (std baixo). Camisetas com logos ou roupas
    com vincos têm H variando muito internamente.
    Retorna valor normalizado [0, 1] onde 0 = perfeitamente uniforme.
    """
    mask = np.zeros(imagem_hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, cv2.FILLED)
    pixels_h = imagem_hsv[mask > 0, 0].astype(np.float32)
    if len(pixels_h) < 5:
        return 1.0
    # H é circular (0–179); usa desvio circular simplificado
    std_h = float(np.std(pixels_h))
    return std_h / 90.0  # normaliza: 90 = desvio máximo possível


def _defect_ratio(contorno: np.ndarray, area: float) -> float:
    """Soma das profundidades dos convexity defects dividida pela área.

    Balões têm poucos e pequenos defects (corpo convexo).
    Formas irregulares (camiseta dobrada, mesa) têm defects grandes.
    """
    if len(contorno) < 5 or area <= 0:
        return 0.0
    hull_idx = cv2.convexHull(contorno, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0.0
    try:
        defects = cv2.convexityDefects(contorno, hull_idx)
    except cv2.error:
        return 0.0
    if defects is None:
        return 0.0
    # Profundidade em pixels (defects[:, 0, 3] / 256.0)
    depths = defects[:, 0, 3] / 256.0
    return float(depths.sum()) / area


def _forma_de_balao(
    circularidade: float,
    solidity: float,
    aspect_ratio: float,
    cfg: dict,
) -> bool:
    """Retorna True se forma básica (contorno) é compatível com balão."""
    return (
        circularidade >= cfg["circularidade_minima"]
        and solidity >= cfg["solidity_minima"]
        and cfg["aspect_ratio_min"] <= aspect_ratio <= cfg["aspect_ratio_max"]
    )


def _textura_de_balao(
    var_cor: float,
    def_ratio: float,
    cfg: dict,
) -> bool:
    """Retorna True se a textura interna é compatível com balão."""
    var_max = cfg.get("variancia_cor_maxima", 0.25)
    def_max = cfg.get("defect_ratio_maximo", 0.15)
    return var_cor <= var_max and def_ratio <= def_max


def detectar_baloes(
    imagem_bgr: np.ndarray,
    config: dict,
    mascaras: Dict[str, np.ndarray],
) -> List[Deteccao]:
    """Detecta balões filtrando por área, forma e uniformidade de cor interna."""
    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura

    cfg = config["deteccao"]
    area_min = cfg["area_minima_relativa"] * area_total
    area_max = cfg["area_maxima_relativa"] * area_total

    imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
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

            var_cor = _variancia_cor(imagem_hsv, contorno)
            def_r = _defect_ratio(contorno, area)
            if not _textura_de_balao(var_cor, def_r, cfg):
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
