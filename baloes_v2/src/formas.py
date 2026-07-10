"""Métricas de forma e geração de candidatos por watershed.

Contém as três medidas geométricas do score (circularidade, solidity,
energia espectral de Fourier) e o watershed que separa balões colados.
"""

import math

import cv2
import numpy as np


def metricas_forma(contorno: np.ndarray) -> tuple[float, float, float]:
    """(circularidade, solidity, aspect_ratio) de um contorno.

    circularidade = 4π·A/P² (protegida contra perímetro zero);
    solidity = A/A_hull; aspect_ratio = w/h do retângulo envolvente.
    """
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    circularidade = 4 * math.pi * area / (perimetro ** 2) if perimetro > 0 else 0.0

    hull = cv2.convexHull(contorno)
    area_hull = cv2.contourArea(hull)
    solidity = area / area_hull if area_hull > 0 else 0.0

    _, _, w, h = cv2.boundingRect(contorno)
    aspect_ratio = w / h if h > 0 else 0.0
    return circularidade, solidity, aspect_ratio


def fourier_energia_baixa(contorno: np.ndarray, n: int = 64, k_baixo: int = 4) -> float:
    """Fração da energia do sinal distância-ao-centroide nos harmônicos 1..K.

    Amostra N pontos equidistantes no contorno, monta d(t) = distância ao
    centroide, aplica FFT. Formas quase circulares concentram energia nos
    harmônicos baixos; contornos irregulares espalham para os altos.
    """
    pts = contorno.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return 0.0

    diffs = np.diff(pts, axis=0, append=pts[:1])
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cums = np.concatenate([[0.0], np.cumsum(dists)])
    total = cums[-1]
    if total == 0:
        return 0.0
    alvos = np.linspace(0, total, n, endpoint=False)
    xs = np.interp(alvos, cums, np.append(pts[:, 0], pts[0, 0]))
    ys = np.interp(alvos, cums, np.append(pts[:, 1], pts[0, 1]))

    momentos = cv2.moments(contorno)
    if momentos["m00"] == 0:
        return 0.0
    cx = momentos["m10"] / momentos["m00"]
    cy = momentos["m01"] / momentos["m00"]

    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    potencia = np.abs(np.fft.rfft(d)) ** 2
    energia_total = potencia[1:].sum()
    if energia_total == 0:
        return 0.0
    return float(potencia[1: k_baixo + 1].sum() / energia_total)


def watershed_mascara(mascara: np.ndarray, kernel_maximos: int = 13,
                      limiar_distancia: float = 0.25) -> np.ndarray:
    """Watershed por distance transform: separa balões que se tocam."""
    dist = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
    dist_max = dist.max()
    if dist_max == 0:
        return np.zeros_like(mascara, dtype=np.int32)
    dist_norm = dist / dist_max

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_maximos, kernel_maximos))
    dilatada = cv2.dilate(dist_norm, kernel)
    maximos = (dist_norm == dilatada) & (dist_norm > limiar_distancia)

    _, marcadores = cv2.connectedComponents(maximos.astype(np.uint8), connectivity=8)
    marcadores = marcadores.astype(np.int32)
    marcadores[mascara == 0] = -1
    cv2.watershed(cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR), marcadores)
    return marcadores


def candidatos_de_mascara(mascara: np.ndarray, kernel_maximos: int = 13,
                          limiar_distancia: float = 0.25
                          ) -> list[tuple[np.ndarray, float]]:
    """Watershed → lista de (contorno, área_do_componente_original).

    A área do componente conexo ANTES do watershed é o CONTEXTO do candidato:
    um balão (mesmo num cacho) vem de componente pequeno; um pedaço de teto
    vem do retalho de um componente gigante.
    """
    if cv2.countNonZero(mascara) == 0:
        return []

    _, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    labels = watershed_mascara(mascara, kernel_maximos, limiar_distancia)

    candidatos = []
    for label in range(1, labels.max() + 1):
        reg = np.zeros(mascara.shape, dtype=np.uint8)
        reg[(labels == label) & (mascara > 0)] = 255
        cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contorno = max(cnts, key=cv2.contourArea)
        ys, xs = np.where(reg > 0)
        comp_area = float(stats[comp_labels[ys[0], xs[0]], cv2.CC_STAT_AREA])
        candidatos.append((contorno, comp_area))
    return candidatos
