"""Detecção de balões com filtro adicional por Descritores de Fourier.

Estratégia:
  1. Amostra N pontos equidistantes no contorno.
  2. Calcula sinal d(t) = distância de cada ponto ao centroide.
  3. Aplica FFT no sinal d(t) (1D, periódico).
  4. Energia nos harmônicos baixos (1–K) versus energia total:
       E_baixa = sum(|F[1..K]|²) / sum(|F[1..]|²)
     Balões têm forma quase circular → energia concentrada nos harmônicos 1–3.
     Formas irregulares (camisetas, banners) espalham energia para harmônicos altos.

Integração:
  - Usa o mesmo pipeline de watershed de detection_watershed.py.
  - O parâmetro fourier_limiar_energia_baixa em cores.json controla o filtro.
    Se ausente, o filtro não rejeita nada (compatibilidade com pipelines antigos).

Referência:
  Granlund (1972) — "Fourier Preprocessing for Hand Print Character Recognition"
"""

import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from detection import Deteccao, _metricas_forma, _forma_de_balao


def _amostrar_contorno_equidistante(contorno: np.ndarray, n: int = 64) -> np.ndarray:
    """Retorna N pontos equidistantes ao longo do contorno como array (N, 2)."""
    pts = contorno.reshape(-1, 2).astype(np.float64)
    n_pts = len(pts)
    if n_pts < 3:
        return pts

    # Comprimentos acumulados ao longo do contorno fechado
    diffs = np.diff(pts, axis=0, append=pts[:1])
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    cums = np.concatenate([[0.0], np.cumsum(dists)])
    total = cums[-1]
    if total == 0:
        return pts

    alvos = np.linspace(0, total, n, endpoint=False)
    xs = np.interp(alvos, cums, np.append(pts[:, 0], pts[0, 0]))
    ys = np.interp(alvos, cums, np.append(pts[:, 1], pts[0, 1]))
    return np.stack([xs, ys], axis=1)


def _fourier_energia_baixa(contorno: np.ndarray, n: int = 64, k_baixo: int = 4) -> float:
    """Fração da energia do sinal d(t) nos K harmônicos mais baixos (exceto DC).

    Args:
        contorno: Contorno OpenCV (N, 1, 2).
        n: Número de pontos amostrados no contorno.
        k_baixo: Número de harmônicos considerados 'baixos' (harmônicos 1..k_baixo).

    Returns:
        E_baixa = energia(harmônicos 1..k_baixo) / energia(todos exceto DC)
        Retorna 0.0 se o contorno for degenerado.
    """
    pts = _amostrar_contorno_equidistante(contorno, n)
    if len(pts) < 3:
        return 0.0

    momentos = cv2.moments(contorno)
    if momentos["m00"] == 0:
        return 0.0
    cx = momentos["m10"] / momentos["m00"]
    cy = momentos["m01"] / momentos["m00"]

    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    F = np.fft.rfft(d)
    potencia = np.abs(F) ** 2

    # Índice 0 = DC, harmônicos 1..k_baixo, resto = altos
    energia_baixa = potencia[1 : k_baixo + 1].sum()
    energia_total = potencia[1:].sum()

    if energia_total == 0:
        return 0.0
    return float(energia_baixa / energia_total)


def _fourier_de_balao(
    contorno: np.ndarray,
    limiar: float,
    n: int = 64,
    k_baixo: int = 4,
) -> bool:
    """Retorna True se o contorno tem energia concentrada nos harmônicos baixos."""
    return _fourier_energia_baixa(contorno, n, k_baixo) >= limiar


def detectar_baloes_fourier(
    imagem_bgr: np.ndarray,
    mascaras: Dict[str, np.ndarray],
    config: dict,
) -> List[Deteccao]:
    """Watershed + filtro de Descritores de Fourier.

    Usa os mesmos parâmetros de watershed e detecção do pipeline padrão,
    adicionando um filtro por energia espectral de baixa frequência.

    Args:
        imagem_bgr: Imagem original BGR.
        mascaras: {nome_cor: máscara_binária} de segmentar_cores*().
        config: cores.json com seção opcional 'fourier'.

    Returns:
        Lista de Deteccao filtrada por forma de balão + Fourier.
    """
    from detection_watershed import _watershed_mascara

    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura

    cfg = config["deteccao"]
    area_min = cfg["area_minima_relativa"] * area_total
    area_max = cfg["area_maxima_relativa"] * area_total

    ws_cfg = config.get("watershed", {})
    kernel_maximos = ws_cfg.get("kernel_maximos", 9)
    limiar_distancia = ws_cfg.get("limiar_distancia", 0.2)

    f_cfg = config.get("fourier", {})
    limiar_fourier: Optional[float] = f_cfg.get("limiar_energia_baixa", None)
    n_amostras: int = f_cfg.get("n_amostras", 64)
    k_baixo: int = f_cfg.get("k_harmonicos_baixos", 4)

    deteccoes: List[Deteccao] = []

    for nome_cor, mascara in mascaras.items():
        if cv2.countNonZero(mascara) == 0:
            continue

        labels = _watershed_mascara(mascara, kernel_maximos, limiar_distancia)
        n_labels = labels.max()

        for label in range(1, n_labels + 1):
            regiao = np.zeros(mascara.shape, dtype=np.uint8)
            regiao[(labels == label) & (mascara > 0)] = 255

            contornos, _ = cv2.findContours(
                regiao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contornos:
                continue

            contorno = max(contornos, key=cv2.contourArea)
            area = cv2.contourArea(contorno)
            if area < area_min or area > area_max:
                continue

            circularidade, solidity, aspect_ratio = _metricas_forma(contorno)
            if not _forma_de_balao(circularidade, solidity, aspect_ratio, cfg):
                continue

            # Filtro de Fourier (opcional — se não configurado, não rejeita nada)
            if limiar_fourier is not None:
                if not _fourier_de_balao(contorno, limiar_fourier, n_amostras, k_baixo):
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
