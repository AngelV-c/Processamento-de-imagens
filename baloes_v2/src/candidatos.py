"""Geração de candidatos por técnicas pesquisadas na literatura de PDI.

Via B — Círculos por SEGMENTOS DE ARCO (família EDCircles):
  Canny → segmentos de borda conectados → ajuste algébrico de círculo
  (mínimos quadrados de Kåsa) por segmento → validação por resíduo e
  cobertura angular. Um balão 40–70% ocluído deixa um arco de borda
  visível que ajusta num círculo com resíduo baixo — exatamente o caso
  em que nenhum blob fechado de cor existe. Diferente do HoughCircles,
  cada arco gera no máximo um círculo (sem votação global densa) e o
  resíduo dá uma medida direta de qualidade.

Via C — MSER (Maximally Stable Extremal Regions, Matas et al. 2002):
  Regiões cuja área é estável através de uma faixa de limiares de
  intensidade. Invariante a transformações monotônicas de iluminação —
  substitui o limiar de saturação por percentil, que ainda dependia da
  distribuição global da cena. Rodamos MSER em dois canais:
    - S (saturação): balões cromáticos são blobs estáveis de S alta;
    - L (luminância): balões brancos são blobs claros estáveis.
"""

import math

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Via B — arcos de borda
# ---------------------------------------------------------------------------

def _ajuste_kasa(pontos: np.ndarray) -> tuple[float, float, float] | None:
    """Ajuste algébrico de círculo (Kåsa): resolve A·[a b c]ᵀ = x²+y².

    Returns:
        (cx, cy, r) ou None se o sistema for degenerado (pontos colineares).
    """
    x, y = pontos[:, 0], pontos[:, 1]
    A = np.column_stack([x, y, np.ones(len(pontos))])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    raio2 = sol[2] + cx ** 2 + cy ** 2
    if raio2 <= 0:
        return None
    return float(cx), float(cy), float(math.sqrt(raio2))


def _validar_arco(pontos: np.ndarray, cx: float, cy: float, r: float,
                  residuo_max: float, cobertura_min_graus: float) -> bool:
    """Valida o ajuste: resíduo mediano baixo E cobertura angular suficiente."""
    dists = np.sqrt((pontos[:, 0] - cx) ** 2 + (pontos[:, 1] - cy) ** 2)
    if float(np.median(np.abs(dists - r))) > residuo_max:
        return False

    angulos = np.sort(np.degrees(np.arctan2(pontos[:, 1] - cy, pontos[:, 0] - cx)))
    gaps = np.diff(angulos)
    gap_circular = 360.0 - (angulos[-1] - angulos[0])
    maior_gap = max(float(gaps.max()) if len(gaps) else 360.0, gap_circular)
    cobertura = 360.0 - maior_gap
    return cobertura >= cobertura_min_graus


def candidatos_arcos(
    bgr_norm: np.ndarray,
    cfg: dict,
    r_min: float,
    r_max: float,
) -> list[tuple[float, float, float]]:
    """Círculos ajustados a segmentos de arco de borda → [(cx, cy, r), ...]."""
    cinza = cv2.medianBlur(cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2GRAY), 5)
    bordas = cv2.Canny(cinza, cfg.get("canny_baixo", 80), cfg.get("canny_alto", 160))

    segmentos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    comprimento_min = cfg.get("comprimento_min", 25)
    residuo_max = cfg.get("residuo_max", 2.0)
    cobertura_min = cfg.get("cobertura_min_graus", 70.0)

    circulos: list[tuple[float, float, float]] = []

    def _tentar(pontos: np.ndarray, profundidade: int = 0) -> None:
        if len(pontos) < comprimento_min:
            return
        ajuste = _ajuste_kasa(pontos)
        if ajuste is not None:
            cx, cy, r = ajuste
            if r_min <= r <= r_max and _validar_arco(
                    pontos, cx, cy, r, residuo_max, cobertura_min):
                circulos.append((cx, cy, r))
                return
        # Segmento pode misturar estruturas (borda do balão + mesa):
        # divide ao meio e tenta cada metade
        if profundidade < 2:
            meio = len(pontos) // 2
            _tentar(pontos[:meio], profundidade + 1)
            _tentar(pontos[meio:], profundidade + 1)

    for seg in segmentos:
        _tentar(seg.reshape(-1, 2).astype(np.float64))

    return circulos


# ---------------------------------------------------------------------------
# Via C — MSER
# ---------------------------------------------------------------------------

def candidatos_mser(
    bgr_norm: np.ndarray,
    hsv: np.ndarray,
    cfg: dict,
    area_min: float,
    area_max: float,
) -> list[tuple[np.ndarray, float]]:
    """Regiões MSER no canal S → [(contorno, área_da_região), ...].

    Só o canal de saturação: balões CROMÁTICOS são blobs estáveis de S alta
    contra fundo neutro, invariante a mudanças monotônicas de iluminação.
    O canal L foi testado e removido — propõe camisetas/rostos claros que
    são idênticos ao modelo branco em cor; balões brancos já são cobertos
    pela via A (máscara acromática com portões de brilho) e pela via B.
    """
    mser = cv2.MSER_create(
        delta=cfg.get("delta", 8),
        min_area=int(area_min),
        max_area=int(area_max),
        max_variation=cfg.get("max_variation", 0.5),
    )

    candidatos: list[tuple[np.ndarray, float]] = []
    regioes, _ = mser.detectRegions(hsv[:, :, 1])
    for pontos in regioes:
        if len(pontos) < 3:
            continue
        # Balões são convexos — o casco convexo é um contorno fiel e barato
        contorno = cv2.convexHull(pontos.reshape(-1, 1, 2))
        candidatos.append((contorno, float(len(pontos))))
    return candidatos
