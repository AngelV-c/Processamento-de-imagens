"""Segmentação híbrida: HSV por superpixel + Mahalanobis LAB como desempate.

Estratégia:
  1. Mean-shift leve reduz ruído.
  2. SLIC divide a imagem em ~800 superpixels compactos.
  3. Para cada superpixel, conta quantos pixels passam a faixa HSV de cada cor.
     A cor com MAIOR cobertura HSV vence — rejeita empates (< fracao_min).
  4. Desempate via Mahalanobis LAB quando duas cores têm cobertura próxima.
  5. Morfologia + watershed + filtro de forma (pipeline existente).

Vantagem sobre HSV+meanshift:
  - Fronteiras dos superpixels acompanham bordas reais — máscaras mais limpas.
  - Um superpixel que cruza a borda balão/fundo é classificado pela cor dominante,
    não por cada pixel individualmente → menos ruído de borda.
  - Superpixels de fundo com poucos pixels de balão são rejeitados (fracao_min).

Vantagem sobre LAB pixel-a-pixel:
  - A cor HSV é conhecida e calibrada — não depende de amostras de calibração.
  - LAB entra só para desempate (casos ambíguos como rosa vs vermelho).
"""

import cv2
import numpy as np
from typing import Dict
from skimage.segmentation import slic
from skimage.util import img_as_float


def _distancia_mahalanobis_batch(cores: np.ndarray, media: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = cores - media
    return np.sqrt(np.maximum((diff @ cov_inv * diff).sum(axis=1), 0))


def segmentar_cores_superpixel(
    bgr: np.ndarray,
    config: dict,
    calibracao: dict,
    n_segmentos: int = 800,
    compacidade: float = 15.0,
    fracao_min: float = 0.20,
    margem_empate: float = 0.10,
    kernel_tam: int | None = None,
) -> Dict[str, np.ndarray]:
    """Segmenta cores: cada superpixel vai para a cor com maior cobertura HSV.

    Args:
        bgr: Imagem original BGR.
        config: cores.json (faixas HSV + morfologia).
        calibracao: calibracao_lab.json (distribuições LAB — usado no desempate).
        n_segmentos: Número de superpixels SLIC (600–1200).
        compacidade: Equilíbrio forma/cor no SLIC.
        fracao_min: Fração mínima de pixels HSV para aceitar a classificação.
            Superpixel com <20% de pixels na cor vencedora = rejeitado (fundo).
        margem_empate: Se 1ª e 2ª coberturas diferem menos que isso,
            usa Mahalanobis LAB para desempatar.
        kernel_tam: Kernel morfológico (None = usa config).

    Returns:
        {nome_cor: máscara_binária} compatível com detectar_baloes_watershed().
    """
    bgr_ms = cv2.pyrMeanShiftFiltering(bgr, sp=7, sr=12)

    rgb_ms = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2RGB)
    segmentos = slic(
        img_as_float(rgb_ms),
        n_segments=n_segmentos,
        compactness=compacidade,
        sigma=1,
        start_label=0,
        channel_axis=2,
    )

    hsv = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Máscaras HSV por cor
    cores_config = config["cores"]
    nomes = [c["nome"] for c in cores_config]
    mascaras_hsv = {}
    for cor in cores_config:
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for faixa in cor["faixas"]:
            m |= cv2.inRange(hsv, np.array(faixa["lower"], np.uint8),
                             np.array(faixa["upper"], np.uint8))
        mascaras_hsv[cor["nome"]] = m

    # Modelos LAB para desempate
    modelos_lab = {}
    for nome, dados in calibracao.items():
        media = np.array(dados["media"], dtype=np.float64)
        cov = np.array(dados["cov"], dtype=np.float64) + np.eye(3) * 1e-3
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(3)
        modelos_lab[nome] = (media, cov_inv)

    n_sp = segmentos.max() + 1
    classificacao = np.full(n_sp, -1, dtype=np.int32)  # -1 = não classificado

    for sp_id in range(n_sp):
        sp_mask = segmentos == sp_id
        n_total = sp_mask.sum()
        if n_total == 0:
            continue

        # Cobertura HSV de cada cor neste superpixel
        coberturas = np.array([
            (mascaras_hsv[nome][sp_mask] > 0).sum() / n_total
            for nome in nomes
        ])

        idx_top = np.argsort(coberturas)[::-1]
        melhor_frac = coberturas[idx_top[0]]

        # Rejeita se cobertura insuficiente
        if melhor_frac < fracao_min:
            continue

        # Verifica empate
        segunda_frac = coberturas[idx_top[1]] if len(idx_top) > 1 else 0.0
        if melhor_frac - segunda_frac < margem_empate and len(modelos_lab) > 0:
            # Desempata por Mahalanobis LAB
            cor_lab = lab[sp_mask].mean(axis=0).reshape(1, 3)
            candidatos = [idx_top[0], idx_top[1]]
            dists = []
            for ci in candidatos:
                nome_c = nomes[ci]
                if nome_c in modelos_lab:
                    media, cov_inv = modelos_lab[nome_c]
                    d = _distancia_mahalanobis_batch(cor_lab, media, cov_inv)[0]
                else:
                    d = np.inf
                dists.append(d)
            classificacao[sp_id] = candidatos[int(np.argmin(dists))]
        else:
            classificacao[sp_id] = idx_top[0]

    h, w = bgr.shape[:2]
    kt = kernel_tam or config["deteccao"]["kernel_morfologia"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt))

    mascaras: Dict[str, np.ndarray] = {}
    for i, nome in enumerate(nomes):
        mascara = np.zeros((h, w), dtype=np.uint8)
        for sp_id in np.where(classificacao == i)[0]:
            mascara[segmentos == sp_id] = 255
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascaras[nome] = mascara

    return mascaras
