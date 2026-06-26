"""Segmentação de cores no espaço CIE L*a*b* com distância de Mahalanobis.

Vantagem sobre HSV:
  - LAB é perceptualmente uniforme: distâncias iguais = diferenças visuais iguais.
  - L (luminância) separado de a* (verde↔vermelho) e b* (azul↔amarelo).
  - Robusto a variações de iluminação sem precisar ajustar faixas manualmente.
  - Rosa e vermelho ficam bem separados no eixo a* sem truques de saturação.

Calibração:
  Executar calibrar_lab() uma vez para gerar config/calibracao_lab.json
  a partir de imagens de referência com máscaras HSV conhecidas.
"""

import json
import os
import cv2
import numpy as np
from typing import Dict


def _distancia_mahalanobis(pixels: np.ndarray, media: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """Calcula distância de Mahalanobis de cada pixel à distribuição de cor."""
    diff = pixels - media  # (N, 3)
    # d² = diff @ cov_inv @ diff.T  → diagonal
    left = diff @ cov_inv  # (N, 3)
    d2 = (left * diff).sum(axis=1)  # (N,)
    return np.sqrt(np.maximum(d2, 0))


def segmentar_cores_lab(
    bgr: np.ndarray,
    config: dict,
    calibracao: dict,
    limiar_distancia: float = 18.0,
    kernel_tam: int | None = None,
) -> Dict[str, np.ndarray]:
    """Segmenta cores classificando pixels pela distância de Mahalanobis em LAB.

    Args:
        bgr: Imagem original em BGR.
        config: Dicionário carregado de cores.json (usado para morfologia).
        calibracao: Dicionário carregado de calibracao_lab.json.
        limiar_distancia: Distância de Mahalanobis máxima para aceitar pixel.
            Menor = mais restritivo. Tipicamente 15–25.
        kernel_tam: Tamanho do kernel morfológico (None = usa config).

    Returns:
        Dicionário {nome_cor: máscara_binária} compatível com os métodos de detecção.
    """
    # Mean-shift leve para reduzir ruído de sensor sem destruir pastéis
    bgr_ms = cv2.pyrMeanShiftFiltering(bgr, sp=10, sr=15)

    lab = cv2.cvtColor(bgr_ms, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3)  # (N, 3)

    kt = kernel_tam or config["deteccao"]["kernel_morfologia"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt))

    # Pré-calcular inversas das covariâncias
    modelos = {}
    for nome, dados in calibracao.items():
        media = np.array(dados["media"], dtype=np.float32)
        cov = np.array(dados["cov"], dtype=np.float32)
        # Regularização para evitar covariância singular
        cov += np.eye(3) * 1e-4
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(3)
        modelos[nome] = (media, cov_inv)

    # Calcular distância de cada pixel a cada cor
    distancias = {}
    for nome, (media, cov_inv) in modelos.items():
        distancias[nome] = _distancia_mahalanobis(pixels, media, cov_inv)

    # Empilhar: shape (N, n_cores)
    nomes = list(distancias.keys())
    matriz = np.stack([distancias[n] for n in nomes], axis=1)  # (N, n_cores)

    # Cada pixel vai para a cor mais próxima (se dentro do limiar)
    idx_min = matriz.argmin(axis=1)       # (N,) índice da cor mais próxima
    dist_min = matriz.min(axis=1)         # (N,) distância mínima

    mascaras: Dict[str, np.ndarray] = {}
    for i, nome in enumerate(nomes):
        pertence = (idx_min == i) & (dist_min <= limiar_distancia)
        mascara = pertence.reshape(h, w).astype(np.uint8) * 255
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascaras[nome] = mascara

    return mascaras


def calibrar_lab(
    imagens_bgr: list,
    mascaras_hsv: list,
    caminho_saida: str,
    max_pixels_por_cor: int = 2000,
) -> dict:
    """Gera calibracao_lab.json a partir de imagens e suas máscaras HSV.

    Args:
        imagens_bgr: Lista de imagens BGR de referência.
        mascaras_hsv: Lista de dicts {nome_cor: máscara} correspondentes.
        caminho_saida: Caminho para salvar o JSON de calibração.
        max_pixels_por_cor: Limite de pixels por cor para evitar dominância.

    Returns:
        Dicionário de calibração {nome_cor: {media, cov}}.
    """
    amostras: Dict[str, list] = {}

    for bgr, mascaras in zip(imagens_bgr, mascaras_hsv):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        for nome, mask in mascaras.items():
            px = lab[mask > 0]
            if len(px) > 30:
                amostras.setdefault(nome, []).append(px)

    result = {}
    for nome, lista in amostras.items():
        todos = np.vstack(lista)
        if len(todos) > max_pixels_por_cor:
            idx = np.random.choice(len(todos), max_pixels_por_cor, replace=False)
            todos = todos[idx]
        media = todos.mean(axis=0)
        cov = np.cov(todos.T)
        result[nome] = {
            "media": media.tolist(),
            "cov": cov.tolist(),
            "n_amostras": int(len(todos)),
        }

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
