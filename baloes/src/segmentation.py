"""Segmentação de cores no espaço HSV com morfologia."""

import cv2
import numpy as np
from typing import Dict, List


def _mascara_de_faixa(hsv: np.ndarray, faixas: List[dict]) -> np.ndarray:
    """Gera máscara binária combinando uma ou mais faixas HSV com OR."""
    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for faixa in faixas:
        lower = np.array(faixa["lower"], dtype=np.uint8)
        upper = np.array(faixa["upper"], dtype=np.uint8)
        mascara |= cv2.inRange(hsv, lower, upper)
    return mascara


def _morfologia(mascara: np.ndarray, kernel_tamanho: int) -> np.ndarray:
    """Aplica abertura (remove ruído) e depois fechamento (preenche buracos)."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_tamanho, kernel_tamanho)
    )
    aberta = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    fechada = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel)
    return fechada


def segmentar_cores(
    hsv: np.ndarray,
    config: dict,
) -> Dict[str, np.ndarray]:
    """Gera dicionário {nome_cor: máscara_binária} para cada cor configurada.

    Args:
        hsv: Imagem no espaço HSV (H: 0-179, S e V: 0-255).
        config: Dicionário carregado de cores.json.

    Returns:
        Dicionário mapeando nome da cor para máscara binária.
    """
    kernel_tamanho = config["deteccao"]["kernel_morfologia"]
    mascaras: Dict[str, np.ndarray] = {}

    for cor in config["cores"]:
        nome = cor["nome"]
        mascara = _mascara_de_faixa(hsv, cor["faixas"])
        mascaras[nome] = _morfologia(mascara, kernel_tamanho)

    return mascaras
