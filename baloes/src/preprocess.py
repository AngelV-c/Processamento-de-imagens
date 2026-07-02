"""Carregamento e pré-processamento de imagem para detecção de balões."""

import cv2
import numpy as np
from typing import Optional, Tuple


def carregar_e_preprocessar(
    caminho: str,
    largura_maxima: Optional[int] = None,
    usar_clahe: bool = False,
    usar_white_balance: bool = False,
    kernel_blur: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega imagem, redimensiona, normaliza iluminação e converte para HSV.

    Args:
        caminho: Caminho para o arquivo de imagem.
        largura_maxima: Se fornecido, redimensiona mantendo proporção.
        usar_clahe: Aplica CLAHE no canal V do HSV para equalizar iluminação.
        usar_white_balance: Aplica gray-world white balance antes da conversão HSV.
        kernel_blur: Tamanho do kernel do blur gaussiano (deve ser ímpar).

    Returns:
        Tupla (imagem_hsv, imagem_bgr_original).

    Raises:
        FileNotFoundError: Se o caminho não existir ou a imagem não puder ser lida.
    """
    bgr = cv2.imread(caminho)
    if bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    if largura_maxima is not None and bgr.shape[1] > largura_maxima:
        escala = largura_maxima / bgr.shape[1]
        nova_altura = int(bgr.shape[0] * escala)
        bgr = cv2.resize(bgr, (largura_maxima, nova_altura), interpolation=cv2.INTER_AREA)

    bgr_original = bgr.copy()

    if usar_white_balance:
        bgr = _gray_world_white_balance(bgr)

    if kernel_blur > 1:
        # Garante kernel ímpar
        k = kernel_blur if kernel_blur % 2 == 1 else kernel_blur + 1
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if usar_clahe:
        hsv = _aplicar_clahe(hsv)

    return hsv, bgr_original


def normalizar_iluminacao(bgr: np.ndarray) -> np.ndarray:
    """Normalização de iluminação obrigatória do pipeline v2.

    Gray-world white balance + CLAHE no canal V. Objetivo: fazer o resto do
    pipeline ver imagens parecidas mesmo com câmeras/iluminações diferentes —
    os limiares deixam de depender da cena.
    """
    bgr_wb = _gray_world_white_balance(bgr)
    hsv = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2HSV)
    hsv = _aplicar_clahe(hsv)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _gray_world_white_balance(bgr: np.ndarray, ganho_max: float = 1.25) -> np.ndarray:
    """Gray-world white balance com ganhos limitados.

    Sem o limite, cenas dominadas por uma cor quente (paredes creme) recebem
    correções agressivas que distorcem os matizes dos objetos (rosa→violeta).
    O clamp em [1/ganho_max, ganho_max] corrige o cast sem reescrever a cena.
    """
    resultado = bgr.astype(np.float32)
    media_global = resultado.mean()
    for canal in range(3):
        media_canal = resultado[:, :, canal].mean()
        if media_canal > 0:
            ganho = media_global / media_canal
            ganho = min(max(ganho, 1.0 / ganho_max), ganho_max)
            resultado[:, :, canal] *= ganho
    return np.clip(resultado, 0, 255).astype(np.uint8)


def _aplicar_clahe(hsv: np.ndarray) -> np.ndarray:
    """Aplica CLAHE no canal V do espaço HSV."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    h, s, v = cv2.split(hsv)
    v_equalizado = clahe.apply(v)
    return cv2.merge([h, s, v_equalizado])
