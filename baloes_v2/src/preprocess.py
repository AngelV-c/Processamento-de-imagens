"""Carregamento e normalização de iluminação (recomendação nº 2).

A normalização é OBRIGATÓRIA no pipeline: gray-world com ganhos limitados
corrige o cast da câmera sem reescrever a cena (sem o limite, uma sala de
paredes creme recebe correção agressiva que distorce matizes — rosa vira
violeta), e o CLAHE no canal V equaliza a iluminação local. Depois dela, os
modelos calibrados e os limiares relativos valem para câmeras diferentes.
"""

import cv2
import numpy as np


def carregar(caminho: str, largura_maxima: int | None = 1200) -> np.ndarray:
    """Carrega a imagem BGR, redimensionando se exceder largura_maxima.

    Raises:
        FileNotFoundError: Se a imagem não puder ser lida.
    """
    bgr = cv2.imread(caminho)
    if bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")
    if largura_maxima is not None and bgr.shape[1] > largura_maxima:
        escala = largura_maxima / bgr.shape[1]
        bgr = cv2.resize(bgr, (largura_maxima, int(bgr.shape[0] * escala)),
                         interpolation=cv2.INTER_AREA)
    return bgr


def normalizar_iluminacao(bgr: np.ndarray, ganho_max: float = 1.25) -> np.ndarray:
    """Gray-world com ganhos limitados + CLAHE no V."""
    resultado = bgr.astype(np.float32)
    media_global = resultado.mean()
    for canal in range(3):
        media_canal = resultado[:, :, canal].mean()
        if media_canal > 0:
            ganho = media_global / media_canal
            ganho = min(max(ganho, 1.0 / ganho_max), ganho_max)
            resultado[:, :, canal] *= ganho
    bgr_wb = np.clip(resultado, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv = cv2.merge([h, s, clahe.apply(v)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
