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


def normalizar_iluminacao(bgr: np.ndarray, ganho_max: float = 1.25,
                          p: float = 6.0) -> np.ndarray:
    """Shades-of-Gray (norma de Minkowski p=6) com ganhos limitados + CLAHE no V.

    Shades-of-Gray (Finlayson & Trezzi, 2004) generaliza o gray-world: em vez
    da média simples (p=1), estima o iluminante pela norma-p de cada canal.
    Com p=6 os pixels claros pesam mais — a estimativa aproxima o white-patch
    sem a fragilidade dele a um único pixel estourado. Na literatura de
    constância de cor supera o gray-world de forma consistente.

    O clamp de ganhos continua: cenas dominadas por uma cor quente não devem
    receber correção agressiva que distorça matizes (rosa→violeta).
    """
    resultado = bgr.astype(np.float32)
    normas = [float(np.power(np.power(resultado[:, :, c] / 255.0, p).mean(), 1.0 / p))
              for c in range(3)]
    norma_media = sum(normas) / 3.0
    for canal in range(3):
        if normas[canal] > 1e-6:
            ganho = norma_media / normas[canal]
            ganho = min(max(ganho, 1.0 / ganho_max), ganho_max)
            resultado[:, :, canal] *= ganho
    bgr_wb = np.clip(resultado, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(bgr_wb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv = cv2.merge([h, s, clahe.apply(v)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
