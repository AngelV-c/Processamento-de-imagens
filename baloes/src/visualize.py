"""Geração de overlays de debug e salvamento de máscaras."""

import os
import cv2
import numpy as np
from typing import Dict, List

from detection import Deteccao

# Paleta BGR para cada nome de cor conhecido
_CORES_BGR: Dict[str, tuple] = {
    "vermelho": (0, 0, 220),
    "amarelo":  (0, 220, 220),
    "verde":    (0, 200, 0),
    "azul":     (220, 80, 0),
    "rosa":     (180, 60, 220),
    "ciano":    (220, 200, 0),
    "roxo":     (160, 0, 160),
}
_COR_PADRAO_BGR = (200, 200, 200)


def desenhar_deteccoes(
    imagem_bgr: np.ndarray,
    deteccoes: List[Deteccao],
) -> np.ndarray:
    """Desenha círculos, centroides e rótulos sobre uma cópia da imagem.

    Args:
        imagem_bgr: Imagem original em BGR.
        deteccoes: Lista de Deteccao a desenhar.

    Returns:
        Cópia da imagem com overlays.
    """
    saida = imagem_bgr.copy()
    for det in deteccoes:
        cor_bgr = _CORES_BGR.get(det.cor, _COR_PADRAO_BGR)
        cx, cy, raio = int(det.cx), int(det.cy), int(det.raio)

        cv2.circle(saida, (cx, cy), raio, cor_bgr, 2)
        cv2.circle(saida, (cx, cy), 4, cor_bgr, -1)

        texto = f"{det.cor} ({det.circularidade:.2f})"
        cv2.putText(
            saida, texto, (cx - raio, cy - raio - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_bgr, 1, cv2.LINE_AA,
        )
    return saida


def salvar_resultados(
    diretorio_saida: str,
    imagem_com_overlay: np.ndarray,
    mascaras: Dict[str, np.ndarray],
    prefixo: str = "",
) -> None:
    """Salva o overlay final e cada máscara de cor em diretorio_saida.

    Args:
        diretorio_saida: Pasta de destino (criada se não existir).
        imagem_com_overlay: Imagem BGR com anotações.
        mascaras: Dicionário {nome_cor: máscara_binária}.
        prefixo: Prefixo opcional para os nomes de arquivo.
    """
    os.makedirs(diretorio_saida, exist_ok=True)

    nome_overlay = f"{prefixo}overlay.jpg" if prefixo else "overlay.jpg"
    cv2.imwrite(os.path.join(diretorio_saida, nome_overlay), imagem_com_overlay)

    for nome_cor, mascara in mascaras.items():
        nome_arquivo = f"{prefixo}mascara_{nome_cor}.jpg" if prefixo else f"mascara_{nome_cor}.jpg"
        cv2.imwrite(os.path.join(diretorio_saida, nome_arquivo), mascara)
