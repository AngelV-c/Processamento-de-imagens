"""Etapa 5 — Correção de perspectiva por homografia.

Por que: numa visão frontal-superior, mesas distantes aparecem menores e
comprimidas — a distância em pixels entre balões da fileira de trás encolhe,
e um único raio de agrupamento não serviria para todas as fileiras. Como as
mesas estão sobre um mesmo plano, a relação entre a imagem e a vista de cima
é uma homografia (3×3 projetiva, 8 GL, determinada por 4 correspondências).

Decisão de projeto: NÃO deformamos a imagem inteira para agrupar — apenas os
CENTROIDES dos balões são transformados com cv2.perspectiveTransform. O warp
completo (cv2.warpPerspective) existe só como recurso de debug visual.
"""

import json
import os

import cv2
import numpy as np

from tipos import Deteccao


def carregar_homografia(caminho: str) -> dict | None:
    """Carrega config/homografia.json; retorna None se não existir.

    Formato esperado:
        {"H": [[...],[...],[...]], "px_por_cm": float,
         "largura_cm": float, "altura_cm": float}
    """
    if not os.path.exists(caminho):
        return None
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    dados["H"] = np.array(dados["H"], dtype=np.float64)
    return dados


def retificar(deteccoes: list[Deteccao], H: np.ndarray) -> list[Deteccao]:
    """Preenche cx_ret/cy_ret transformando os centroides pela homografia.

    cv2.perspectiveTransform exige shape (N, 1, 2) e dtype float32.
    Modifica as detecções in-place e as retorna, por conveniência.
    """
    if not deteccoes:
        return deteccoes

    pontos = np.array(
        [[[d.cx, d.cy]] for d in deteccoes], dtype=np.float32
    )
    retificados = cv2.perspectiveTransform(pontos, H)

    for det, (x, y) in zip(deteccoes, retificados.reshape(-1, 2)):
        det.cx_ret = float(x)
        det.cy_ret = float(y)

    return deteccoes


def warp_debug(
    imagem_bgr: np.ndarray,
    H: np.ndarray,
    largura_destino: int,
    altura_destino: int,
) -> np.ndarray:
    """Warp da imagem inteira — APENAS para visualização/debug.

    O fluxo principal não usa isto: transformar só os centroides é mais
    rápido e evita artefatos de reamostragem.
    """
    return cv2.warpPerspective(imagem_bgr, H, (largura_destino, altura_destino))
