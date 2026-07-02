"""Etapa 5 — Correção de perspectiva por homografia.

Mesas distantes aparecem menores e comprimidas; como todas estão sobre um
mesmo plano, a relação imagem→vista-de-cima é uma homografia (3×3 projetiva,
determinada por 4 correspondências). Transformamos apenas os CENTROIDES
(cv2.perspectiveTransform) — mais rápido e sem artefatos de reamostragem;
o warp completo existe só para debug visual.
"""

import json
import os

import cv2
import numpy as np

from tipos import Deteccao


def carregar_homografia(caminho: str) -> dict | None:
    """Carrega homografia.json; None se não existir (pipeline degrada p/ pixels)."""
    if not os.path.exists(caminho):
        return None
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    dados["H"] = np.array(dados["H"], dtype=np.float64)
    return dados


def retificar(deteccoes: list[Deteccao], H: np.ndarray) -> list[Deteccao]:
    """Preenche cx_ret/cy_ret via perspectiveTransform (shape (N,1,2) float32)."""
    if not deteccoes:
        return deteccoes
    pontos = np.array([[[d.cx, d.cy]] for d in deteccoes], dtype=np.float32)
    for det, (x, y) in zip(deteccoes, cv2.perspectiveTransform(pontos, H).reshape(-1, 2)):
        det.cx_ret = float(x)
        det.cy_ret = float(y)
    return deteccoes


def warp_debug(imagem_bgr: np.ndarray, H: np.ndarray,
               largura: int, altura: int) -> np.ndarray:
    """Warp da imagem inteira — APENAS para visualização/debug."""
    return cv2.warpPerspective(imagem_bgr, H, (largura, altura))
