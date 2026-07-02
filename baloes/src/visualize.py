"""Geração de overlays de debug e salvamento de máscaras."""

import os
import cv2
import numpy as np
from typing import Dict, List

from tipos import Deteccao, Grupo

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


def desenhar_grupos(
    imagem_bgr: np.ndarray,
    deteccoes: List[Deteccao],
    grupos: List[Grupo],
) -> np.ndarray:
    """Desenha o casco convexo de cada grupo com rótulo da equipe e problemas.

    O casco é calculado nas coordenadas ORIGINAIS (cx, cy) — a homografia só
    é usada para agrupar, o desenho volta para a imagem da câmera.

    Args:
        imagem_bgr: Imagem BGR (tipicamente já com as detecções desenhadas).
        deteccoes: Detecções com grupo_id preenchido.
        grupos: Relatório da etapa 7 (para rótulos de equipe/problemas).

    Returns:
        Cópia da imagem com os grupos destacados.
    """
    saida = imagem_bgr.copy()
    rotulos = {g.grupo_id: g for g in grupos}

    ids = sorted({d.grupo_id for d in deteccoes if d.grupo_id is not None and d.grupo_id >= 0})
    for gid in ids:
        membros = [d for d in deteccoes if d.grupo_id == gid]
        pontos = np.array([[int(d.cx), int(d.cy)] for d in membros], dtype=np.int32)

        if len(pontos) >= 3:
            casco = cv2.convexHull(pontos)
            cv2.polylines(saida, [casco], True, (255, 255, 255), 2, cv2.LINE_AA)
        elif len(pontos) == 2:
            cv2.line(saida, tuple(pontos[0]), tuple(pontos[1]), (255, 255, 255), 2, cv2.LINE_AA)

        # Rótulo no ponto mais alto do grupo, com margem do raio do balão
        topo = min(membros, key=lambda d: d.cy)
        grupo = rotulos.get(gid)
        if grupo is not None:
            texto = f"{grupo.equipe}: {', '.join(sorted(grupo.problemas))}"
        else:
            texto = f"grupo {gid}"
        pos = (int(topo.cx - topo.raio), int(topo.cy - topo.raio - 24))
        cv2.putText(saida, texto, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

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
