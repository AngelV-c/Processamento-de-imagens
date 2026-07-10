"""Overlays: detecções individuais + cascos convexos por equipe."""

import os

import cv2
import numpy as np

from tipos import Deteccao, Grupo

_CORES_BGR: dict[str, tuple] = {
    "vermelho": (0, 0, 220),
    "amarelo":  (0, 220, 220),
    "verde":    (0, 200, 0),
    "azul":     (220, 80, 0),
    "rosa":     (180, 60, 220),
    "ciano":    (220, 200, 0),
    "roxo":     (160, 0, 160),
    "branco":   (255, 255, 255),
    "laranja":  (0, 130, 255),
}
_COR_PADRAO = (200, 200, 200)


def desenhar_deteccoes(imagem_bgr: np.ndarray,
                       deteccoes: list[Deteccao]) -> np.ndarray:
    """Círculo, centroide e rótulo 'cor (score)' por balão."""
    saida = imagem_bgr.copy()
    for det in deteccoes:
        cor = _CORES_BGR.get(det.cor, _COR_PADRAO)
        cx, cy, raio = int(det.cx), int(det.cy), int(det.raio)
        cv2.circle(saida, (cx, cy), raio, cor, 2)
        cv2.circle(saida, (cx, cy), 4, cor, -1)
        cv2.putText(saida, f"{det.cor} ({det.score:.2f})",
                    (cx - raio, cy - raio - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)
    return saida


def desenhar_grupos(imagem_bgr: np.ndarray, deteccoes: list[Deteccao],
                    grupos: list[Grupo]) -> np.ndarray:
    """Casco convexo de cada equipe com rótulo dos problemas resolvidos."""
    saida = imagem_bgr.copy()
    rotulos = {g.grupo_id: g for g in grupos}
    ids = sorted({d.grupo_id for d in deteccoes
                  if d.grupo_id is not None and d.grupo_id >= 0})
    for gid in ids:
        membros = [d for d in deteccoes if d.grupo_id == gid]
        pontos = np.array([[int(d.cx), int(d.cy)] for d in membros], dtype=np.int32)
        if len(pontos) >= 3:
            cv2.polylines(saida, [cv2.convexHull(pontos)], True,
                          (255, 255, 255), 2, cv2.LINE_AA)
        elif len(pontos) == 2:
            cv2.line(saida, tuple(pontos[0]), tuple(pontos[1]),
                     (255, 255, 255), 2, cv2.LINE_AA)

        topo = min(membros, key=lambda d: d.cy)
        grupo = rotulos.get(gid)
        texto = (f"{grupo.equipe}: {', '.join(sorted(grupo.problemas))}"
                 if grupo else f"grupo {gid}")
        cv2.putText(saida, texto,
                    (int(topo.cx - topo.raio), int(topo.cy - topo.raio - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return saida


def salvar_resultados(diretorio: str, overlay: np.ndarray,
                      mascaras: dict | None = None) -> None:
    """Salva overlay.jpg e, opcionalmente, cada máscara por cor."""
    os.makedirs(diretorio, exist_ok=True)
    cv2.imwrite(os.path.join(diretorio, "overlay.jpg"), overlay)
    for nome, mascara in (mascaras or {}).items():
        cv2.imwrite(os.path.join(diretorio, f"mascara_{nome}.jpg"), mascara)
