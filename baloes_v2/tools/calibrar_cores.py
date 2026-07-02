"""Calibração de cores por clique: 1+ balão de cada cor → calibração local.

O fluxo de 2 minutos por evento: com a câmera na posição definitiva, clique
no centro de um balão de cada cor (de preferência um perto E um longe da
mesma cor — as amostras são fundidas). O modelo nasce da própria cena.

Interativo:
    python tools/calibrar_cores.py --imagem foto_local.jpg --largura 1200

Headless (sem janela):
    python tools/calibrar_cores.py --imagem foto_local.jpg --largura 1200 \
        --amostra "rosa:370,138:A" --amostra "rosa:580,248:A" \
        --amostra "branco:357,207:B"

IMPORTANTE: use o MESMO --largura ao rodar o pipeline — as coordenadas dos
cliques são relativas à imagem redimensionada.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import carregar, normalizar_iluminacao
from calibracao import pixels_do_disco, modelo_de_pixels


def _coletar_cliques(bgr: np.ndarray, raio: int) -> list[tuple[str, int, int, str]]:
    """Modo interativo: clique → nome da cor + letra do problema no terminal."""
    amostras: list[tuple[str, int, int, str]] = []
    exibicao = bgr.copy()
    cliques: list[tuple[int, int]] = []

    def _on_mouse(evento, x, y, _flags, _param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            cliques.append((x, y))

    cv2.namedWindow("calibracao", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibracao", _on_mouse)
    print("Clique no CENTRO de um balão de cada cor (pode repetir a cor). "
          "'q' na janela encerra.")

    while True:
        cv2.imshow("calibracao", exibicao)
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break
        if cliques:
            x, y = cliques.pop()
            nome = input(f"  Cor do balão em ({x},{y}): ").strip().lower()
            if not nome:
                continue
            problema = input(f"  Letra do problema de '{nome}': ").strip().upper() or "?"
            amostras.append((nome, x, y, problema))
            cv2.circle(exibicao, (x, y), raio, (0, 255, 0), 2)
            cv2.putText(exibicao, nome, (x + raio, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.destroyAllWindows()
    return amostras


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra os modelos de cor da cena.")
    parser.add_argument("--imagem", required=True)
    parser.add_argument("--largura", type=int, default=1200,
                        help="Use o MESMO valor no pipeline.")
    parser.add_argument("--raio", type=int, default=8,
                        help="Raio do disco de amostragem (px).")
    parser.add_argument("--amostra", action="append", default=None,
                        metavar="COR:X,Y[:PROBLEMA]",
                        help="Headless: 'rosa:370,138:E' (repetível).")
    parser.add_argument("--saida",
                        default=os.path.join(_DIR, "config", "calibracao_local.json"))
    args = parser.parse_args()

    bgr = carregar(args.imagem, largura_maxima=args.largura)
    # Calibra na imagem NORMALIZADA — o detector também normaliza, então
    # os modelos casam exatamente com o que ele vê.
    bgr = normalizar_iluminacao(bgr)

    if args.amostra:
        amostras = []
        for spec in args.amostra:
            partes = spec.split(":")
            nome = partes[0].strip().lower()
            x, y = (int(v) for v in partes[1].split(","))
            problema = partes[2].strip().upper() if len(partes) > 2 else "?"
            amostras.append((nome, x, y, problema))
    else:
        amostras = _coletar_cliques(bgr, args.raio)

    if not amostras:
        sys.exit("Nenhuma amostra coletada.")

    # Amostras da mesma cor são fundidas
    por_cor: dict[str, dict] = {}
    for nome, x, y, problema in amostras:
        h, s, v, pix_lab = pixels_do_disco(bgr, x, y, args.raio)
        d = por_cor.setdefault(nome, {"h": [], "s": [], "v": [], "lab": [],
                                      "problema": problema, "n_cliques": 0})
        d["h"].append(h); d["s"].append(s); d["v"].append(v); d["lab"].append(pix_lab)
        d["n_cliques"] += 1

    cores = {}
    for nome, d in por_cor.items():
        modelo = modelo_de_pixels(np.concatenate(d["h"]), np.concatenate(d["s"]),
                                  np.concatenate(d["v"]), np.concatenate(d["lab"]))
        modelo["problema"] = d["problema"]
        modelo["n_cliques"] = d["n_cliques"]
        cores[nome] = modelo
        tipo = "ACROMÁTICA (L + a-b)" if modelo["acromatica"] else "cromática (matiz)"
        print(f"  {nome} ({d['n_cliques']} clique(s)): "
              f"H={modelo['h_mediana']}±{modelo['h_tolerancia']} "
              f"S_p50={modelo['s_p50']:.0f} — {tipo}")

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump({"imagem_referencia": os.path.basename(args.imagem),
                   "largura_calibracao": args.largura,
                   "raio_amostra": args.raio,
                   "cores": cores}, f, ensure_ascii=False, indent=2)
    print(f"\nCalibração salva em {args.saida} ({len(cores)} cores).")


if __name__ == "__main__":
    main()
