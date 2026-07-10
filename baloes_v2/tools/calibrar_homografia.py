"""Calibração da homografia: 4 pontos sobre o plano das mesas → homografia.json.

Uso interativo (com display):
    python tools/calibrate_homografia.py --imagem foto.jpg \
        --largura-cm 400 --altura-cm 300

    Clique nos 4 cantos de um retângulo de dimensões conhecidas sobre o plano
    das mesas, na ordem: superior-esquerdo, superior-direito,
    inferior-direito, inferior-esquerdo. Tecla 'r' recomeça, 'q' cancela.

Uso headless (sem display, ex.: servidor):
    python tools/calibrate_homografia.py --imagem foto.jpg \
        --largura-cm 400 --altura-cm 300 \
        --pontos 120,340 980,320 1050,690 60,700

O retângulo de destino é gerado na escala px_por_cm (padrão 2.0), de modo
que 1 px retificado = 0.5 cm. Isso torna o eps do DBSCAN interpretável em cm:
eps_px_retificado = eps_cm * px_por_cm.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clicar_pontos(imagem) -> list[tuple[int, int]]:
    """Coleta 4 cliques do usuário numa janela OpenCV."""
    pontos: list[tuple[int, int]] = []
    exibicao = imagem.copy()

    def _on_mouse(evento, x, y, _flags, _param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pontos) < 4:
            pontos.append((x, y))
            cv2.circle(exibicao, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(exibicao, str(len(pontos)), (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.namedWindow("calibracao", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibracao", _on_mouse)

    while True:
        cv2.imshow("calibracao", exibicao)
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord("r"):
            pontos.clear()
            exibicao = imagem.copy()
        elif tecla == ord("q"):
            cv2.destroyAllWindows()
            sys.exit("Calibração cancelada.")
        elif len(pontos) == 4:
            cv2.destroyAllWindows()
            return pontos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibra a homografia do plano das mesas (4 pontos → homografia.json)."
    )
    parser.add_argument("--imagem", required=True, help="Imagem de referência do local.")
    parser.add_argument("--largura-cm", type=float, required=True,
                        help="Largura real do retângulo clicado, em cm.")
    parser.add_argument("--altura-cm", type=float, required=True,
                        help="Altura real do retângulo clicado, em cm.")
    parser.add_argument("--px-por-cm", type=float, default=2.0,
                        help="Escala do plano retificado (padrão 2 px/cm).")
    parser.add_argument(
        "--pontos", nargs=4, metavar="X,Y", default=None,
        help="4 pontos 'x,y' na ordem sup-esq, sup-dir, inf-dir, inf-esq "
             "(modo headless, sem janela).",
    )
    parser.add_argument(
        "--saida",
        default=os.path.join(_DIR, "config", "homografia.json"),
        help="Destino do JSON (padrão config/homografia.json).",
    )
    args = parser.parse_args()

    imagem = cv2.imread(args.imagem)
    if imagem is None:
        sys.exit(f"Não foi possível ler a imagem: {args.imagem}")

    if args.pontos:
        origem = [tuple(int(v) for v in p.split(",")) for p in args.pontos]
    else:
        print("Clique os 4 cantos do retângulo de referência "
              "(sup-esq, sup-dir, inf-dir, inf-esq). 'r' recomeça, 'q' cancela.")
        origem = _clicar_pontos(imagem)

    w = args.largura_cm * args.px_por_cm
    h = args.altura_cm * args.px_por_cm
    destino = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    H = cv2.getPerspectiveTransform(
        np.array(origem, dtype=np.float32),
        np.array(destino, dtype=np.float32),
    )

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(
            {
                "H": H.tolist(),
                "px_por_cm": args.px_por_cm,
                "largura_cm": args.largura_cm,
                "altura_cm": args.altura_cm,
                "pontos_origem": [list(p) for p in origem],
                "imagem_referencia": os.path.basename(args.imagem),
            },
            f, indent=2,
        )

    print(f"Homografia salva em {args.saida}")
    print(f"Escala: {args.px_por_cm} px/cm — eps do DBSCAN pode ser dado em cm.")


if __name__ == "__main__":
    main()
