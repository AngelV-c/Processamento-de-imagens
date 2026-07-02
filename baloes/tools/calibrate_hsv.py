"""Ferramenta interativa de calibração de faixas HSV para cores de balão."""

import argparse
import json
import sys

import cv2
import numpy as np


_JANELA_CONTROLES = "Calibracao HSV - Controles"
_JANELA_MASCARA = "Mascara resultante"
_JANELA_ORIGINAL = "Imagem original"


def _criar_trackbars(janela: str) -> None:
    cv2.createTrackbar("H min", janela, 0,   179, lambda _: None)
    cv2.createTrackbar("H max", janela, 179, 179, lambda _: None)
    cv2.createTrackbar("S min", janela, 0,   255, lambda _: None)
    cv2.createTrackbar("S max", janela, 255, 255, lambda _: None)
    cv2.createTrackbar("V min", janela, 0,   255, lambda _: None)
    cv2.createTrackbar("V max", janela, 255, 255, lambda _: None)


def _ler_trackbars(janela: str) -> tuple:
    h_min = cv2.getTrackbarPos("H min", janela)
    h_max = cv2.getTrackbarPos("H max", janela)
    s_min = cv2.getTrackbarPos("S min", janela)
    s_max = cv2.getTrackbarPos("S max", janela)
    v_min = cv2.getTrackbarPos("V min", janela)
    v_max = cv2.getTrackbarPos("V max", janela)
    return h_min, h_max, s_min, s_max, v_min, v_max


def _faixa_para_json(h_min, h_max, s_min, s_max, v_min, v_max) -> dict:
    return {
        "lower": [h_min, s_min, v_min],
        "upper": [h_max, s_max, v_max],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibração interativa de faixas HSV. "
            "Pressione 's' para salvar faixa, 'q' para sair e imprimir JSON."
        )
    )
    parser.add_argument("--imagem", required=True, help="Imagem de exemplo para calibrar.")
    parser.add_argument("--cor", default="nova_cor", help="Nome da cor sendo calibrada.")
    args = parser.parse_args()

    bgr = cv2.imread(args.imagem)
    if bgr is None:
        print(f"Erro: não foi possível carregar '{args.imagem}'", file=sys.stderr)
        sys.exit(1)

    # Redimensiona para caber na tela, mantendo proporção
    altura, largura = bgr.shape[:2]
    if largura > 1000:
        escala = 1000 / largura
        bgr = cv2.resize(bgr, (1000, int(altura * escala)))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Janela de controles precisa de uma imagem mínima para hospedar trackbars
    painel_ctrl = np.zeros((10, 400, 3), dtype=np.uint8)
    cv2.namedWindow(_JANELA_CONTROLES)
    cv2.imshow(_JANELA_CONTROLES, painel_ctrl)
    _criar_trackbars(_JANELA_CONTROLES)

    cv2.namedWindow(_JANELA_ORIGINAL)
    cv2.imshow(_JANELA_ORIGINAL, bgr)

    cv2.namedWindow(_JANELA_MASCARA)

    faixas_salvas: list[dict] = []

    print("=" * 50)
    print(f"Calibrando cor: {args.cor!r}")
    print("Teclas:")
    print("  s  — salva a faixa atual (permite capturar 2ª faixa para vermelho)")
    print("  q  — sai e imprime JSON")
    print("=" * 50)

    while True:
        vals = _ler_trackbars(_JANELA_CONTROLES)
        h_min, h_max, s_min, s_max, v_min, v_max = vals

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mascara = cv2.inRange(hsv, lower, upper)

        mascara_colorida = cv2.bitwise_and(bgr, bgr, mask=mascara)
        cv2.imshow(_JANELA_MASCARA, mascara_colorida)

        tecla = cv2.waitKey(30) & 0xFF

        if tecla == ord("s"):
            faixa = _faixa_para_json(*vals)
            faixas_salvas.append(faixa)
            print(f"\nFaixa {len(faixas_salvas)} salva: {faixa}")
            if len(faixas_salvas) == 1:
                print("Ajuste os trackbars para capturar a 2ª faixa (vermelho) ou pressione 'q'.")

        elif tecla == ord("q"):
            break

    cv2.destroyAllWindows()

    if not faixas_salvas:
        # Usa os valores atuais dos trackbars se nenhuma foi salva explicitamente
        faixas_salvas.append(_faixa_para_json(*_ler_trackbars(_JANELA_CONTROLES)))

    entrada_json = {
        "nome": args.cor,
        "faixas": faixas_salvas,
    }

    print("\n=== Cole o trecho abaixo em config/cores.json (dentro de 'cores') ===")
    print(json.dumps(entrada_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
