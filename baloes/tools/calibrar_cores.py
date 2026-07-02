"""Calibração de cores por clique: 1 balão de cada cor → calibracao_local.json.

Este é o fluxo de calibração de 2 minutos por evento: com a câmera na posição
definitiva, o operador clica no centro de UM balão de cada cor presente no
local. O sistema amostra um disco ao redor de cada clique e constrói o modelo
da cor a partir da própria cena:

  - mediana circular do matiz (H) + dispersão → faixa de matiz
  - percentis de S e V das amostras → limites relativos (nunca absolutos)
  - média e covariância em LAB → distância de Mahalanobis
  - detecção automática de cor ACROMÁTICA (branco/cinza/prata): se a
    saturação mediana da amostra é baixa, o matiz é ruído — a cor passa a
    ser modelada apenas por LAB + S baixa. É assim que balões BRANCOS
    (impossíveis em segmentação por matiz) entram no pipeline.

Uso interativo (com display):
    python tools/calibrar_cores.py --imagem foto_local.jpg --largura 1200
    # clique no balão, digite o nome da cor e a letra do problema no terminal

Uso headless:
    python tools/calibrar_cores.py --imagem foto_local.jpg --largura 1200 \
        --amostra "rosa:370,138:E" --amostra "branco:357,207:F" \
        --amostra "amarelo:52,286:B" --amostra "verde:233,293:C"

IMPORTANTE: use o MESMO --largura da calibração ao rodar o pipeline — as
coordenadas dos cliques são relativas à imagem redimensionada.
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import normalizar_iluminacao

# S mediana abaixo disso → cor sem matiz confiável (branco/cinza/prata).
# 35 e não mais: um rosa pálido com S≈45 ainda tem matiz utilizável — tratá-lo
# como acromático faz o modelo casar com QUALQUER superfície branca da cena.
SATURACAO_ACROMATICA = 35


def _mediana_circular_h(h_vals: np.ndarray) -> float:
    """Mediana do matiz respeitando a circularidade (H de 0 a 179).

    Converte para ângulos, tira o vetor médio e retorna o ângulo dele —
    correto mesmo para o vermelho, que ocupa as duas pontas da escala.
    """
    angulos = h_vals.astype(np.float64) * (2 * math.pi / 180.0)
    seno, cosseno = np.sin(angulos).mean(), np.cos(angulos).mean()
    angulo_medio = math.atan2(seno, cosseno)
    if angulo_medio < 0:
        angulo_medio += 2 * math.pi
    return angulo_medio * 180.0 / (2 * math.pi)


def _pixels_do_disco(bgr: np.ndarray, x: int, y: int, raio: int):
    """(h, s, v, lab) dos pixels num disco centrado em (x, y)."""
    mascara = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mascara, (x, y), raio, 255, -1)
    sel = mascara > 0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return (
        hsv[:, :, 0][sel].astype(np.float64),
        hsv[:, :, 1][sel].astype(np.float64),
        hsv[:, :, 2][sel].astype(np.float64),
        lab[sel],
    )


def _modelo_de_pixels(h: np.ndarray, s: np.ndarray, v: np.ndarray,
                      pix_lab: np.ndarray) -> dict:
    """Constrói o modelo de cor a partir dos pixels de TODAS as amostras da cor.

    Clicar num balão perto E num longe da mesma cor faz o modelo cobrir a
    variação real de iluminação/distância — um clique só tende a ficar
    estreito demais para balões pastel no fundo da sala.
    """

    s_p50 = float(np.percentile(s, 50))
    acromatica = s_p50 < SATURACAO_ACROMATICA

    h_med = _mediana_circular_h(h)
    # Dispersão circular do matiz: desvio dos pixels em relação à mediana.
    # O piso de 8 dá folga para balões da MESMA cor mais distantes/pálidos.
    desvio = np.minimum(np.abs(h - h_med), 180.0 - np.abs(h - h_med))
    h_tol = float(max(8.0, min(18.0, 3.0 * np.percentile(desvio, 90))))

    cov = np.cov(pix_lab.T) + np.eye(3) * 2.0

    return {
        "acromatica": bool(acromatica),
        "h_mediana": round(h_med, 1),
        "h_tolerancia": round(h_tol, 1),
        "s_p10": float(np.percentile(s, 10)),
        "s_p50": s_p50,
        "s_p90": float(np.percentile(s, 90)),
        "v_p10": float(np.percentile(v, 10)),
        "v_p50": float(np.percentile(v, 50)),
        "l_p10": float(np.percentile(pix_lab[:, 0], 10)),
        "l_p90": float(np.percentile(pix_lab[:, 0], 90)),
        "lab_media": [round(float(m), 2) for m in pix_lab.mean(axis=0)],
        "lab_cov": [[round(float(c), 3) for c in linha] for linha in cov],
        "n_pixels": int(len(h)),
    }


def _coletar_cliques(bgr: np.ndarray, raio: int) -> list[tuple[str, int, int, str]]:
    """Modo interativo: clique → nome da cor + letra do problema no terminal."""
    amostras: list[tuple[str, int, int, str]] = []
    exibicao = bgr.copy()
    clique: list[tuple[int, int]] = []

    def _on_mouse(evento, x, y, _flags, _param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            clique.append((x, y))

    cv2.namedWindow("calibracao_cores", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibracao_cores", _on_mouse)
    print("Clique no CENTRO de um balão de cada cor. 'q' na janela encerra.")

    while True:
        cv2.imshow("calibracao_cores", exibicao)
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord("q"):
            break
        if clique:
            x, y = clique.pop()
            nome = input(f"  Cor do balão em ({x},{y}): ").strip().lower()
            if not nome:
                continue
            problema = input(f"  Letra do problema para '{nome}': ").strip().upper() or "?"
            amostras.append((nome, x, y, problema))
            cv2.circle(exibicao, (x, y), raio, (0, 255, 0), 2)
            cv2.putText(exibicao, nome, (x + raio, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.destroyAllWindows()
    return amostras


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibra os modelos de cor clicando em 1 balão de cada cor."
    )
    parser.add_argument("--imagem", required=True, help="Foto do local com a câmera definitiva.")
    parser.add_argument("--largura", type=int, default=1200,
                        help="Redimensiona antes de calibrar — use o MESMO valor no pipeline.")
    parser.add_argument("--raio", type=int, default=10,
                        help="Raio do disco de amostragem ao redor do clique (px).")
    parser.add_argument(
        "--amostra", action="append", default=None, metavar="COR:X,Y[:PROBLEMA]",
        help="Modo headless: 'rosa:370,138:E' (repetível, uma por cor).",
    )
    parser.add_argument("--saida", default=os.path.join(_DIR, "config", "calibracao_local.json"))
    args = parser.parse_args()

    bgr = cv2.imread(args.imagem)
    if bgr is None:
        sys.exit(f"Não foi possível ler a imagem: {args.imagem}")
    if args.largura and bgr.shape[1] > args.largura:
        escala = args.largura / bgr.shape[1]
        bgr = cv2.resize(bgr, (args.largura, int(bgr.shape[0] * escala)),
                         interpolation=cv2.INTER_AREA)

    # A calibração é feita na imagem NORMALIZADA — o pipeline v2 também
    # normaliza antes de segmentar, então os modelos casam com o que ele vê.
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

    # Amostras da MESMA cor são fundidas: clicar num balão perto e num longe
    # faz o modelo cobrir a variação real de iluminação/distância.
    por_cor: dict[str, dict] = {}
    for nome, x, y, problema in amostras:
        h, s, v, pix_lab = _pixels_do_disco(bgr, x, y, args.raio)
        if nome not in por_cor:
            por_cor[nome] = {"h": [], "s": [], "v": [], "lab": [], "problema": problema,
                             "n_cliques": 0}
        por_cor[nome]["h"].append(h)
        por_cor[nome]["s"].append(s)
        por_cor[nome]["v"].append(v)
        por_cor[nome]["lab"].append(pix_lab)
        por_cor[nome]["n_cliques"] += 1

    cores = {}
    for nome, dados in por_cor.items():
        modelo = _modelo_de_pixels(
            np.concatenate(dados["h"]), np.concatenate(dados["s"]),
            np.concatenate(dados["v"]), np.concatenate(dados["lab"]),
        )
        modelo["problema"] = dados["problema"]
        modelo["n_cliques"] = dados["n_cliques"]
        cores[nome] = modelo
        tipo = "ACROMÁTICA (S/V+LAB)" if modelo["acromatica"] else "cromática (matiz)"
        print(f"  {nome} ({dados['n_cliques']} clique(s)): "
              f"H={modelo['h_mediana']}±{modelo['h_tolerancia']} "
              f"S_p50={modelo['s_p50']:.0f} — {tipo}")

    saida = {
        "imagem_referencia": os.path.basename(args.imagem),
        "largura_calibracao": args.largura,
        "raio_amostra": args.raio,
        "cores": cores,
    }
    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print(f"\nCalibração salva em {args.saida} ({len(cores)} cores).")


if __name__ == "__main__":
    main()
