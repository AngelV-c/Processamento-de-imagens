"""Pipeline principal de detecção de balões (etapas 1–4)."""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import carregar_e_preprocessar
from segmentation import segmentar_cores
from segmentation_meanshift import segmentar_cores_meanshift
from segmentation_lab import segmentar_cores_lab
from detection import detectar_baloes
from detection_hough import detectar_baloes_hough
from detection_watershed import detectar_baloes_watershed
from visualize import desenhar_deteccoes, salvar_resultados

_DIR = os.path.dirname(__file__)


def _carregar_config(caminho: str) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta balões coloridos em imagem de maratona de programação."
    )
    parser.add_argument("--imagem", required=True, help="Caminho para a imagem de entrada.")
    parser.add_argument(
        "--config",
        default=os.path.join(_DIR, "config", "cores.json"),
        help="Caminho para cores.json.",
    )
    parser.add_argument(
        "--calibracao",
        default=os.path.join(_DIR, "config", "calibracao_lab.json"),
        help="Caminho para calibracao_lab.json (usado com --segmentacao lab).",
    )
    parser.add_argument(
        "--saida",
        default=os.path.join(_DIR, "output"),
        help="Diretório de saída para overlays e máscaras.",
    )
    parser.add_argument("--largura", type=int, default=None, help="Largura máxima em pixels.")
    parser.add_argument("--clahe", action="store_true", help="Habilita CLAHE no canal V.")
    parser.add_argument("--wb", action="store_true", help="Habilita gray-world white balance.")
    parser.add_argument("--blur", type=int, default=5, help="Kernel do blur gaussiano (ímpar).")
    parser.add_argument(
        "--limiar-lab",
        type=float,
        default=18.0,
        help="Distância de Mahalanobis máxima para aceitar pixel (usado com --segmentacao lab).",
    )
    parser.add_argument(
        "--segmentacao",
        choices=["hsv", "meanshift", "lab"],
        default="hsv",
        help=(
            "Método de segmentação de cor:\n"
            "  hsv       — faixas HSV fixas (padrão)\n"
            "  meanshift — mean-shift + faixas HSV\n"
            "  lab       — distância de Mahalanobis no espaço LAB (mais robusto)"
        ),
    )
    parser.add_argument(
        "--metodo",
        choices=["contorno", "watershed", "hough"],
        default="contorno",
        help=(
            "Método de detecção:\n"
            "  contorno   — contorno + circularidade na máscara (padrão)\n"
            "  watershed  — separa balões sobrepostos com watershed\n"
            "  hough      — HoughCircles + classificação de cor por amostragem"
        ),
    )
    args = parser.parse_args()

    config = _carregar_config(args.config)

    print(f"[1/4] Carregando e pré-processando: {args.imagem}")
    hsv, bgr_original = carregar_e_preprocessar(
        args.imagem,
        largura_maxima=args.largura,
        usar_clahe=args.clahe,
        usar_white_balance=args.wb,
        kernel_blur=args.blur,
    )

    print(f"[2/4] Segmentando por cor [{args.segmentacao}]...")
    if args.segmentacao == "meanshift":
        mascaras = segmentar_cores_meanshift(bgr_original, config)
    elif args.segmentacao == "lab":
        calibracao = _carregar_config(args.calibracao)
        mascaras = segmentar_cores_lab(bgr_original, config, calibracao, limiar_distancia=args.limiar_lab)
    else:
        mascaras = segmentar_cores(hsv, config)

    print(f"[3/4] Detectando balões [{args.metodo}]...")
    if args.metodo == "contorno":
        deteccoes = detectar_baloes(bgr_original, config, mascaras)
    elif args.metodo == "watershed":
        deteccoes = detectar_baloes_watershed(bgr_original, mascaras, config)
    else:
        deteccoes = detectar_baloes_hough(bgr_original, hsv, config)

    print("[4/4] Gerando overlays em:", args.saida)
    imagem_anotada = desenhar_deteccoes(bgr_original, deteccoes)
    salvar_resultados(args.saida, imagem_anotada, mascaras)

    print(f"\n=== Resumo de detecções [{args.segmentacao}+{args.metodo}] ===")
    if not deteccoes:
        print("  Nenhum balão detectado.")
    else:
        contagem = Counter(d.cor for d in deteccoes)
        for cor, qtd in sorted(contagem.items()):
            print(f"  {cor}: {qtd} balão(ões)")
        print(f"  Total: {len(deteccoes)}")

    # TODO: etapa 5 — homografia para coordenadas reais
    # TODO: etapa 6 — DBSCAN para agrupamento por mesa


if __name__ == "__main__":
    main()
