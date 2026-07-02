"""Avaliação automática do pipeline contra o gabarito de todas as imagens.

Para cada imagem em config/gabaritos.json, roda o pipeline de detecção e
compara o resultado com o esperado. Isso permite ajustar parâmetros contra
TODAS as imagens simultaneamente, em vez de calibrar numa foto e regredir
silenciosamente nas outras.

Uso:
    python tools/avaliar.py                       # métodos padrão
    python tools/avaliar.py --segmentacao hsv --metodo contorno
"""

import argparse
import json
import os
import sys
from collections import Counter

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import carregar_e_preprocessar
from segmentation import segmentar_cores
from segmentation_meanshift import segmentar_cores_meanshift
from detection import detectar_baloes
from detection_watershed import detectar_baloes_watershed
from detection_fourier import detectar_baloes_fourier


def _detectar(imagem: str, config: dict, segmentacao: str, metodo: str, largura: int):
    hsv, bgr = carregar_e_preprocessar(imagem, largura_maxima=largura)
    if segmentacao == "meanshift":
        mascaras = segmentar_cores_meanshift(bgr, config)
    else:
        mascaras = segmentar_cores(hsv, config)

    if metodo == "watershed":
        return detectar_baloes_watershed(bgr, mascaras, config)
    if metodo == "fourier":
        return detectar_baloes_fourier(bgr, mascaras, config)
    return detectar_baloes(bgr, config, mascaras)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o pipeline contra config/gabaritos.json.")
    parser.add_argument("--segmentacao", choices=["hsv", "meanshift"], default="meanshift")
    parser.add_argument("--metodo", choices=["contorno", "watershed", "fourier"], default="watershed")
    parser.add_argument("--largura", type=int, default=1200)
    parser.add_argument("--gabarito", default=os.path.join(_DIR, "config", "gabaritos.json"))
    args = parser.parse_args()

    with open(os.path.join(_DIR, "config", "cores.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(_DIR, "config", "params.json"), encoding="utf-8") as f:
        config.update(json.load(f))
    with open(args.gabarito, encoding="utf-8") as f:
        gabaritos = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    print(f"Avaliação [{args.segmentacao}+{args.metodo}, largura={args.largura}]\n")
    total_detectado = total_esperado = 0

    for nome_imagem, esperado in gabaritos.items():
        caminho = os.path.join(_DIR, nome_imagem)
        if not os.path.exists(caminho):
            print(f"  {nome_imagem}: [pulada — arquivo não encontrado]")
            continue

        deteccoes = _detectar(caminho, config, args.segmentacao, args.metodo, args.largura)
        contagem = Counter(d.cor for d in deteccoes)

        n_det = len(deteccoes)
        n_esp = esperado["total"]
        total_detectado += n_det
        total_esperado += n_esp

        recall = n_det / n_esp if n_esp else 0.0
        detalhe = ", ".join(f"{c}:{q}" for c, q in sorted(contagem.items()))
        print(f"  {nome_imagem}: {n_det}/{n_esp} ({recall:.0%})  [{detalhe}]")

        cores_esperadas = esperado.get("cores")
        if cores_esperadas:
            for cor, qtd in sorted(cores_esperadas.items()):
                obtido = contagem.get(cor, 0)
                marca = "ok" if obtido == qtd else f"esperado {qtd}"
                print(f"      {cor}: {obtido} ({marca})")

    if total_esperado:
        print(f"\n  Global: {total_detectado}/{total_esperado} "
              f"({total_detectado / total_esperado:.0%})")


if __name__ == "__main__":
    main()
