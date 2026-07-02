"""Avaliação automática contra o gabarito de todas as imagens (rec. nº 5).

Para cada imagem em config/gabaritos.json, roda o pipeline com a calibração
da cena e compara com o esperado — qualquer mudança de parâmetro mostra na
hora se regrediu em outra cena.

Uso:
    python tools/avaliar.py
"""

import argparse
import json
import os
import sys
from collections import Counter

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import carregar
from deteccao import detectar_baloes


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia contra config/gabaritos.json.")
    parser.add_argument("--largura", type=int, default=1200)
    parser.add_argument("--gabarito", default=os.path.join(_DIR, "config", "gabaritos.json"))
    parser.add_argument("--params", default=os.path.join(_DIR, "config", "params.json"))
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    with open(args.gabarito, encoding="utf-8") as f:
        gabaritos = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    print(f"Avaliação [largura={args.largura}]\n")
    total_det = total_esp = 0

    for nome_imagem, esperado in gabaritos.items():
        caminho = os.path.join(_DIR, nome_imagem)
        cam_calib = os.path.join(_DIR, esperado.get("calibracao", ""))
        if not os.path.exists(caminho):
            print(f"  {nome_imagem}: [pulada — imagem não encontrada]")
            continue
        if not os.path.exists(cam_calib):
            print(f"  {nome_imagem}: [pulada — sem calibração ({cam_calib})]")
            continue

        with open(cam_calib, encoding="utf-8") as f:
            calibracao = json.load(f)

        bgr = carregar(caminho, largura_maxima=args.largura)
        deteccoes, _ = detectar_baloes(bgr, params, calibracao)
        contagem = Counter(d.cor for d in deteccoes)

        n_det, n_esp = len(deteccoes), esperado["total"]
        total_det += n_det
        total_esp += n_esp
        detalhe = ", ".join(f"{c}:{q}" for c, q in sorted(contagem.items()))
        print(f"  {nome_imagem}: {n_det}/{n_esp} ({n_det / n_esp:.0%})  [{detalhe}]")

        for cor, qtd in sorted(esperado.get("cores", {}).items()):
            obtido = contagem.get(cor, 0)
            marca = "ok" if obtido == qtd else f"esperado {qtd}"
            print(f"      {cor}: {obtido} ({marca})")

    if total_esp:
        print(f"\n  Global: {total_det}/{total_esp} ({total_det / total_esp:.0%})")


if __name__ == "__main__":
    main()
