"""Diagnóstico do eps do DBSCAN: curva de distâncias ao vizinho mais próximo.

A premissa de separação entre equipes aparece como um SALTO na curva; o eps
ideal fica dentro dele.

Uso:
    python tools/diagnostico_eps.py --imagem foto.jpg \
        --calibracao config/calibracoes/Cena.json
"""

import argparse
import json
import os
import sys

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import carregar
from deteccao import detectar_baloes
from homografia import carregar_homografia, retificar
from agrupamento import distancias_vizinho_mais_proximo


def main() -> None:
    parser = argparse.ArgumentParser(description="Sugere o eps do DBSCAN.")
    parser.add_argument("--imagem", required=True)
    parser.add_argument("--calibracao",
                        default=os.path.join(_DIR, "config", "calibracao_local.json"))
    parser.add_argument("--params", default=os.path.join(_DIR, "config", "params.json"))
    parser.add_argument("--homografia",
                        default=os.path.join(_DIR, "config", "homografia.json"))
    parser.add_argument("--largura", type=int, default=1200)
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    with open(args.calibracao, encoding="utf-8") as f:
        calibracao = json.load(f)

    bgr = carregar(args.imagem, largura_maxima=args.largura)
    deteccoes, _ = detectar_baloes(bgr, params, calibracao)

    homo = carregar_homografia(args.homografia)
    unidade = "px"
    if homo is not None:
        retificar(deteccoes, homo["H"])
        unidade = f"px retificados ({homo['px_por_cm']} px/cm)"
    else:
        print("[aviso] Sem homografia — distâncias em pixels da imagem.")

    dists = distancias_vizinho_mais_proximo(deteccoes)
    if len(dists) == 0:
        sys.exit("Menos de 2 balões detectados.")

    print(f"\n{len(deteccoes)} balões — distância ao vizinho mais próximo ({unidade}):")
    for i, d in enumerate(dists):
        print(f"  {i + 1:3d}  {d:8.1f}  {'#' * int(d / dists.max() * 50)}")

    saltos = dists[1:] - dists[:-1]
    if len(saltos) > 0 and saltos.max() > 0:
        idx = int(saltos.argmax())
        sugestao = (dists[idx] + dists[idx + 1]) / 2
        print(f"\nMaior salto entre {dists[idx]:.1f} e {dists[idx + 1]:.1f} → "
              f"eps sugerido ≈ {sugestao:.1f} {unidade}")
        if homo is not None:
            print(f"Em cm: eps ≈ {sugestao / homo['px_por_cm']:.1f} cm")


if __name__ == "__main__":
    main()
