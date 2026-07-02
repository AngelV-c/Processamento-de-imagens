"""Diagnóstico do eps do DBSCAN: curva de distâncias ao vizinho mais próximo.

Roda o pipeline de detecção numa imagem de exemplo e imprime, ordenadas, as
distâncias de cada centroide ao seu vizinho mais próximo. A premissa de
separação entre equipes aparece como um SALTO claro na curva: as distâncias
intragrupo formam o trecho baixo, as intergrupo o trecho alto. Qualquer eps
dentro do salto separa as equipes sem erro — escolha um valor no meio dele.

Uso:
    python tools/diagnostico_eps.py --imagem foto.jpg --largura 1200
"""

import argparse
import json
import os
import sys

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import carregar_e_preprocessar
from segmentation_meanshift import segmentar_cores_meanshift
from detection_watershed import detectar_baloes_watershed
from homography import carregar_homografia, retificar
from grouping import distancias_vizinho_mais_proximo


def main() -> None:
    parser = argparse.ArgumentParser(description="Sugere o eps do DBSCAN a partir de uma imagem.")
    parser.add_argument("--imagem", required=True)
    parser.add_argument("--largura", type=int, default=1200)
    parser.add_argument("--config", default=os.path.join(_DIR, "config", "cores.json"))
    parser.add_argument("--params", default=os.path.join(_DIR, "config", "params.json"))
    parser.add_argument("--homografia", default=os.path.join(_DIR, "config", "homografia.json"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    with open(args.params, encoding="utf-8") as f:
        config.update(json.load(f))

    _, bgr = carregar_e_preprocessar(args.imagem, largura_maxima=args.largura)
    mascaras = segmentar_cores_meanshift(bgr, config)
    deteccoes = detectar_baloes_watershed(bgr, mascaras, config)

    homo = carregar_homografia(args.homografia)
    unidade = "px"
    if homo is not None:
        retificar(deteccoes, homo["H"])
        unidade = f"px retificados ({homo['px_por_cm']} px/cm)"
    else:
        print("[aviso] Sem homografia.json — distâncias em pixels da imagem.")

    dists = distancias_vizinho_mais_proximo(deteccoes)
    if len(dists) == 0:
        sys.exit("Menos de 2 balões detectados — impossível diagnosticar eps.")

    print(f"\n{len(deteccoes)} balões — distâncias ao vizinho mais próximo ({unidade}):")
    for i, d in enumerate(dists):
        barra = "#" * int(d / dists.max() * 50)
        print(f"  {i + 1:3d}  {d:8.1f}  {barra}")

    # Sugestão: maior salto relativo na curva
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
