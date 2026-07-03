"""Busca em grade de parâmetros contra o gabarito de TODAS as imagens.

Em vez de ajustar um limiar olhando uma foto (e quebrar as outras), varre
combinações de valores e ranqueia pela taxa global de detecção — o ajuste
passa a ser mensurável e multi-cena por construção.

Uso:
    python tools/busca_grade.py \
        --grade '{"deteccao.score_minimo": [0.65, 0.7, 0.75, 0.8],
                  "deteccao.sat_percentil": [80, 85, 90]}'

Cada chave usa notação pontuada de params.json; os valores são listas.
"""

import argparse
import copy
import itertools
import json
import os
import sys

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import carregar
from deteccao import detectar_baloes


def _aplicar(params: dict, chave: str, valor) -> None:
    alvo = params
    partes = chave.split(".")
    for parte in partes[:-1]:
        alvo = alvo.setdefault(parte, {})
    alvo[partes[-1]] = valor


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca em grade contra gabaritos.json.")
    parser.add_argument("--grade", required=True,
                        help='JSON {"chave.pontuada": [valores], ...}')
    parser.add_argument("--largura", type=int, default=1200)
    parser.add_argument("--params", default=os.path.join(_DIR, "config", "params.json"))
    parser.add_argument("--gabarito", default=os.path.join(_DIR, "config", "gabaritos.json"))
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params_base = json.load(f)
    with open(args.gabarito, encoding="utf-8") as f:
        gabaritos = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    grade = json.loads(args.grade)

    # Carrega imagens e calibrações uma vez
    cenas = []
    for nome, esperado in gabaritos.items():
        caminho = os.path.join(_DIR, nome)
        cam_calib = os.path.join(_DIR, esperado.get("calibracao", ""))
        if not (os.path.exists(caminho) and os.path.exists(cam_calib)):
            continue
        with open(cam_calib, encoding="utf-8") as f:
            calibracao = json.load(f)
        cenas.append((nome, carregar(caminho, args.largura), calibracao, esperado["total"]))

    if not cenas:
        sys.exit("Nenhuma cena avaliável (imagem + calibração).")

    chaves = list(grade.keys())
    combos = list(itertools.product(*(grade[c] for c in chaves)))
    print(f"{len(combos)} combinações × {len(cenas)} cenas\n")

    resultados = []
    for combo in combos:
        params = copy.deepcopy(params_base)
        for chave, valor in zip(chaves, combo):
            _aplicar(params, chave, valor)

        total_det = total_esp = 0
        excesso = 0  # detecções acima do esperado contam contra
        for nome, bgr, calibracao, esperado_total in cenas:
            deteccoes, _ = detectar_baloes(bgr, params, calibracao)
            total_det += min(len(deteccoes), esperado_total)
            excesso += max(0, len(deteccoes) - esperado_total)
            total_esp += esperado_total

        taxa = total_det / total_esp
        resultados.append((taxa, excesso, combo))
        rotulo = ", ".join(f"{c}={v}" for c, v in zip(chaves, combo))
        print(f"  {taxa:.0%} (excesso {excesso})  {rotulo}")

    resultados.sort(key=lambda r: (-r[0], r[1]))
    taxa, excesso, combo = resultados[0]
    print(f"\nMelhor: {taxa:.0%} (excesso {excesso}) → "
          + ", ".join(f"{c}={v}" for c, v in zip(chaves, combo)))


if __name__ == "__main__":
    main()
