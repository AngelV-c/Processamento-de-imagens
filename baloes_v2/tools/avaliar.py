"""Avaliação automática contra o gabarito de todas as imagens (rec. nº 5).

Para cada imagem em config/gabaritos.json, roda o pipeline com a calibração
da cena e compara com o esperado — qualquer mudança de parâmetro mostra na
hora se regrediu em outra cena.

Quando o gabarito traz as POSIÇÕES anotadas ("baloes": [{"x","y","cor"}]),
o script casa cada detecção com o balão real mais próximo e reporta
precisão, recall e acurácia de cor — não só a contagem total.

Uso:
    python tools/avaliar.py
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DIR, "src"))

from preprocess import carregar
from deteccao import detectar_baloes


def _casar_posicoes(deteccoes, baloes_gt, raio_max: float = 30.0):
    """Casamento guloso detecção↔anotação por distância crescente.

    Returns:
        (pares, det_sem_par, gt_sem_par) — pares = [(det, gt, dist), ...]
    """
    pares_possiveis = []
    for i, det in enumerate(deteccoes):
        for j, gt in enumerate(baloes_gt):
            dist = math.hypot(det.cx - gt["x"], det.cy - gt["y"])
            limite = max(raio_max, det.raio * 1.5)
            if dist <= limite:
                pares_possiveis.append((dist, i, j))
    pares_possiveis.sort()

    usados_det, usados_gt, pares = set(), set(), []
    for dist, i, j in pares_possiveis:
        if i in usados_det or j in usados_gt:
            continue
        usados_det.add(i)
        usados_gt.add(j)
        pares.append((deteccoes[i], baloes_gt[j], dist))

    det_sem_par = [d for i, d in enumerate(deteccoes) if i not in usados_det]
    gt_sem_par = [g for j, g in enumerate(baloes_gt) if j not in usados_gt]
    return pares, det_sem_par, gt_sem_par


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

        # Precisão/recall por posição, quando o gabarito traz anotações
        baloes_gt = esperado.get("baloes")
        if baloes_gt:
            pares, det_livres, gt_livres = _casar_posicoes(deteccoes, baloes_gt)
            precisao = len(pares) / len(deteccoes) if deteccoes else 0.0
            recall = len(pares) / len(baloes_gt)
            cor_ok = sum(1 for det, gt, _ in pares if det.cor == gt.get("cor", det.cor))
            acc_cor = cor_ok / len(pares) if pares else 0.0
            parcial = " (anotação parcial)" if esperado.get("parcial") else ""
            print(f"      posições{parcial}: precisão {precisao:.0%} "
                  f"({len(pares)}/{len(deteccoes)} detecções casam), "
                  f"recall {recall:.0%} ({len(pares)}/{len(baloes_gt)} anotados), "
                  f"cor correta {acc_cor:.0%}")
            for det in det_livres:
                print(f"        FP? {det.cor} em ({det.cx:.0f},{det.cy:.0f}) via {det.via}")
            for gt in gt_livres:
                print(f"        perdido: {gt.get('cor','?')} em ({gt['x']},{gt['y']})")

    if total_esp:
        print(f"\n  Global: {total_det}/{total_esp} ({total_det / total_esp:.0%})")


if __name__ == "__main__":
    main()
