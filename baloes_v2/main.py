"""Pipeline completo de rastreio de balões por equipe (etapas 1–7).

Fluxo: carregar → normalizar → detectar (calibração local + multi-via +
score) → retificar por homografia (se calibrada) → agrupar (DBSCAN) →
relatar problemas por equipe.

Uso:
    python main.py --imagem foto.jpg --calibracao config/calibracoes/Cena.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import carregar
from deteccao import detectar_baloes
from homografia import carregar_homografia, retificar, warp_debug
from agrupamento import agrupar
from relatorio import montar_relatorio, salvar_relatorio_json, imprimir_relatorio
from visualizacao import desenhar_deteccoes, desenhar_grupos, salvar_resultados

_DIR = os.path.dirname(os.path.abspath(__file__))


def _json(caminho: str) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def executar_pipeline(
    caminho_imagem: str,
    params: dict,
    calibracao: dict,
    largura: int | None = 1200,
    caminho_homografia: str | None = None,
    eps: float | None = None,
    debug: bool = False,
    saida: str | None = None,
) -> dict:
    """Executa as etapas 1–7; retorna deteccoes, grupos, mascaras, overlay, avisos."""
    avisos: list[str] = []

    bgr = carregar(caminho_imagem, largura_maxima=largura)
    deteccoes, mascaras = detectar_baloes(bgr, params, calibracao)

    # Etapa 5 — homografia (com degradação elegante para pixels)
    homo = carregar_homografia(caminho_homografia) if caminho_homografia else None
    ag = params["agrupamento"]
    if homo is not None:
        retificar(deteccoes, homo["H"])
        eps_usado = eps if eps is not None else ag["eps_cm"] * homo["px_por_cm"]
    else:
        avisos.append("Sem homografia calibrada — agrupando em PIXELS "
                      "(tools/calibrar_homografia.py).")
        eps_usado = eps if eps is not None else ag["eps_px_fallback"]

    # Etapa 6 — DBSCAN
    agrupar(deteccoes, eps=eps_usado, min_pts=ag["min_pts"])

    # Etapa 7 — relatório (letras de problema vêm da calibração local)
    config_cores = {"cores": [
        {"nome": nome, "problema": modelo.get("problema", "?")}
        for nome, modelo in calibracao["cores"].items()
    ]}
    grupos = montar_relatorio(deteccoes, config_cores)

    overlay = desenhar_grupos(desenhar_deteccoes(bgr, deteccoes), deteccoes, grupos)

    if saida:
        salvar_resultados(saida, overlay, mascaras if debug else None)
        salvar_relatorio_json(grupos, os.path.join(saida, "relatorio.json"))
        if debug and homo is not None:
            import cv2
            w = int(homo["largura_cm"] * homo["px_por_cm"])
            h = int(homo["altura_cm"] * homo["px_por_cm"])
            cv2.imwrite(os.path.join(saida, "vista_retificada.jpg"),
                        warp_debug(bgr, homo["H"], w, h))

    return {"deteccoes": deteccoes, "grupos": grupos, "mascaras": mascaras,
            "overlay": overlay, "avisos": avisos, "retificado": homo is not None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rastreia balões coloridos por equipe (PDI clássico, sem ML)."
    )
    parser.add_argument("--imagem", required=True)
    parser.add_argument("--calibracao",
                        default=os.path.join(_DIR, "config", "calibracao_local.json"),
                        help="Modelos de cor do local (tools/calibrar_cores.py).")
    parser.add_argument("--params", default=os.path.join(_DIR, "config", "params.json"))
    parser.add_argument("--homografia",
                        default=os.path.join(_DIR, "config", "homografia.json"))
    parser.add_argument("--saida", default=os.path.join(_DIR, "output"))
    parser.add_argument("--largura", type=int, default=1200,
                        help="Use o MESMO valor da calibração.")
    parser.add_argument("--eps", type=float, default=None,
                        help="Raio do DBSCAN (cm com homografia, px sem).")
    parser.add_argument("--debug", action="store_true",
                        help="Salva máscaras por cor e vista retificada.")
    args = parser.parse_args()

    if not os.path.exists(args.calibracao):
        sys.exit(f"Calibração não encontrada: {args.calibracao}\n"
                 "Gere com: python tools/calibrar_cores.py --imagem sua_foto.jpg")

    params = _json(args.params)
    calibracao = _json(args.calibracao)

    print(f"[1-4/7] Detectando em {args.imagem} "
          f"(calibração: {os.path.basename(args.calibracao)})")
    r = executar_pipeline(args.imagem, params, calibracao,
                          largura=args.largura,
                          caminho_homografia=args.homografia,
                          eps=args.eps, debug=args.debug, saida=args.saida)

    for aviso in r["avisos"]:
        print(f"[aviso] {aviso}")
    print(f"[5/7]   {'retificado por homografia' if r['retificado'] else 'sem retificação (pixels)'}")
    print(f"[6/7]   {len(r['grupos'])} equipe(s)")

    print("\n=== Detecções ===")
    if not r["deteccoes"]:
        print("  Nenhum balão detectado.")
    else:
        for cor, qtd in sorted(Counter(d.cor for d in r["deteccoes"]).items()):
            print(f"  {cor}: {qtd}")
        print(f"  Total: {len(r['deteccoes'])}")

    print("\n=== [7/7] Relatório por equipe ===")
    imprimir_relatorio(r["grupos"])
    print(f"\nSaída: {args.saida} (overlay.jpg, relatorio.json)")


if __name__ == "__main__":
    main()
