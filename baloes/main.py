"""Pipeline completo de rastreio de balões (etapas 1–7).

Fluxo: carregar → pré-processar → segmentar por cor → detectar balões →
retificar por homografia (se calibrada) → agrupar por equipe (DBSCAN) →
relatar problemas por equipe.

Degradação elegante: sem config/homografia.json o pipeline avisa e agrupa
em pixels, usando agrupamento.eps_px_fallback de params.json.
"""

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
from segmentation_superpixel import segmentar_cores_superpixel
from detection import detectar_baloes
from detection_hough import detectar_baloes_hough
from detection_watershed import detectar_baloes_watershed
from detection_fourier import detectar_baloes_fourier
from detection_shape_first import detectar_baloes_shape_first
from detection_v2 import detectar_baloes_v2
from homography import carregar_homografia, retificar, warp_debug
from grouping import agrupar
from reporting import montar_relatorio, salvar_relatorio_json, imprimir_relatorio
from visualize import desenhar_deteccoes, desenhar_grupos, salvar_resultados

_DIR = os.path.dirname(os.path.abspath(__file__))


def _carregar_json(caminho: str) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def montar_config(caminho_cores: str, caminho_params: str) -> dict:
    """Une cores.json e params.json num único dicionário de configuração."""
    config = _carregar_json(caminho_cores)
    config.update(_carregar_json(caminho_params))
    return config


def executar_pipeline(
    caminho_imagem: str,
    config: dict,
    segmentacao: str = "meanshift",
    metodo: str = "watershed",
    largura: int | None = 1200,
    usar_clahe: bool = False,
    usar_wb: bool = False,
    kernel_blur: int = 5,
    caminho_homografia: str | None = None,
    eps: float | None = None,
    calibracao_lab: dict | None = None,
    calibracao_local: dict | None = None,
    debug: bool = False,
    saida: str | None = None,
) -> dict:
    """Executa as etapas 1–7 e retorna um dicionário com todos os artefatos.

    Retorno: {"deteccoes", "grupos", "mascaras", "overlay", "avisos",
              "retificado", "bgr_original"}.
    Usado tanto pela CLI quanto pela interface web.
    """
    avisos: list[str] = []

    # Etapas 1–2: carga e pré-processamento
    hsv, bgr_original = carregar_e_preprocessar(
        caminho_imagem,
        largura_maxima=largura,
        usar_clahe=usar_clahe,
        usar_white_balance=usar_wb,
        kernel_blur=kernel_blur,
    )

    # Pipeline v2: segmentação calibrada + multi-via + score num passo só
    if metodo == "v2":
        if not calibracao_local:
            raise ValueError(
                "O método v2 exige config/calibracao_local.json — "
                "gere com tools/calibrar_cores.py."
            )
        deteccoes, mascaras = detectar_baloes_v2(bgr_original, config, calibracao_local)
        # No v2 as cores (e letras de problema) vêm da calibração local
        config_v2 = dict(config)
        config_v2["cores"] = [
            {"nome": nome, "problema": modelo.get("problema", "?")}
            for nome, modelo in calibracao_local["cores"].items()
        ]
        return _finalizar_pipeline(
            bgr_original, deteccoes, mascaras, config_v2, avisos,
            caminho_homografia, eps, debug, saida,
        )

    # Etapa 3: segmentação por cor
    if segmentacao == "meanshift":
        mascaras = segmentar_cores_meanshift(bgr_original, config)
    elif segmentacao == "lab":
        mascaras = segmentar_cores_lab(bgr_original, config, calibracao_lab or {})
    elif segmentacao == "superpixel":
        mascaras = segmentar_cores_superpixel(bgr_original, config, calibracao_lab or {})
    else:
        mascaras = segmentar_cores(hsv, config)

    # Etapa 4: detecção
    if metodo == "watershed":
        deteccoes = detectar_baloes_watershed(bgr_original, mascaras, config)
    elif metodo == "fourier":
        deteccoes = detectar_baloes_fourier(bgr_original, mascaras, config)
    elif metodo == "hough":
        deteccoes = detectar_baloes_hough(bgr_original, hsv, config)
    elif metodo == "shape_first":
        deteccoes = detectar_baloes_shape_first(bgr_original, config)
    else:
        deteccoes = detectar_baloes(bgr_original, config, mascaras)

    return _finalizar_pipeline(
        bgr_original, deteccoes, mascaras, config, avisos,
        caminho_homografia, eps, debug, saida,
    )


def _finalizar_pipeline(
    bgr_original,
    deteccoes,
    mascaras,
    config: dict,
    avisos: list[str],
    caminho_homografia: str | None,
    eps: float | None,
    debug: bool,
    saida: str | None,
) -> dict:
    """Etapas 5–7 + visualização/saída, compartilhadas por todos os métodos."""
    # Etapa 5: retificação por homografia (opcional, com fallback)
    homo = None
    if caminho_homografia:
        homo = carregar_homografia(caminho_homografia)
    ag_cfg = config.get("agrupamento", {})
    if homo is not None:
        retificar(deteccoes, homo["H"])
        eps_usado = eps if eps is not None else ag_cfg.get("eps_cm", 60.0) * homo["px_por_cm"]
    else:
        avisos.append(
            "Sem homografia calibrada — agrupando em PIXELS. O eps pode não "
            "servir para todas as fileiras; calibre com tools/calibrate_homografia.py."
        )
        eps_usado = eps if eps is not None else ag_cfg.get("eps_px_fallback", 100.0)

    # Etapa 6: agrupamento por equipe
    min_pts = ag_cfg.get("min_pts", 1)
    agrupar(deteccoes, eps=eps_usado, min_pts=min_pts)

    # Etapa 7: relatório
    grupos = montar_relatorio(deteccoes, config)

    # Visualização
    overlay = desenhar_deteccoes(bgr_original, deteccoes)
    overlay = desenhar_grupos(overlay, deteccoes, grupos)

    resultado = {
        "deteccoes": deteccoes,
        "grupos": grupos,
        "mascaras": mascaras,
        "overlay": overlay,
        "avisos": avisos,
        "retificado": homo is not None,
        "bgr_original": bgr_original,
    }

    if saida:
        salvar_resultados(saida, overlay, mascaras if debug else {})
        salvar_relatorio_json(grupos, os.path.join(saida, "relatorio.json"))
        if debug and homo is not None:
            import cv2
            w = int(homo["largura_cm"] * homo["px_por_cm"])
            h = int(homo["altura_cm"] * homo["px_por_cm"])
            vista = warp_debug(bgr_original, homo["H"], w, h)
            cv2.imwrite(os.path.join(saida, "vista_retificada.jpg"), vista)

    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rastreia balões coloridos por equipe em maratona de programação."
    )
    parser.add_argument("--imagem", required=True, help="Caminho para a imagem de entrada.")
    parser.add_argument("--config", default=os.path.join(_DIR, "config", "cores.json"),
                        help="Caminho para cores.json.")
    parser.add_argument("--params", default=os.path.join(_DIR, "config", "params.json"),
                        help="Caminho para params.json.")
    parser.add_argument("--homografia", default=os.path.join(_DIR, "config", "homografia.json"),
                        help="Caminho para homografia.json (opcional).")
    parser.add_argument("--calibracao", default=os.path.join(_DIR, "config", "calibracao_lab.json"),
                        help="Calibração LAB (para --segmentacao lab/superpixel).")
    parser.add_argument("--calibracao-local",
                        default=os.path.join(_DIR, "config", "calibracao_local.json"),
                        help="Modelos de cor do local (para --metodo v2; tools/calibrar_cores.py).")
    parser.add_argument("--saida", default=os.path.join(_DIR, "output"),
                        help="Diretório de saída.")
    parser.add_argument("--largura", type=int, default=1200, help="Largura máxima em pixels.")
    parser.add_argument("--clahe", action="store_true", help="Habilita CLAHE no canal V.")
    parser.add_argument("--wb", action="store_true", help="Habilita gray-world white balance.")
    parser.add_argument("--blur", type=int, default=5, help="Kernel do blur gaussiano (ímpar).")
    parser.add_argument("--eps", type=float, default=None,
                        help="Sobrescreve o eps do DBSCAN (cm com homografia, px sem).")
    parser.add_argument("--debug", action="store_true",
                        help="Salva máscaras por cor e vista retificada.")
    parser.add_argument(
        "--segmentacao",
        choices=["hsv", "meanshift", "lab", "superpixel"],
        default=None,
        help="Método de segmentação (padrão: segmentacao.metodo_padrao de params.json).",
    )
    parser.add_argument(
        "--metodo",
        choices=["contorno", "watershed", "hough", "fourier", "shape_first", "v2"],
        default="watershed",
        help="Método de detecção (padrão: watershed; v2 = calibração local + multi-via + score).",
    )
    args = parser.parse_args()

    config = montar_config(args.config, args.params)
    segmentacao = args.segmentacao or config.get("segmentacao", {}).get("metodo_padrao", "meanshift")

    calibracao_lab = None
    if segmentacao in ("lab", "superpixel") and os.path.exists(args.calibracao):
        calibracao_lab = _carregar_json(args.calibracao)

    calibracao_local = None
    if args.metodo == "v2":
        if not os.path.exists(args.calibracao_local):
            sys.exit(f"--metodo v2 exige {args.calibracao_local} — gere com tools/calibrar_cores.py.")
        calibracao_local = _carregar_json(args.calibracao_local)

    print(f"[1-2/7] Carregando e pré-processando: {args.imagem}")
    print(f"[3/7]   Segmentação: {segmentacao} | [4/7] Detecção: {args.metodo}")

    resultado = executar_pipeline(
        args.imagem, config,
        segmentacao=segmentacao, metodo=args.metodo,
        largura=args.largura, usar_clahe=args.clahe, usar_wb=args.wb,
        kernel_blur=args.blur, caminho_homografia=args.homografia,
        eps=args.eps, calibracao_lab=calibracao_lab,
        calibracao_local=calibracao_local,
        debug=args.debug, saida=args.saida,
    )

    for aviso in resultado["avisos"]:
        print(f"[aviso] {aviso}")

    etapa5 = "retificado por homografia" if resultado["retificado"] else "sem retificação (pixels)"
    print(f"[5/7]   {etapa5}")
    print(f"[6/7]   {len(resultado['grupos'])} equipe(s) identificada(s)")

    deteccoes = resultado["deteccoes"]
    print(f"\n=== Detecções [{segmentacao}+{args.metodo}] ===")
    if not deteccoes:
        print("  Nenhum balão detectado.")
    else:
        contagem = Counter(d.cor for d in deteccoes)
        for cor, qtd in sorted(contagem.items()):
            print(f"  {cor}: {qtd} balão(ões)")
        print(f"  Total: {len(deteccoes)}")

    print(f"\n=== [7/7] Relatório por equipe ===")
    imprimir_relatorio(resultado["grupos"])
    print(f"\nSaída em: {args.saida} (overlay.jpg, relatorio.json)")


if __name__ == "__main__":
    main()
