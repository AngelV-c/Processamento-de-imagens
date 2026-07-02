"""Etapa 7 — Relatório de problemas resolvidos por equipe.

Como cada cor de balão corresponde a um problema e dentro de uma equipe
cada cor aparece no máximo uma vez, o relatório é o CONJUNTO de cores por
grupo — presença/ausência, sem contagem de duplicatas.
"""

import json
import os

from tipos import Deteccao, Grupo


def montar_relatorio(deteccoes: list[Deteccao], config: dict) -> list[Grupo]:
    """Agrega as detecções por grupo_id e mapeia cores → letras de problema.

    As equipes são nomeadas por posição espacial nas coordenadas usadas no
    agrupamento (ordenação por linha e depois por coluna), de modo que
    "Equipe 1" é a mais acima/à esquerda da vista retificada.

    Args:
        deteccoes: Detecções com grupo_id preenchido pela etapa 6.
        config: Configuração contendo config["cores"] com o campo "problema".

    Returns:
        Lista de Grupo ordenada pelo nome da equipe.
    """
    cor_para_problema = {
        c["nome"]: c.get("problema", "?") for c in config["cores"]
    }

    grupos: dict[int, Grupo] = {}
    for det in deteccoes:
        if det.grupo_id is None:
            continue
        if det.grupo_id not in grupos:
            grupos[det.grupo_id] = Grupo(grupo_id=det.grupo_id, centro_ret=(0.0, 0.0))
        grupos[det.grupo_id].cores.add(det.cor)
        grupos[det.grupo_id].problemas.add(cor_para_problema.get(det.cor, "?"))

    # Centro de cada grupo = média dos centroides (retificados se houver)
    for gid, grupo in grupos.items():
        membros = [d for d in deteccoes if d.grupo_id == gid]
        if all(d.cx_ret is not None for d in membros):
            xs = [d.cx_ret for d in membros]
            ys = [d.cy_ret for d in membros]
        else:
            xs = [d.cx for d in membros]
            ys = [d.cy for d in membros]
        grupo.centro_ret = (sum(xs) / len(xs), sum(ys) / len(ys))

    # Nomear por posição espacial: linha (y) primeiro, coluna (x) depois.
    # A tolerância de linha evita que pequenas diferenças de y embaralhem
    # equipes lado a lado.
    ordenados = sorted(grupos.values(), key=lambda g: g.centro_ret[1])
    if ordenados:
        alturas = [g.centro_ret[1] for g in ordenados]
        faixa = (max(alturas) - min(alturas)) or 1.0
        tol_linha = faixa * 0.15

        linhas: list[list[Grupo]] = [[ordenados[0]]]
        for g in ordenados[1:]:
            if g.centro_ret[1] - linhas[-1][-1].centro_ret[1] <= tol_linha:
                linhas[-1].append(g)
            else:
                linhas.append([g])

        indice = 1
        for linha in linhas:
            for g in sorted(linha, key=lambda g: g.centro_ret[0]):
                g.equipe = f"Equipe {indice}"
                indice += 1

    return sorted(grupos.values(), key=lambda g: g.equipe or "")


def salvar_relatorio_json(grupos: list[Grupo], caminho: str) -> None:
    """Serializa o relatório em JSON estruturado (sets viram listas ordenadas)."""
    dados = [
        {
            "equipe": g.equipe,
            "grupo_id": g.grupo_id,
            "centro": [round(g.centro_ret[0], 1), round(g.centro_ret[1], 1)],
            "cores": sorted(g.cores),
            "problemas": sorted(g.problemas),
        }
        for g in grupos
    ]
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def imprimir_relatorio(grupos: list[Grupo]) -> None:
    """Resumo legível no terminal: uma linha por equipe."""
    if not grupos:
        print("  Nenhuma equipe identificada.")
        return
    for g in grupos:
        cx, cy = g.centro_ret
        problemas = ", ".join(sorted(g.problemas))
        print(f"  {g.equipe} (posição {cx:.0f},{cy:.0f}): problemas {problemas}")
