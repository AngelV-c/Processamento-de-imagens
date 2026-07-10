"""Etapa 6 — Agrupamento de balões por equipe com DBSCAN.

Por que DBSCAN: um grupo de balões de uma equipe é uma região densa de
centroides cercada de espaço vazio. DBSCAN agrupa por densidade, não exige
saber o número de equipes de antemão e isola pontos espúrios naturalmente.

Premissa de separação (garantia do domínio): a maior distância entre balões
de uma MESMA equipe é bem menor que a menor distância entre balões de equipes
DIFERENTES. Logo existe uma faixa de eps que separa as equipes sem erro — e a
homografia (etapa 5) torna essa faixa única para a imagem inteira, porque as
distâncias retificadas são proporcionais às reais.

min_pts = 1 por padrão: uma equipe pode ter um único balão (resolveu só um
problema); com min_pts >= 2 esse balão solitário viraria "ruído" (-1) e seria
descartado. Com min_pts = 1 todo balão pertence a um grupo e o DBSCAN se
reduz a componentes conectados por distância eps (single-linkage no raio eps),
que é exatamente o comportamento desejado dada a premissa de separação.

RESTRIÇÃO DE UNICIDADE (regra da maratona): dentro de uma equipe cada cor
aparece NO MÁXIMO uma vez — um balão por problema resolvido. Isso vira uma
restrição estrutural do agrupamento: um cluster com duas detecções da mesma
cor está errado por construção. agrupar_com_restricao() usa a regra em dois
níveis:
  1. Cluster violador é re-agrupado recursivamente com eps menor — o caso
     típico é o eps grande demais ter fundido duas equipes vizinhas.
  2. Se nem o eps mínimo separa (as duas detecções da mesma cor estão
     praticamente juntas), a de MENOR score é descartada — pela regra do
     domínio, uma delas é necessariamente um falso positivo.
"""

import numpy as np
from sklearn.cluster import DBSCAN

from tipos import Deteccao


def agrupar(
    deteccoes: list[Deteccao],
    eps: float,
    min_pts: int = 1,
) -> list[Deteccao]:
    """Roda DBSCAN sobre os centroides e preenche grupo_id em cada Deteccao.

    Usa (cx_ret, cy_ret) quando disponíveis (pipeline com homografia);
    caso contrário cai para (cx, cy) em pixels — nesse caso o chamador
    deve ter avisado o usuário que o eps precisa estar calibrado em px.

    Args:
        deteccoes: Detecções da etapa 4 (com ou sem retificação).
        eps: Raio de vizinhança do DBSCAN — em CM se retificado, senão em px.
        min_pts: Mínimo de vizinhos para região densa (padrão 1, ver módulo).

    Returns:
        As mesmas detecções, com grupo_id preenchido (in-place).
    """
    if not deteccoes:
        return deteccoes

    retificado = all(d.cx_ret is not None and d.cy_ret is not None for d in deteccoes)
    if retificado:
        pontos = np.array([[d.cx_ret, d.cy_ret] for d in deteccoes])
    else:
        pontos = np.array([[d.cx, d.cy] for d in deteccoes])

    rotulos = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(pontos)

    for det, rotulo in zip(deteccoes, rotulos):
        det.grupo_id = int(rotulo)

    return deteccoes


def _pontos(deteccoes: list[Deteccao]) -> np.ndarray:
    retificado = all(d.cx_ret is not None and d.cy_ret is not None for d in deteccoes)
    if retificado:
        return np.array([[d.cx_ret, d.cy_ret] for d in deteccoes])
    return np.array([[d.cx, d.cy] for d in deteccoes])


def agrupar_com_restricao(
    deteccoes: list[Deteccao],
    eps: float,
    min_pts: int = 1,
    fator_reducao: float = 0.75,
    eps_min_relativo: float = 0.35,
) -> tuple[list[Deteccao], list[Deteccao]]:
    """DBSCAN com a restrição de unicidade de cor por equipe.

    Args:
        deteccoes: Detecções com cor e score preenchidos.
        eps: Raio inicial do DBSCAN.
        min_pts: Mínimo de vizinhos (padrão 1).
        fator_reducao: Fator aplicado ao eps a cada tentativa de separar
            um cluster violador (0.75 → 3 tentativas até o piso).
        eps_min_relativo: Piso do eps como fração do inicial — abaixo
            disso, desiste de separar e descarta o duplicado de menor score.

    Returns:
        (deteccoes_validas, descartadas) — as válidas com grupo_id final;
        as descartadas são duplicatas de cor eliminadas pela regra.
    """
    if not deteccoes:
        return deteccoes, []

    agrupar(deteccoes, eps=eps, min_pts=min_pts)
    eps_piso = eps * eps_min_relativo
    descartadas: list[Deteccao] = []

    proximo_id = max((d.grupo_id for d in deteccoes if d.grupo_id is not None),
                     default=-1) + 1

    def _cores_duplicadas(grupo: list[Deteccao]) -> bool:
        cores = [d.cor for d in grupo]
        return len(cores) != len(set(cores))

    # Fila de clusters a validar
    pendentes = {}
    for d in deteccoes:
        pendentes.setdefault(d.grupo_id, []).append(d)
    fila = [(membros, eps) for membros in pendentes.values()]

    validas: list[Deteccao] = []
    while fila:
        membros, eps_atual = fila.pop()
        if not _cores_duplicadas(membros) or len(membros) < 2:
            validas.extend(membros)
            continue

        novo_eps = eps_atual * fator_reducao
        if novo_eps >= eps_piso:
            # Tenta separar: o caso típico é duas equipes fundidas
            pontos = _pontos(membros)
            rotulos = DBSCAN(eps=novo_eps, min_samples=min_pts).fit_predict(pontos)
            if len(set(rotulos)) > 1:
                subgrupos: dict[int, list[Deteccao]] = {}
                for det, rot in zip(membros, rotulos):
                    subgrupos.setdefault(int(rot), []).append(det)
                for sub in subgrupos.values():
                    novo_id = proximo_id
                    proximo_id += 1
                    for det in sub:
                        det.grupo_id = novo_id
                    fila.append((sub, novo_eps))
                continue
            # Não separou — tenta de novo com eps ainda menor
            fila.append((membros, novo_eps))
            continue

        # eps no piso e ainda há duplicata: uma delas é falso positivo —
        # mantém a de maior score de cada cor
        melhores: dict[str, Deteccao] = {}
        for det in membros:
            atual = melhores.get(det.cor)
            if atual is None or det.score > atual.score:
                if atual is not None:
                    atual.grupo_id = None
                    descartadas.append(atual)
                melhores[det.cor] = det
            else:
                det.grupo_id = None
                descartadas.append(det)
        validas.extend(melhores.values())

    return validas, descartadas


def distancias_vizinho_mais_proximo(deteccoes: list[Deteccao]) -> np.ndarray:
    """Distância de cada centroide ao vizinho mais próximo, ordenada.

    Base do diagnóstico de eps: a premissa de separação aparece como um
    "salto" claro nesta curva — o eps ideal fica dentro do salto.
    """
    retificado = all(d.cx_ret is not None and d.cy_ret is not None for d in deteccoes)
    if retificado:
        pontos = np.array([[d.cx_ret, d.cy_ret] for d in deteccoes])
    else:
        pontos = np.array([[d.cx, d.cy] for d in deteccoes])

    n = len(pontos)
    if n < 2:
        return np.array([])

    dists = np.sqrt(((pontos[:, None, :] - pontos[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.sort(dists.min(axis=1))
