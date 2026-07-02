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
