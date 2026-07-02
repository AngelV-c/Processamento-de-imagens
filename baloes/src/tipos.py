"""Contrato de dados entre as etapas do pipeline.

Nota sobre o nome do arquivo: o plano original previa `types.py`, mas esse
nome colide com o módulo `types` da biblioteca padrão do Python — numpy e
sklearn o importam internamente e, com `src/` no início do sys.path, o
arquivo local seria carregado no lugar do módulo padrão, quebrando as
bibliotecas. Por isso o módulo se chama `tipos.py`.
"""

from dataclasses import dataclass, field


@dataclass
class Deteccao:
    """Representa um balão detectado na imagem.

    Atributos:
        cor: Nome da cor conforme definido em cores.json.
        cx: Coordenada x do centroide na imagem original (px).
        cy: Coordenada y do centroide na imagem original (px).
        area: Área do contorno em pixels².
        circularidade: Métrica 4π·A/P² no intervalo (0, 1].
        raio: Raio aproximado do balão em pixels.
        cx_ret: Coordenada x retificada pela homografia (unidade métrica).
        cy_ret: Coordenada y retificada pela homografia (unidade métrica).
        grupo_id: Identificador do grupo/equipe atribuído pelo DBSCAN.
    """

    cor: str
    cx: float
    cy: float
    area: float
    circularidade: float
    raio: float
    cx_ret: float | None = None
    cy_ret: float | None = None
    grupo_id: int | None = None


@dataclass
class Grupo:
    """Representa uma equipe (grupo de balões) identificada na cena.

    Atributos:
        grupo_id: Identificador vindo do DBSCAN.
        centro_ret: Centro do grupo nas coordenadas usadas no agrupamento
            (retificadas se houver homografia, senão pixels).
        cores: Conjunto de cores de balão presentes no grupo.
        problemas: Letras dos problemas correspondentes às cores.
        equipe: Nome/rótulo da equipe (por posição espacial ou mapa de assentos).
    """

    grupo_id: int
    centro_ret: tuple[float, float]
    cores: set[str] = field(default_factory=set)
    problemas: set[str] = field(default_factory=set)
    equipe: str | None = None
