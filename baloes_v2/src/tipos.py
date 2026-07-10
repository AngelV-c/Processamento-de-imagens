"""Contrato de dados entre as etapas do pipeline.

(Chama-se `tipos.py` e não `types.py` porque `types` colide com o módulo
homônimo da biblioteca padrão, que numpy e sklearn importam.)
"""

from dataclasses import dataclass, field


@dataclass
class Deteccao:
    """Um balão detectado na imagem.

    Atributos:
        cor: Nome da cor conforme a calibração local.
        cx, cy: Centroide na imagem original (px).
        area: Área do contorno (px²).
        circularidade: 4π·A/P² em (0, 1].
        raio: Raio aproximado (px).
        score: Pontuação combinada de forma+cor que aceitou o candidato.
        via: Qual detector gerou o candidato ("A" máscara, "B" Hough, "C" saturação).
        cx_ret, cy_ret: Centroide retificado pela homografia (unidade métrica).
        grupo_id: Grupo/equipe atribuído pelo DBSCAN.
    """

    cor: str
    cx: float
    cy: float
    area: float
    circularidade: float
    raio: float
    score: float = 0.0
    via: str = "?"
    cx_ret: float | None = None
    cy_ret: float | None = None
    grupo_id: int | None = None


@dataclass
class Grupo:
    """Uma equipe (grupo de balões) identificada na cena."""

    grupo_id: int
    centro_ret: tuple[float, float]
    cores: set[str] = field(default_factory=set)
    problemas: set[str] = field(default_factory=set)
    equipe: str | None = None
