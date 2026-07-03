"""Modelos de cor calibrados por local (recomendação nº 1).

Um modelo por cor, construído a partir de cliques em balões reais da cena:

  - cromática: mediana circular do matiz + tolerância, pisos de S/V relativos
    aos percentis da amostra, e portão de cromaticidade no plano a-b do LAB
    (o matiz sozinho não separa balão amarelo de parede creme).
  - acromática (branco/cinza/prata, S_p50 < 35): matiz é ruído — o modelo usa
    croma a-b apertado + faixa de brilho L (piso separa de teto/camiseta,
    teto separa de luminária/glare) + corte de pixels estourados.

Este módulo é usado tanto pela ferramenta de calibração (construção) quanto
pelo detector (casamento pixel-a-pixel e classificação de regiões).
"""

import json
import math

import cv2
import numpy as np

# S mediana abaixo disso → cor sem matiz confiável. 35 e não mais: um rosa
# pálido com S≈45 ainda tem matiz utilizável — tratá-lo como acromático faz
# o modelo casar com qualquer superfície branca da cena.
SATURACAO_ACROMATICA = 35


# ---------------------------------------------------------------------------
# Construção (usada por tools/calibrar_cores.py)
# ---------------------------------------------------------------------------

def mediana_circular_h(h_vals: np.ndarray) -> float:
    """Mediana do matiz respeitando a circularidade (H de 0 a 179).

    Correta mesmo para o vermelho, que ocupa as duas pontas da escala.
    """
    angulos = h_vals.astype(np.float64) * (2 * math.pi / 180.0)
    seno, cosseno = np.sin(angulos).mean(), np.cos(angulos).mean()
    angulo = math.atan2(seno, cosseno)
    if angulo < 0:
        angulo += 2 * math.pi
    return angulo * 180.0 / (2 * math.pi)


def pixels_do_disco(bgr: np.ndarray, x: int, y: int, raio: int):
    """(h, s, v, lab) dos pixels num disco centrado em (x, y)."""
    mascara = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mascara, (x, y), raio, 255, -1)
    sel = mascara > 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    return (hsv[:, :, 0][sel].astype(np.float64),
            hsv[:, :, 1][sel].astype(np.float64),
            hsv[:, :, 2][sel].astype(np.float64),
            lab[sel])


def modelo_de_pixels(h: np.ndarray, s: np.ndarray, v: np.ndarray,
                     pix_lab: np.ndarray) -> dict:
    """Constrói o modelo a partir dos pixels de TODAS as amostras da cor.

    Clicar num balão perto E num longe da mesma cor faz o modelo cobrir a
    variação real de iluminação/distância.
    """
    s_p50 = float(np.percentile(s, 50))
    acromatica = s_p50 < SATURACAO_ACROMATICA

    h_med = mediana_circular_h(h)
    desvio = np.minimum(np.abs(h - h_med), 180.0 - np.abs(h - h_med))
    h_tol = float(max(8.0, min(18.0, 3.0 * np.percentile(desvio, 90))))

    cov = np.cov(pix_lab.T) + np.eye(3) * 2.0

    return {
        "acromatica": bool(acromatica),
        "h_mediana": round(h_med, 1),
        "h_tolerancia": round(h_tol, 1),
        "s_p10": float(np.percentile(s, 10)),
        "s_p50": s_p50,
        "s_p90": float(np.percentile(s, 90)),
        "v_p10": float(np.percentile(v, 10)),
        "v_p50": float(np.percentile(v, 50)),
        "l_p10": float(np.percentile(pix_lab[:, 0], 10)),
        "l_p90": float(np.percentile(pix_lab[:, 0], 90)),
        "lab_media": [round(float(m), 2) for m in pix_lab.mean(axis=0)],
        "lab_cov": [[round(float(c), 3) for c in linha] for linha in cov],
        "n_pixels": int(len(h)),
    }


def carregar_calibracao(caminho: str) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Casamento (usado pelo detector)
# ---------------------------------------------------------------------------

def _dist_h_circular(h: np.ndarray, h_ref: float) -> np.ndarray:
    d = np.abs(h.astype(np.float64) - h_ref)
    return np.minimum(d, 180.0 - d)


# Ajustes de casamento — todos os fatores/pisos são calibráveis via
# params.json (seção "modelo"); estes são apenas os padrões.
AJUSTES_PADRAO = {
    "s_piso_fator": 0.5,        # piso de S = fator × s_p10 da amostra
    "s_piso_min": 30.0,         # nunca abaixo disso (ruído de fundo)
    "v_piso_fator": 0.5,
    "v_piso_min": 30.0,
    "ab_tol_min": 15.0,         # cromática: janela a-b mínima/máxima
    "ab_tol_max": 30.0,
    "ab_tol_min_acrom": 8.0,    # acromática: janela a-b apertada
    "ab_tol_max_acrom": 15.0,
    "s_teto_fator": 1.2,        # acromática: teto de S = fator × s_p90
    "s_teto_min": 50.0,
    "v_piso_acrom_fator": 0.75,
    "l_piso_fator": 0.9,        # separa balão branco de teto/camiseta
    "l_teto_fator": 1.08,       # separa balão branco de luminária
    "v_glare_max": 250.0,       # corte absoluto de pixels estourados
}


def casar_modelo(hsv: np.ndarray, lab: np.ndarray, modelo: dict,
                 ajustes: dict | None = None) -> np.ndarray:
    """Máscara booleana dos pixels compatíveis com um modelo de cor."""
    aj = {**AJUSTES_PADRAO, **(ajustes or {})}
    h = hsv[:, :, 0].astype(np.float64)
    s = hsv[:, :, 1].astype(np.float64)
    v = hsv[:, :, 2].astype(np.float64)
    media = modelo["lab_media"]
    cov = modelo["lab_cov"]
    dist_ab = np.sqrt((lab[:, :, 1] - media[1]) ** 2 + (lab[:, :, 2] - media[2]) ** 2)
    disp_ab = 2.5 * math.sqrt(cov[1][1] + cov[2][2])

    if modelo["acromatica"]:
        ab_tol = max(aj["ab_tol_min_acrom"], min(aj["ab_tol_max_acrom"], disp_ab))
        s_teto = max(aj["s_teto_min"], modelo["s_p90"] * aj["s_teto_fator"])
        v_piso = modelo["v_p10"] * aj["v_piso_acrom_fator"]
        l_piso = modelo["l_p10"] * aj["l_piso_fator"]
        l_teto = modelo["l_p90"] * aj["l_teto_fator"]
        L = lab[:, :, 0]
        return ((dist_ab <= ab_tol) & (s <= s_teto) & (v >= v_piso)
                & (v <= aj["v_glare_max"]) & (L >= l_piso) & (L <= l_teto))

    dh = _dist_h_circular(h, modelo["h_mediana"])
    s_piso = max(aj["s_piso_min"], modelo["s_p10"] * aj["s_piso_fator"])
    v_piso = max(aj["v_piso_min"], modelo["v_p10"] * aj["v_piso_fator"])
    ab_tol = max(aj["ab_tol_min"], min(aj["ab_tol_max"], disp_ab))
    return (dh <= modelo["h_tolerancia"]) & (s >= s_piso) & (v >= v_piso) & (dist_ab <= ab_tol)


def mascaras_calibradas(hsv: np.ndarray, lab: np.ndarray, modelos: dict,
                        kernel_tam: int = 5, ajustes: dict | None = None) -> dict:
    """Uma máscara binária limpa (abertura+fechamento) por cor calibrada."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_tam, kernel_tam))
    mascaras = {}
    for nome, modelo in modelos.items():
        m = (casar_modelo(hsv, lab, modelo, ajustes) * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        mascaras[nome] = m
    return mascaras


def classificar_regiao(regiao_sel: np.ndarray, hsv: np.ndarray, lab: np.ndarray,
                       modelos: dict,
                       ajustes: dict | None = None) -> tuple[str | None, float]:
    """(cor, confiança): cor com maior fração de pixels compatíveis na região.

    Empates próximos (diferença < 0.10) são desfeitos pela distância no plano
    a-b — resolve a fronteira magenta entre rosa e vermelho, onde as janelas
    de matiz se sobrepõem.
    """
    n_total = int(regiao_sel.sum())
    if n_total == 0:
        return None, 0.0

    fracoes: list[tuple[float, str]] = []
    for nome, modelo in modelos.items():
        match = casar_modelo(hsv, lab, modelo, ajustes)
        frac = float((match & regiao_sel).sum()) / n_total
        fracoes.append((frac, nome))
    fracoes.sort(reverse=True)

    (f1, nome1), *resto = fracoes
    if f1 == 0.0:
        return None, 0.0

    if resto and f1 - resto[0][0] < 0.10:
        # Desempate por proximidade a-b da cor média da região
        ab_regiao = lab[regiao_sel][:, 1:].mean(axis=0)
        candidatos = [nome1, resto[0][1]]
        dists = []
        for nome in candidatos:
            media = modelos[nome]["lab_media"]
            dists.append(math.hypot(ab_regiao[0] - media[1], ab_regiao[1] - media[2]))
        nome1 = candidatos[int(np.argmin(dists))]

    return nome1, f1
