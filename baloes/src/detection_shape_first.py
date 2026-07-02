"""Pipeline invertido: detecta formas circulares primeiro, classifica cor depois.

Estratégia:
  1. Threshold de saturação no HSV — balões são coloridos, fundo é neutro.
  2. Morfologia para limpar ruído e fechar buracos.
  3. Watershed para separar balões que se tocam.
  4. Filtro de forma: circularidade + solidity + aspect_ratio + Fourier.
  5. Para cada região aprovada, amostra o interior e classifica a cor
     comparando com as faixas HSV de cores.json (voto majoritário).

Vantagem sobre o pipeline HSV→forma:
  - Os limiares de forma são independentes de cena e câmera.
  - A classificação de cor usa apenas os pixels confirmadamente dentro
    de um balão — sem contaminar com fundo ou outros objetos.
  - Não há overfitting de faixa HSV por cor: uma nova cor aparece
    automaticamente como "desconhecida" em vez de falso positivo.
"""

import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from detection import Deteccao, _metricas_forma, _forma_de_balao
from detection_fourier import _fourier_energia_baixa
from detection_watershed import _watershed_mascara


def _mascara_candidatos(
    hsv: np.ndarray,
    sat_min: int = 60,
    val_min: int = 40,
    kernel_tam: int = 7,
) -> np.ndarray:
    """Máscara binária de pixels coloridos (alta saturação), independente de cor.

    Args:
        hsv: Imagem em espaço HSV (OpenCV: H 0-179, S 0-255, V 0-255).
        sat_min: Saturação mínima para considerar pixel colorido.
        val_min: Valor mínimo para excluir sombras muito escuras.
        kernel_tam: Kernel morfológico para fechar buracos e limpar ruído.
    """
    mascara = cv2.inRange(
        hsv,
        np.array([0, sat_min, val_min], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_tam, kernel_tam))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara


def _classificar_cor(
    regiao_mask: np.ndarray,
    hsv: np.ndarray,
    cores_config: List[dict],
    fracao_min: float = 0.15,
) -> Optional[str]:
    """Classifica a cor dominante dentro de uma região.

    Para cada cor em cores.json, conta quantos pixels da região
    caem dentro das faixas HSV. A cor com mais pixels vence.

    Args:
        regiao_mask: Máscara binária da região (255 = interior do balão).
        hsv: Imagem HSV completa.
        cores_config: Lista de cores de config["cores"].
        fracao_min: Fração mínima de pixels para aceitar a cor vencedora.
            Regiões onde nenhuma cor domina são rejeitadas.

    Returns:
        Nome da cor dominante ou None se nenhuma cor dominar.
    """
    n_total = int((regiao_mask > 0).sum())
    if n_total == 0:
        return None

    melhor_nome = None
    melhor_contagem = 0

    for cor in cores_config:
        contagem = 0
        for faixa in cor["faixas"]:
            m = cv2.inRange(
                hsv,
                np.array(faixa["lower"], dtype=np.uint8),
                np.array(faixa["upper"], dtype=np.uint8),
            )
            contagem += int(cv2.bitwise_and(m, regiao_mask).sum() // 255)

        if contagem > melhor_contagem:
            melhor_contagem = contagem
            melhor_nome = cor["nome"]

    if melhor_nome is None or melhor_contagem / n_total < fracao_min:
        return None

    return melhor_nome


def _mascara_unificada(
    hsv: np.ndarray,
    cores_config: List[dict],
    kernel_tam: int = 5,
) -> np.ndarray:
    """Une todas as máscaras HSV por cor numa única máscara de candidatos.

    Garante que só pixels que pertencem a alguma cor de balão entram no
    watershed — elimina fundo, mesas, roupas e outros objetos coloridos
    não catalogados.
    """
    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for cor in cores_config:
        for faixa in cor["faixas"]:
            mascara |= cv2.inRange(
                hsv,
                np.array(faixa["lower"], dtype=np.uint8),
                np.array(faixa["upper"], dtype=np.uint8),
            )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_tam, kernel_tam))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara


def detectar_baloes_shape_first(
    imagem_bgr: np.ndarray,
    config: dict,
    fracao_cor_min: float = 0.15,
    limiar_fourier: Optional[float] = None,
    usar_meanshift: bool = True,
    sp: int = 21,
    sr: int = 25,
) -> List[Deteccao]:
    """Detecta balões separando formas na máscara unificada de cores, classifica cor depois.

    Diferença do pipeline convencional (watershed por cor):
      - Aqui o watershed roda UMA VEZ na união de todas as máscaras HSV.
      - Isso separa balões de cores DIFERENTES que se tocam — o watershed
        convencional não consegue isso porque roda dentro de cada cor.
      - Depois que a forma é aprovada, a cor é classificada por voto
        majoritário dos pixels no interior.

    Args:
        imagem_bgr: Imagem original BGR.
        config: Dicionário de cores.json.
        fracao_cor_min: Fração mínima de pixels de cor para aceitar classificação.
        limiar_fourier: Se fornecido, aplica filtro espectral (None = usa cores.json).

    Returns:
        Lista de Deteccao ordenada por cor e área decrescente.
    """
    bgr_proc = cv2.pyrMeanShiftFiltering(imagem_bgr, sp=sp, sr=sr) if usar_meanshift else imagem_bgr
    hsv = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2HSV)

    altura, largura = imagem_bgr.shape[:2]
    area_total = altura * largura
    cfg = config["deteccao"]
    area_min = cfg["area_minima_relativa"] * area_total
    area_max = cfg["area_maxima_relativa"] * area_total

    ws_cfg = config.get("watershed", {})
    kernel_maximos   = ws_cfg.get("kernel_maximos", 9)
    limiar_distancia = ws_cfg.get("limiar_distancia", 0.2)

    if limiar_fourier is None:
        f_cfg = config.get("fourier", {})
        limiar_fourier = f_cfg.get("limiar_energia_baixa", None)

    kt = cfg["kernel_morfologia"]
    cores_config = config["cores"]
    mascara = _mascara_unificada(hsv, cores_config, kt)

    labels = _watershed_mascara(mascara, kernel_maximos, limiar_distancia)
    n_labels = labels.max()

    deteccoes: List[Deteccao] = []

    for label in range(1, n_labels + 1):
        regiao = np.zeros(mascara.shape, dtype=np.uint8)
        regiao[(labels == label) & (mascara > 0)] = 255

        contornos, _ = cv2.findContours(
            regiao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contornos:
            continue

        contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(contorno)
        if area < area_min or area > area_max:
            continue

        circularidade, solidity, aspect_ratio = _metricas_forma(contorno)
        if not _forma_de_balao(circularidade, solidity, aspect_ratio, cfg):
            continue

        if limiar_fourier is not None:
            if _fourier_energia_baixa(contorno) < limiar_fourier:
                continue

        nome_cor = _classificar_cor(regiao, hsv, cores_config, fracao_cor_min)
        if nome_cor is None:
            continue

        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue
        cx = momentos["m10"] / momentos["m00"]
        cy = momentos["m01"] / momentos["m00"]
        raio = math.sqrt(area / math.pi)

        deteccoes.append(
            Deteccao(
                cor=nome_cor,
                cx=cx, cy=cy,
                area=area,
                circularidade=circularidade,
                raio=raio,
            )
        )

    deteccoes.sort(key=lambda d: (d.cor, -d.area))
    return deteccoes
