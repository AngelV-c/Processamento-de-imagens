"""Interface web do rastreador de balões (Flask).

Três abas:
  🎯 Detecção — foto + todos os parâmetros do pipeline; resultado com
     placar estilo ICPC (equipes × problemas) e a via de detecção de
     cada balão (A máscara / B arco / C MSER).
  🏷️ Cores & Problemas — escolhe quais cores da calibração participam e
     atribui a letra do problema de cada cor, seguindo as regras da
     maratona (cores distintas, uma letra por cor, A–Z).
  🎨 Calibrar cena — clique nos balões direto no navegador para
     construir os modelos de cor do local.

Uso:
    python app.py            # http://localhost:5000
"""

import argparse
import base64
import glob
import json
import os
import sys
import uuid
from collections import Counter

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "src"))

from main import executar_pipeline
from preprocess import carregar, normalizar_iluminacao
from calibracao import pixels_do_disco, modelo_de_pixels

app = Flask(__name__)

_UPLOADS = os.path.join(_DIR, "output", "uploads")

_NOMES_VIAS = {"A": "máscara de cor + watershed",
               "B": "arco de borda (Kåsa)",
               "C": "MSER (canal S)"}

# (nome_no_form, caminho.pontuado.em.params, tipo)
_MAPA_FORM = [
    ("score_minimo",        "deteccao.score_minimo",         float),
    ("peso_circ",           "deteccao.pesos.circularidade",  float),
    ("peso_sol",            "deteccao.pesos.solidity",       float),
    ("peso_fourier",        "deteccao.pesos.fourier",        float),
    ("peso_cor",            "deteccao.pesos.cor",            float),
    ("peso_brilho",         "deteccao.pesos.brilho",         float),
    ("conf_cor",            "deteccao.conf_cor_minima",      float),
    ("conf_cor_hough",      "deteccao.conf_cor_minima_hough", float),
    ("altura_min",          "deteccao.fracao_altura_min",    float),
    ("altura_max",          "deteccao.fracao_altura_max",    float),
    ("anel_max",            "deteccao.anel_mesma_cor_max",   float),
    ("comp_razao",          "deteccao.componente_max_razao", float),
    ("area_min",            "deteccao.area_minima_relativa", float),
    ("area_max",            "deteccao.area_maxima_relativa", float),
    ("kernel_morf",         "deteccao.kernel_morfologia",    int),
    ("ms_sp",               "deteccao.meanshift_sp",         int),
    ("ms_sr",               "deteccao.meanshift_sr",         int),
    ("ws_kernel",           "watershed.kernel_maximos",      int),
    ("ws_limiar",           "watershed.limiar_distancia",    float),
    ("arcos_residuo",       "arcos.residuo_max",             float),
    ("arcos_cobertura",     "arcos.cobertura_min_graus",     float),
    ("arcos_comprimento",   "arcos.comprimento_min",         int),
    ("mser_delta",          "mser.delta",                    int),
    ("min_pts",             "agrupamento.min_pts",           int),
    ("mod_s_piso_fator",    "modelo.s_piso_fator",           float),
    ("mod_s_piso_min",      "modelo.s_piso_min",             float),
    ("mod_v_piso_fator",    "modelo.v_piso_fator",           float),
    ("mod_v_piso_min",      "modelo.v_piso_min",             float),
    ("mod_ab_tol_min",      "modelo.ab_tol_min",             float),
    ("mod_ab_tol_max",      "modelo.ab_tol_max",             float),
    ("mod_ab_tol_min_ac",   "modelo.ab_tol_min_acrom",       float),
    ("mod_ab_tol_max_ac",   "modelo.ab_tol_max_acrom",       float),
    ("mod_s_teto_fator",    "modelo.s_teto_fator",           float),
    ("mod_s_teto_min",      "modelo.s_teto_min",             float),
    ("mod_v_piso_ac_fator", "modelo.v_piso_acrom_fator",     float),
    ("mod_l_piso_fator",    "modelo.l_piso_fator",           float),
    ("mod_l_teto_fator",    "modelo.l_teto_fator",           float),
    ("mod_v_glare",         "modelo.v_glare_max",            float),
]


def _obter(params: dict, caminho: str):
    alvo = params
    for parte in caminho.split("."):
        alvo = alvo[parte]
    return alvo


def _definir(params: dict, caminho: str, valor) -> None:
    alvo = params
    partes = caminho.split(".")
    for parte in partes[:-1]:
        alvo = alvo.setdefault(parte, {})
    alvo[partes[-1]] = valor


def _params_base() -> dict:
    with open(os.path.join(_DIR, "config", "params.json"), encoding="utf-8") as f:
        return json.load(f)


def _valores_form(params: dict) -> dict:
    return {nome: _obter(params, caminho) for nome, caminho, _ in _MAPA_FORM}


def _aplicar_form(params: dict) -> None:
    for nome, caminho, tipo in _MAPA_FORM:
        bruto = request.form.get(nome, "").strip()
        if bruto:
            try:
                _definir(params, caminho, tipo(float(bruto)))
            except ValueError:
                pass
    # Checkbox desmarcado não é enviado — só interpretamos ausência como
    # "desligado" quando o marcador do formulário veio junto.
    if "vias_enviadas" in request.form:
        _definir(params, "deteccao.vias", {
            "mascaras": "via_a" in request.form,
            "arcos": "via_b" in request.form,
            "mser": "via_c" in request.form,
        })


def _listar_calibracoes() -> list[str]:
    opcoes = sorted(
        os.path.relpath(p, _DIR)
        for p in glob.glob(os.path.join(_DIR, "config", "calibracoes", "*.json"))
    )
    local = os.path.join(_DIR, "config", "calibracao_local.json")
    if os.path.exists(local):
        opcoes.insert(0, os.path.relpath(local, _DIR))
    return opcoes


def _jpeg_b64(imagem) -> str:
    _, buf = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("ascii")


def _swatch_hex(lab_media: list[float]) -> str:
    """Cor aproximada do modelo (LAB OpenCV → hex RGB) para a interface."""
    lab = np.array([[[lab_media[0], lab_media[1], lab_media[2]]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def _info_calibracao(caminho_rel: str) -> dict | None:
    caminho = os.path.join(_DIR, caminho_rel)
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    cores = []
    for nome, m in dados.get("cores", {}).items():
        cores.append({
            "nome": nome,
            "problema": m.get("problema", "?"),
            "ativa": m.get("ativa", True),
            "acromatica": m.get("acromatica", False),
            "swatch": _swatch_hex(m["lab_media"]),
            "n_cliques": m.get("n_cliques", 1),
        })
    return {"arquivo": caminho_rel, "cores": cores}


_PAGINA = r"""
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rastreador de Balões</title>
<style>
  :root {
    --bg: #0c1018; --painel: #141b29; --painel2: #1b2436; --borda: #293650;
    --texto: #e8edf7; --texto2: #8fa0c0; --realce: #7c5cff; --realce2: #23c4a4;
    --ouro: #f6c344; --perigo: #ff5c7a; --raio: 14px;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; color: var(--texto);
         background:
           radial-gradient(900px 480px at 85% -5%, #22194a 0%, transparent 60%),
           radial-gradient(700px 420px at -10% 10%, #0d2b33 0%, transparent 55%),
           var(--bg);
         min-height: 100vh; }
  header { display: flex; align-items: center; gap: 18px; padding: 16px 28px;
           border-bottom: 1px solid var(--borda);
           background: rgba(12,16,24,.72); backdrop-filter: blur(10px);
           position: sticky; top: 0; z-index: 10; }
  header .logo { font-size: 1.6rem; }
  header h1 { margin: 0; font-size: 1.1rem; font-weight: 600; letter-spacing: .3px; }
  header h1 small { display: block; font-size: .72rem; color: var(--texto2);
                    font-weight: 400; letter-spacing: 1px; }
  nav { margin-left: auto; display: flex; gap: 8px; }
  nav button { background: transparent; color: var(--texto2); border: 1px solid var(--borda);
               border-radius: 999px; padding: 8px 18px; font-size: .88rem; cursor: pointer;
               transition: all .15s; }
  nav button:hover { color: var(--texto); border-color: var(--texto2); }
  nav button.ativo { background: var(--realce); border-color: var(--realce); color: #fff; }
  main { display: grid; grid-template-columns: 390px 1fr; gap: 22px;
         max-width: 1560px; margin: 22px auto; padding: 0 22px; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
  .painel { background: var(--painel); border: 1px solid var(--borda);
            border-radius: var(--raio); padding: 18px; }
  .painel h2 { margin: 0 0 12px; font-size: .82rem; color: var(--texto2);
               text-transform: uppercase; letter-spacing: 1.4px; }
  label { display: block; font-size: .8rem; color: var(--texto2); margin: 12px 0 4px; }
  input[type=text], input[type=number], select {
    width: 100%; padding: 9px 11px; background: var(--painel2); color: var(--texto);
    border: 1px solid var(--borda); border-radius: 8px; font-size: .9rem; }
  input[type=file] { width: 100%; color: var(--texto2); font-size: .85rem; }
  input[type=file]::file-selector-button {
    background: var(--realce2); color: #06281f; border: 0; border-radius: 8px;
    padding: 8px 14px; margin-right: 10px; cursor: pointer; font-weight: 600; }
  .faixa { display: grid; grid-template-columns: 1fr 54px; gap: 10px; align-items: center; }
  input[type=range] { width: 100%; accent-color: var(--realce); }
  output { font-size: .8rem; color: var(--realce2); text-align: right;
           font-variant-numeric: tabular-nums; }
  details { border: 1px solid var(--borda); border-radius: 10px;
            padding: 10px 14px; margin-top: 12px; background: var(--painel2); }
  details summary { cursor: pointer; font-size: .85rem; font-weight: 600; user-select: none; }
  details[open] summary { margin-bottom: 6px; }
  .linha2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .chks { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; }
  .chks label { display: flex; gap: 6px; align-items: center; margin: 0;
                font-size: .85rem; color: var(--texto); }
  .btn { width: 100%; margin-top: 18px; background: linear-gradient(135deg, var(--realce), #5b3df5);
         color: #fff; border: 0; border-radius: 10px; padding: 13px;
         font-size: 1rem; font-weight: 700; cursor: pointer; letter-spacing: .3px; }
  .btn:hover { filter: brightness(1.12); }
  .btn.sec { background: var(--realce2); color: #06281f; }
  .aviso { background: rgba(246,195,68,.1); border: 1px solid #7a621f;
           border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; font-size: .85rem; }
  .erro { background: rgba(255,92,122,.12); border-color: var(--perigo); }
  .hero img { width: 100%; border-radius: var(--raio); border: 1px solid var(--borda); }
  .kpis { display: flex; gap: 14px; margin: 0 0 14px; flex-wrap: wrap; }
  .kpi { background: var(--painel2); border: 1px solid var(--borda); border-radius: 12px;
         padding: 12px 22px; }
  .kpi b { display: block; font-size: 1.6rem; color: var(--realce2); }
  .kpi span { font-size: .72rem; color: var(--texto2); text-transform: uppercase;
              letter-spacing: 1px; }
  h2.sec { font-size: 1rem; margin: 26px 0 6px; }
  h2.sec small { color: var(--texto2); font-weight: 400; font-size: .78rem; }
  table.placar { border-collapse: collapse; width: 100%; background: var(--painel);
                 border-radius: 12px; overflow: hidden; font-size: .88rem; }
  .placar th, .placar td { border: 1px solid var(--borda); padding: 9px 12px;
                           text-align: center; }
  .placar th { background: var(--painel2); color: var(--texto2);
               text-transform: uppercase; font-size: .72rem; letter-spacing: 1px; }
  .placar td.equipe { text-align: left; font-weight: 600; }
  .placar td.total { color: var(--ouro); font-weight: 700; }
  .bolinha { display: inline-block; width: 18px; height: 18px; border-radius: 50%;
             border: 2px solid rgba(255,255,255,.75); vertical-align: middle; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
           gap: 12px; margin-top: 10px; }
  .card { background: var(--painel); border: 1px solid var(--borda);
          border-radius: 12px; padding: 14px; }
  .card h3 { margin: 0 0 8px; font-size: .95rem; display: flex; align-items: center; gap: 8px; }
  .via { display: inline-block; padding: 1px 8px; border-radius: 6px; font-size: .7rem;
         font-weight: 700; letter-spacing: .5px; }
  .via.A { background: #1d4ed8; } .via.B { background: #b45309; } .via.C { background: #15803d; }
  .swatch { display: inline-block; width: 22px; height: 22px; border-radius: 6px;
            border: 1px solid rgba(255,255,255,.4); vertical-align: middle; }
  .prob { display: inline-flex; align-items: center; justify-content: center;
          min-width: 26px; height: 26px; border-radius: 8px; margin: 3px 3px 0 0;
          background: var(--realce); color: #fff; font-weight: 700; font-size: .85rem; }
  .mascaras { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
              gap: 12px; margin-top: 12px; }
  .mascaras figure { margin: 0; }
  .mascaras img { width: 100%; border-radius: 10px; border: 1px solid var(--borda); }
  .mascaras figcaption { font-size: .78rem; color: var(--texto2); padding: 4px 2px; }
  #alvo-wrap { position: relative; display: inline-block; max-width: 100%; }
  #alvo { max-width: 100%; border-radius: var(--raio); border: 1px solid var(--borda);
          cursor: crosshair; display: block; }
  .marca { position: absolute; width: 18px; height: 18px; border: 3px solid #fff;
           border-radius: 50%; transform: translate(-50%, -50%);
           box-shadow: 0 0 0 2px rgba(0,0,0,.55); pointer-events: none; }
  .amostras li { font-size: .85rem; margin: 4px 0; }
  .amostras button { background: none; border: 0; color: var(--perigo);
                     cursor: pointer; font-size: .9rem; }
  .vazio { color: var(--texto2); text-align: center; padding: 90px 20px; }
  .vazio div { font-size: 3rem; margin-bottom: 10px; }
  .cores-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 14px; margin-top: 14px; }
  .cor-card { background: var(--painel2); border: 1px solid var(--borda);
              border-radius: 12px; padding: 14px; display: grid; gap: 10px; }
  .cor-card .topo { display: flex; align-items: center; gap: 10px; }
  .cor-card .topo b { font-size: 1rem; text-transform: capitalize; }
  .cor-card .topo small { color: var(--texto2); margin-left: auto; }
  .cor-card .campos { display: grid; grid-template-columns: 90px 1fr; gap: 10px;
                      align-items: end; }
  .cor-card label { margin: 0 0 4px; }
  .regra { background: var(--painel2); border-left: 3px solid var(--realce2);
           border-radius: 8px; padding: 12px 16px; font-size: .86rem;
           color: var(--texto2); margin-top: 12px; }
  .regra b { color: var(--texto); }
  .dup { outline: 2px solid var(--perigo); }
</style>
</head>
<body>
<header>
  <span class="logo">🎈</span>
  <h1>Rastreador de Balões
    <small>MARATONA DE PROGRAMAÇÃO · PDI CLÁSSICO · SEM ML</small></h1>
  <nav>
    <button type="button" id="tab-detectar" class="ativo" onclick="aba('detectar')">🎯 Detecção</button>
    <button type="button" id="tab-cores" onclick="aba('cores')">🏷️ Cores &amp; Problemas</button>
    <button type="button" id="tab-calibrar" onclick="aba('calibrar')">🎨 Calibrar cena</button>
  </nav>
</header>

<!-- ============================ DETECÇÃO ============================ -->
<main id="aba-detectar">
  <form class="painel" method="post" action="/processar" enctype="multipart/form-data">
    <h2>Entrada</h2>
    <label>Foto do local</label>
    <input type="file" name="imagem" accept="image/*" required>
    <label>Calibração de cores da cena</label>
    <select name="calibracao">
      {% for c in calibracoes %}<option value="{{ c }}" {% if c == calib_sel %}selected{% endif %}>{{ c }}</option>{% endfor %}
    </select>
    <div class="linha2">
      <div><label>Largura (px) — igual à calibração</label>
        <input type="number" name="largura" value="{{ largura }}" min="400" max="4000"></div>
      <div><label>eps DBSCAN (vazio = config)</label>
        <input type="number" step="0.1" name="eps" value="{{ eps or '' }}" placeholder="cm ou px"></div>
    </div>

    <details open>
      <summary>Score combinado</summary>
      <label>Score mínimo <div class="faixa">
        <input type="range" name="score_minimo" min="0.4" max="0.95" step="0.01"
               value="{{ v.score_minimo }}" oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.score_minimo }}</output></div></label>
      <label>Peso circularidade <div class="faixa">
        <input type="range" name="peso_circ" min="0" max="1" step="0.05" value="{{ v.peso_circ }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.peso_circ }}</output></div></label>
      <label>Peso solidity <div class="faixa">
        <input type="range" name="peso_sol" min="0" max="1" step="0.05" value="{{ v.peso_sol }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.peso_sol }}</output></div></label>
      <label>Peso Fourier <div class="faixa">
        <input type="range" name="peso_fourier" min="0" max="1" step="0.05" value="{{ v.peso_fourier }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.peso_fourier }}</output></div></label>
      <label>Peso confiança de cor <div class="faixa">
        <input type="range" name="peso_cor" min="0" max="1" step="0.05" value="{{ v.peso_cor }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.peso_cor }}</output></div></label>
      <label>Peso brilho especular (látex reflete a luz) <div class="faixa">
        <input type="range" name="peso_brilho" min="0" max="0.3" step="0.01" value="{{ v.peso_brilho }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.peso_brilho }}</output></div></label>
    </details>

    <details>
      <summary>Vias de candidatos (como os balões são achados)</summary>
      <input type="hidden" name="vias_enviadas" value="1">
      <div class="chks">
        <label><input type="checkbox" name="via_a" {% if vias.mascaras %}checked{% endif %}>
          <span class="via A">A</span> máscara+watershed</label>
        <label><input type="checkbox" name="via_b" {% if vias.arcos %}checked{% endif %}>
          <span class="via B">B</span> arcos de borda</label>
        <label><input type="checkbox" name="via_c" {% if vias.mser %}checked{% endif %}>
          <span class="via C">C</span> MSER</label>
      </div>
      <label>Confiança de cor mínima (A/C) <div class="faixa">
        <input type="range" name="conf_cor" min="0" max="1" step="0.05" value="{{ v.conf_cor }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.conf_cor }}</output></div></label>
      <label>Confiança de cor mínima (B — círculo é perfeito por construção) <div class="faixa">
        <input type="range" name="conf_cor_hough" min="0" max="1" step="0.05" value="{{ v.conf_cor_hough }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.conf_cor_hough }}</output></div></label>
      <div class="linha2">
        <div><label>Arcos: resíduo máx (px)</label>
          <input type="number" step="0.1" name="arcos_residuo" value="{{ v.arcos_residuo }}"></div>
        <div><label>Arcos: cobertura mín (°)</label>
          <input type="number" name="arcos_cobertura" value="{{ v.arcos_cobertura }}"></div>
        <div><label>Arcos: comprimento mín</label>
          <input type="number" name="arcos_comprimento" value="{{ v.arcos_comprimento }}"></div>
        <div><label>MSER: delta</label>
          <input type="number" name="mser_delta" value="{{ v.mser_delta }}"></div>
      </div>
    </details>

    <details>
      <summary>Priors de domínio</summary>
      <label>Altura mínima dos balões (fração do topo — corta o teto) <div class="faixa">
        <input type="range" name="altura_min" min="0" max="0.5" step="0.01" value="{{ v.altura_min }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.altura_min }}</output></div></label>
      <label>Altura máxima (corta pessoas/mesas) <div class="faixa">
        <input type="range" name="altura_max" min="0.3" max="1" step="0.01" value="{{ v.altura_max }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.altura_max }}</output></div></label>
      <label>Anel mesma cor máx (isolamento) <div class="faixa">
        <input type="range" name="anel_max" min="0.1" max="1" step="0.05" value="{{ v.anel_max }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.anel_max }}</output></div></label>
      <label>Razão componente/candidato máx <div class="faixa">
        <input type="range" name="comp_razao" min="2" max="30" step="1" value="{{ v.comp_razao }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.comp_razao }}</output></div></label>
    </details>

    <details>
      <summary>Segmentação e watershed</summary>
      <div class="linha2">
        <div><label>Área mín (relativa)</label>
          <input type="number" step="0.00001" name="area_min" value="{{ v.area_min }}"></div>
        <div><label>Área máx (relativa)</label>
          <input type="number" step="0.01" name="area_max" value="{{ v.area_max }}"></div>
        <div><label>Kernel morfologia</label>
          <input type="number" name="kernel_morf" value="{{ v.kernel_morf }}" min="3" max="15" step="2"></div>
        <div><label>Kernel máximos (watershed)</label>
          <input type="number" name="ws_kernel" value="{{ v.ws_kernel }}" min="3" max="31" step="2"></div>
        <div><label>Mean-shift sp</label><input type="number" name="ms_sp" value="{{ v.ms_sp }}"></div>
        <div><label>Mean-shift sr</label><input type="number" name="ms_sr" value="{{ v.ms_sr }}"></div>
      </div>
      <label>Limiar de distância (watershed) <div class="faixa">
        <input type="range" name="ws_limiar" min="0.05" max="0.6" step="0.01" value="{{ v.ws_limiar }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.ws_limiar }}</output></div></label>
    </details>

    <details>
      <summary>Modelo de cor (avançado)</summary>
      <div class="linha2">
        <div><label>S piso fator</label><input type="number" step="0.05" name="mod_s_piso_fator" value="{{ v.mod_s_piso_fator }}"></div>
        <div><label>S piso mín</label><input type="number" name="mod_s_piso_min" value="{{ v.mod_s_piso_min }}"></div>
        <div><label>V piso fator</label><input type="number" step="0.05" name="mod_v_piso_fator" value="{{ v.mod_v_piso_fator }}"></div>
        <div><label>V piso mín</label><input type="number" name="mod_v_piso_min" value="{{ v.mod_v_piso_min }}"></div>
        <div><label>a-b tol mín</label><input type="number" name="mod_ab_tol_min" value="{{ v.mod_ab_tol_min }}"></div>
        <div><label>a-b tol máx</label><input type="number" name="mod_ab_tol_max" value="{{ v.mod_ab_tol_max }}"></div>
        <div><label>a-b tol mín (acrom.)</label><input type="number" name="mod_ab_tol_min_ac" value="{{ v.mod_ab_tol_min_ac }}"></div>
        <div><label>a-b tol máx (acrom.)</label><input type="number" name="mod_ab_tol_max_ac" value="{{ v.mod_ab_tol_max_ac }}"></div>
        <div><label>S teto fator</label><input type="number" step="0.05" name="mod_s_teto_fator" value="{{ v.mod_s_teto_fator }}"></div>
        <div><label>S teto mín</label><input type="number" name="mod_s_teto_min" value="{{ v.mod_s_teto_min }}"></div>
        <div><label>V piso fator (acrom.)</label><input type="number" step="0.05" name="mod_v_piso_ac_fator" value="{{ v.mod_v_piso_ac_fator }}"></div>
        <div><label>L piso fator</label><input type="number" step="0.01" name="mod_l_piso_fator" value="{{ v.mod_l_piso_fator }}"></div>
        <div><label>L teto fator</label><input type="number" step="0.01" name="mod_l_teto_fator" value="{{ v.mod_l_teto_fator }}"></div>
        <div><label>V glare máx</label><input type="number" name="mod_v_glare" value="{{ v.mod_v_glare }}"></div>
      </div>
    </details>

    <div class="chks" style="margin-top:14px">
      <label><input type="checkbox" name="mostrar_mascaras" {% if mostrar_mascaras %}checked{% endif %}> Mostrar máscaras por cor</label>
    </div>
    <button class="btn" type="submit">🎯 Processar imagem</button>
  </form>

  <section>
    {% if erro %}<div class="aviso erro">{{ erro }}</div>{% endif %}
    {% for aviso in avisos %}<div class="aviso">⚠ {{ aviso }}</div>{% endfor %}

    {% if overlay %}
      <div class="kpis">
        <div class="kpi"><b>{{ total }}</b><span>balões</span></div>
        <div class="kpi"><b>{{ grupos|length }}</b><span>equipes</span></div>
        <div class="kpi"><b>{{ problemas_letras|length }}</b><span>problemas com balão</span></div>
      </div>
      <div class="hero"><img src="data:image/jpeg;base64,{{ overlay }}" alt="Detecções"></div>

      <h2 class="sec">🏆 Placar por equipe
        <small>— regra da maratona: 1 balão por problema resolvido, cores distintas por problema</small></h2>
      <table class="placar">
        <tr><th>Equipe</th>{% for l in problemas_letras %}<th>{{ l }}</th>{% endfor %}<th>Total</th></tr>
        {% for g in grupos %}
        <tr>
          <td class="equipe">{{ g.equipe }}</td>
          {% for l in problemas_letras %}
          <td>{% if l in g.problemas %}<span class="bolinha"
              style="background:{{ swatches.get(l, '#888') }}"></span>{% endif %}</td>
          {% endfor %}
          <td class="total">{{ g.problemas|length }}</td>
        </tr>
        {% endfor %}
      </table>

      <h2 class="sec">🔍 Como cada balão foi encontrado</h2>
      <div class="cards">
        {% for d in deteccoes %}
        <div class="card">
          <h3><span class="swatch" style="background:{{ cores_swatch.get(d.cor, '#888') }}"></span>
              {{ d.cor }} <span class="via {{ d.via }}">{{ d.via }}</span></h3>
          <small>score {{ "%.2f"|format(d.score) }} · circ {{ "%.2f"|format(d.circularidade) }}
                 · ({{ "%.0f"|format(d.cx) }}, {{ "%.0f"|format(d.cy) }})</small>
        </div>
        {% endfor %}
      </div>
      <div class="regra">
        <b>Vias:</b> <span class="via A">A</span> máscara de cor calibrada + watershed ·
        <span class="via B">B</span> círculo ajustado a arco de borda (Canny + Kåsa — pega
        balões parcialmente ocluídos) · <span class="via C">C</span> MSER no canal de
        saturação (blobs estáveis, invariante a iluminação)
      </div>

      {% if mascaras %}
      <h2 class="sec">Máscaras por cor</h2>
      <div class="mascaras">
        {% for nome, b64 in mascaras %}
        <figure><img src="data:image/jpeg;base64,{{ b64 }}"><figcaption>{{ nome }}</figcaption></figure>
        {% endfor %}
      </div>
      {% endif %}
    {% else %}
      <div class="painel vazio"><div>🎈</div>
        Envie uma foto e ajuste os parâmetros à esquerda.<br>
        <small>Configure as cores e os problemas na aba 🏷️ antes de processar.</small>
      </div>
    {% endif %}
  </section>
</main>

<!-- ======================== CORES & PROBLEMAS ======================== -->
<main id="aba-cores" style="display:none; grid-template-columns: 1fr;">
  <div class="painel">
    <h2>Cores da cena ↔ problemas da maratona</h2>
    <div class="regra">
      Nas maratonas de programação (regras ICPC), os problemas são identificados por
      <b>letras (A, B, C…)</b> e cada problema tem uma <b>cor de balão única</b>.
      A equipe recebe o balão da cor ao resolver o problema — logo, dentro de uma
      equipe <b>cada cor aparece no máximo uma vez</b> (o detector usa isso como
      restrição do agrupamento). Aqui você escolhe <b>quais cores da calibração
      participam</b> da detecção e <b>qual letra corresponde a cada cor</b>.
      Duas cores com a mesma letra são marcadas em vermelho.
    </div>
    <label>Calibração</label>
    <select id="cores-arquivo" onchange="carregarCores()">
      {% for c in calibracoes %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
    </select>
    <div class="cores-grid" id="cores-grid"></div>
    <button class="btn" type="button" onclick="salvarCores()">💾 Salvar atribuições</button>
    <div id="cores-msg" style="margin-top:10px;font-size:.85rem;color:var(--realce2)"></div>
  </div>
</main>

<!-- ============================ CALIBRAR ============================ -->
<main id="aba-calibrar" style="display:none">
  <div class="painel">
    <h2>Calibração de cores</h2>
    <label>Foto do local (câmera na posição definitiva)</label>
    <input type="file" id="calib-arquivo" accept="image/*">
    <div class="linha2">
      <div><label>Largura (px)</label><input type="number" id="calib-largura" value="1200"></div>
      <div><label>Raio da amostra (px)</label><input type="number" id="calib-raio" value="8"></div>
    </div>
    <button class="btn sec" type="button" onclick="carregarImagemCalib()">📤 Carregar imagem</button>

    <label style="margin-top:18px">Cor do próximo clique</label>
    <input type="text" id="calib-cor" placeholder="ex.: vermelho, branco, rosa...">
    <label>Letra do problema desta cor</label>
    <input type="text" id="calib-problema" maxlength="1" placeholder="A">

    <h2 style="margin-top:18px">Amostras coletadas</h2>
    <ul class="amostras" id="calib-lista"></ul>

    <label>Nome do arquivo de calibração</label>
    <input type="text" id="calib-nome" placeholder="MeuGinasio">
    <button class="btn" type="button" onclick="salvarCalib()">💾 Salvar calibração</button>
    <div id="calib-msg" style="margin-top:10px;font-size:.85rem;color:var(--realce2)"></div>
  </div>

  <div class="painel">
    <h2>Clique no centro de um balão de cada cor</h2>
    <p style="font-size:.85rem;color:var(--texto2)">
      Dica: clique num balão <b>perto</b> e num <b>longe</b> da mesma cor — as amostras
      são fundidas e o modelo cobre a variação de iluminação. A imagem exibida já está
      normalizada (Shades-of-Gray + CLAHE): é exatamente o que o detector vê.</p>
    <div id="alvo-wrap">
      <img id="alvo" src="" alt="" style="display:none">
    </div>
  </div>
</main>

<script>
function aba(qual) {
  for (const nome of ['detectar', 'cores', 'calibrar']) {
    document.getElementById('aba-' + nome).style.display = (qual === nome)
      ? (nome === 'cores' ? 'grid' : 'grid') : 'none';
    document.getElementById('tab-' + nome).classList.toggle('ativo', qual === nome);
  }
  if (qual === 'cores') carregarCores();
}

/* ---------- Cores & Problemas ---------- */
async function carregarCores() {
  const arquivo = document.getElementById('cores-arquivo').value;
  const r = await fetch('/calibracao?arquivo=' + encodeURIComponent(arquivo));
  const dados = await r.json();
  const grid = document.getElementById('cores-grid');
  grid.innerHTML = '';
  if (dados.erro) { grid.innerHTML = '<p>' + dados.erro + '</p>'; return; }
  for (const c of dados.cores) {
    const div = document.createElement('div');
    div.className = 'cor-card';
    div.dataset.nome = c.nome;
    div.innerHTML = `
      <div class="topo">
        <span class="swatch" style="background:${c.swatch}"></span>
        <b>${c.nome}</b>
        <small>${c.acromatica ? 'acromática' : 'cromática'} · ${c.n_cliques} clique(s)</small>
      </div>
      <div class="campos">
        <div><label>Problema</label>
          <input type="text" maxlength="1" class="inp-prob" value="${c.problema === '?' ? '' : c.problema}"
                 oninput="this.value=this.value.toUpperCase(); validarDuplicatas()"></div>
        <div><label style="display:flex;gap:8px;align-items:center">
          <input type="checkbox" class="inp-ativa" ${c.ativa ? 'checked' : ''}>
          usar esta cor na detecção</label></div>
      </div>`;
    grid.appendChild(div);
  }
  validarDuplicatas();
}

function validarDuplicatas() {
  const cards = [...document.querySelectorAll('#cores-grid .cor-card')];
  const contagem = {};
  for (const card of cards) {
    const letra = card.querySelector('.inp-prob').value.trim();
    if (letra) contagem[letra] = (contagem[letra] || 0) + 1;
  }
  for (const card of cards) {
    const letra = card.querySelector('.inp-prob').value.trim();
    card.classList.toggle('dup', !!letra && contagem[letra] > 1);
  }
}

async function salvarCores() {
  const arquivo = document.getElementById('cores-arquivo').value;
  const cores = {};
  for (const card of document.querySelectorAll('#cores-grid .cor-card')) {
    cores[card.dataset.nome] = {
      problema: card.querySelector('.inp-prob').value.trim().toUpperCase() || '?',
      ativa: card.querySelector('.inp-ativa').checked,
    };
  }
  const r = await fetch('/calibracao/atualizar', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arquivo, cores }) });
  const dados = await r.json();
  document.getElementById('cores-msg').textContent = dados.erro || ('✔ ' + dados.mensagem);
}

/* ---------- Calibrar cena ---------- */
let calibId = null;
let amostras = [];

async function carregarImagemCalib() {
  const arq = document.getElementById('calib-arquivo').files[0];
  if (!arq) { alert('Escolha uma imagem primeiro.'); return; }
  const fd = new FormData();
  fd.append('imagem', arq);
  fd.append('largura', document.getElementById('calib-largura').value);
  const r = await fetch('/calibrar/imagem', { method: 'POST', body: fd });
  const dados = await r.json();
  if (dados.erro) { alert(dados.erro); return; }
  calibId = dados.id;
  amostras = [];
  renderLista();
  document.querySelectorAll('.marca').forEach(m => m.remove());
  const img = document.getElementById('alvo');
  img.src = 'data:image/jpeg;base64,' + dados.img;
  img.style.display = 'block';
}

document.getElementById('alvo').addEventListener('click', (ev) => {
  if (!calibId) return;
  const cor = document.getElementById('calib-cor').value.trim().toLowerCase();
  if (!cor) { alert('Digite o nome da cor antes de clicar.'); return; }
  const problema = (document.getElementById('calib-problema').value.trim().toUpperCase() || '?');
  const img = ev.target;
  const rect = img.getBoundingClientRect();
  const escala = img.naturalWidth / rect.width;
  const x = Math.round((ev.clientX - rect.left) * escala);
  const y = Math.round((ev.clientY - rect.top) * escala);
  amostras.push({ cor, x, y, problema });
  const marca = document.createElement('div');
  marca.className = 'marca';
  marca.style.left = (ev.clientX - rect.left) + 'px';
  marca.style.top = (ev.clientY - rect.top) + 'px';
  marca.style.borderColor = 'hsl(' + (amostras.length * 63 % 360) + ',90%,65%)';
  document.getElementById('alvo-wrap').appendChild(marca);
  renderLista();
});

function renderLista() {
  const ul = document.getElementById('calib-lista');
  ul.innerHTML = '';
  amostras.forEach((a, i) => {
    const li = document.createElement('li');
    li.textContent = `${a.cor} (${a.x}, ${a.y}) → problema ${a.problema} `;
    const btn = document.createElement('button');
    btn.textContent = '✕';
    btn.onclick = () => { amostras.splice(i, 1); renderLista(); };
    li.appendChild(btn);
    ul.appendChild(li);
  });
}

async function salvarCalib() {
  if (!calibId || amostras.length === 0) { alert('Carregue a imagem e colete amostras.'); return; }
  const nome = document.getElementById('calib-nome').value.trim() || 'calibracao_local';
  const r = await fetch('/calibrar/salvar', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: calibId, nome, amostras,
                           raio: parseInt(document.getElementById('calib-raio').value) })
  });
  const dados = await r.json();
  document.getElementById('calib-msg').textContent =
    dados.erro || ('✔ ' + dados.mensagem + ' Ajuste as letras na aba 🏷️.');
}
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def pagina_inicial():
    params = _params_base()
    return render_template_string(
        _PAGINA, overlay=None, erro=None, avisos=[],
        calibracoes=_listar_calibracoes(), calib_sel=None,
        v=_valores_form(params), vias=params["deteccao"]["vias"],
        largura=1200, eps=None, mostrar_mascaras=False,
    )


@app.route("/calibracao", methods=["GET"])
def calibracao_info():
    info = _info_calibracao(request.args.get("arquivo", ""))
    if info is None:
        return jsonify({"erro": "Calibração não encontrada."})
    return jsonify(info)


@app.route("/calibracao/atualizar", methods=["POST"])
def calibracao_atualizar():
    dados = request.get_json(force=True)
    caminho = os.path.join(_DIR, dados.get("arquivo", ""))
    if not os.path.exists(caminho):
        return jsonify({"erro": "Calibração não encontrada."})

    with open(caminho, encoding="utf-8") as f:
        calibracao = json.load(f)

    for nome, atualizacao in dados.get("cores", {}).items():
        if nome in calibracao.get("cores", {}):
            calibracao["cores"][nome]["problema"] = atualizacao.get("problema", "?")
            calibracao["cores"][nome]["ativa"] = bool(atualizacao.get("ativa", True))

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(calibracao, f, ensure_ascii=False, indent=2)
    return jsonify({"mensagem": "Atribuições salvas."})


@app.route("/processar", methods=["POST"])
def processar():
    params = _params_base()
    _aplicar_form(params)
    calibracoes = _listar_calibracoes()
    contexto = dict(calibracoes=calibracoes,
                    calib_sel=request.form.get("calibracao"),
                    v=_valores_form(params), vias=params["deteccao"]["vias"],
                    largura=request.form.get("largura", 1200),
                    eps=request.form.get("eps", ""),
                    mostrar_mascaras="mostrar_mascaras" in request.form)

    arquivo = request.files.get("imagem")
    if arquivo is None or arquivo.filename == "":
        return render_template_string(_PAGINA, overlay=None, avisos=[],
                                      erro="Nenhuma imagem enviada.", **contexto)

    caminho_calib = os.path.join(_DIR, request.form.get("calibracao", ""))
    if not os.path.exists(caminho_calib):
        return render_template_string(_PAGINA, overlay=None, avisos=[],
                                      erro="Calibração não encontrada — use a aba 🎨.",
                                      **contexto)

    os.makedirs(_UPLOADS, exist_ok=True)
    caminho = os.path.join(_UPLOADS, os.path.basename(arquivo.filename))
    arquivo.save(caminho)

    with open(caminho_calib, encoding="utf-8") as f:
        calibracao = json.load(f)

    eps_txt = request.form.get("eps", "").strip()
    try:
        resultado = executar_pipeline(
            caminho, params, calibracao,
            largura=int(request.form.get("largura", 1200)),
            caminho_homografia=os.path.join(_DIR, "config", "homografia.json"),
            eps=float(eps_txt) if eps_txt else None,
        )
    except Exception as exc:  # superfície de erro única da UI
        return render_template_string(_PAGINA, overlay=None, avisos=[],
                                      erro=f"Falha no processamento: {exc}", **contexto)

    # Placar ICPC: colunas = letras dos problemas das cores ativas
    cores_ativas = {n: m for n, m in calibracao["cores"].items() if m.get("ativa", True)}
    letra_por_cor = {n: m.get("problema", "?") for n, m in cores_ativas.items()}
    problemas_letras = sorted({l for l in letra_por_cor.values() if l != "?"})
    swatches = {letra_por_cor[n]: _swatch_hex(m["lab_media"])
                for n, m in cores_ativas.items() if letra_por_cor[n] != "?"}
    cores_swatch = {n: _swatch_hex(m["lab_media"]) for n, m in calibracao["cores"].items()}

    mascaras_b64 = []
    if "mostrar_mascaras" in request.form:
        mascaras_b64 = [(nome, _jpeg_b64(m)) for nome, m in resultado["mascaras"].items()]

    deteccoes = sorted(resultado["deteccoes"], key=lambda d: -d.score)
    return render_template_string(
        _PAGINA, overlay=_jpeg_b64(resultado["overlay"]),
        grupos=resultado["grupos"], deteccoes=deteccoes,
        problemas_letras=problemas_letras, swatches=swatches,
        cores_swatch=cores_swatch,
        total=len(deteccoes), mascaras=mascaras_b64,
        avisos=resultado["avisos"], erro=None, **contexto)


@app.route("/calibrar/imagem", methods=["POST"])
def calibrar_imagem():
    arquivo = request.files.get("imagem")
    if arquivo is None or arquivo.filename == "":
        return jsonify({"erro": "Nenhuma imagem enviada."})

    os.makedirs(_UPLOADS, exist_ok=True)
    ident = uuid.uuid4().hex[:12]
    bruto = os.path.join(_UPLOADS, f"calib_{ident}_orig")
    arquivo.save(bruto)

    largura = int(request.form.get("largura", 1200))
    try:
        bgr = carregar(bruto, largura_maxima=largura)
    except FileNotFoundError:
        return jsonify({"erro": "Arquivo inválido."})
    bgr_norm = normalizar_iluminacao(bgr)

    # Persistimos a imagem NORMALIZADA — os cliques serão amostrados nela
    cv2.imwrite(os.path.join(_UPLOADS, f"calib_{ident}.png"), bgr_norm)
    return jsonify({"id": ident, "img": _jpeg_b64(bgr_norm), "largura": largura})


@app.route("/calibrar/salvar", methods=["POST"])
def calibrar_salvar():
    dados = request.get_json(force=True)
    caminho_png = os.path.join(_UPLOADS, f"calib_{dados.get('id','')}.png")
    if not os.path.exists(caminho_png):
        return jsonify({"erro": "Sessão de calibração expirada — recarregue a imagem."})

    bgr = cv2.imread(caminho_png)
    raio = int(dados.get("raio", 8))
    amostras = dados.get("amostras", [])
    if not amostras:
        return jsonify({"erro": "Nenhuma amostra coletada."})

    por_cor: dict = {}
    for a in amostras:
        nome = a["cor"].strip().lower()
        d = por_cor.setdefault(nome, {"h": [], "s": [], "v": [], "lab": [],
                                      "problema": a.get("problema", "?"),
                                      "n_cliques": 0})
        h, s, v, lab = pixels_do_disco(bgr, int(a["x"]), int(a["y"]), raio)
        d["h"].append(h); d["s"].append(s); d["v"].append(v); d["lab"].append(lab)
        d["n_cliques"] += 1

    cores = {}
    for nome, d in por_cor.items():
        modelo = modelo_de_pixels(np.concatenate(d["h"]), np.concatenate(d["s"]),
                                  np.concatenate(d["v"]), np.concatenate(d["lab"]))
        modelo["problema"] = d["problema"]
        modelo["ativa"] = True
        modelo["n_cliques"] = d["n_cliques"]
        cores[nome] = modelo

    nome_arquivo = "".join(c for c in dados.get("nome", "calibracao_local")
                           if c.isalnum() or c in "-_") or "calibracao_local"
    destino = os.path.join(_DIR, "config", "calibracoes", f"{nome_arquivo}.json")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({"imagem_referencia": "upload via navegador",
                   "largura_calibracao": bgr.shape[1],
                   "raio_amostra": raio, "cores": cores},
                  f, ensure_ascii=False, indent=2)

    return jsonify({"mensagem": f"Calibração '{nome_arquivo}' salva com "
                                f"{len(cores)} cores."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface web do rastreador de balões.")
    parser.add_argument("--porta", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.porta, debug=False)
