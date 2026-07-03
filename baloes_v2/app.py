"""Interface web do rastreador de balões (Flask).

Duas abas:
  🎯 Detectar — upload da foto + TODOS os parâmetros do pipeline ajustáveis
     (score, pesos, vias, priors, watershed, Hough, modelo de cor).
  🎨 Calibrar — clique nos balões direto no navegador para construir a
     calibração de cores da cena (sem janela do OpenCV).

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

# (nome_no_form, caminho.pontuado.em.params, tipo)
_MAPA_FORM = [
    ("score_minimo",        "deteccao.score_minimo",         float),
    ("peso_circ",           "deteccao.pesos.circularidade",  float),
    ("peso_sol",            "deteccao.pesos.solidity",       float),
    ("peso_fourier",        "deteccao.pesos.fourier",        float),
    ("peso_cor",            "deteccao.pesos.cor",            float),
    ("conf_cor",            "deteccao.conf_cor_minima",      float),
    ("conf_cor_hough",      "deteccao.conf_cor_minima_hough", float),
    ("sat_percentil",       "deteccao.sat_percentil",        float),
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
    ("hough_dp",            "hough.dp",                      float),
    ("hough_p1",            "hough.param1",                  int),
    ("hough_p2",            "hough.param2",                  int),
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
    # Checkbox desmarcado não é enviado pelo navegador — só interpretamos
    # a ausência como "desligado" quando o marcador do formulário veio junto.
    if "vias_enviadas" in request.form:
        _definir(params, "deteccao.vias", {
            "mascaras": "via_a" in request.form,
            "hough": "via_b" in request.form,
            "saturacao": "via_c" in request.form,
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


_PAGINA = r"""
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rastreador de Balões</title>
<style>
  :root {
    --bg: #0f1420; --painel: #171e2e; --painel2: #1d2639; --borda: #2a3550;
    --texto: #e6ebf5; --texto2: #93a0bd; --realce: #7c5cff; --realce2: #23c4a4;
    --perigo: #ff5c7a; --raio: 14px;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
         background: radial-gradient(1200px 600px at 80% -10%, #1c2440 0%, var(--bg) 55%);
         color: var(--texto); min-height: 100vh; }
  header { display: flex; align-items: center; gap: 18px; padding: 18px 28px;
           border-bottom: 1px solid var(--borda);
           background: rgba(15,20,32,.7); backdrop-filter: blur(8px);
           position: sticky; top: 0; z-index: 10; }
  header h1 { margin: 0; font-size: 1.15rem; font-weight: 600; letter-spacing: .3px; }
  header h1 span { color: var(--realce2); }
  nav { margin-left: auto; display: flex; gap: 8px; }
  nav button { background: transparent; color: var(--texto2); border: 1px solid var(--borda);
               border-radius: 999px; padding: 8px 18px; font-size: .9rem; cursor: pointer; }
  nav button.ativo { background: var(--realce); border-color: var(--realce); color: #fff; }
  main { display: grid; grid-template-columns: 380px 1fr; gap: 22px;
         max-width: 1500px; margin: 22px auto; padding: 0 22px; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .painel { background: var(--painel); border: 1px solid var(--borda);
            border-radius: var(--raio); padding: 18px; }
  .painel h2 { margin: 0 0 12px; font-size: .95rem; color: var(--texto2);
               text-transform: uppercase; letter-spacing: 1.2px; }
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
  details summary { cursor: pointer; font-size: .85rem; color: var(--texto);
                    font-weight: 600; user-select: none; }
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
  .aviso { background: rgba(255,196,0,.12); border: 1px solid #8a6d1a;
           border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; font-size: .85rem; }
  .erro { background: rgba(255,92,122,.12); border-color: var(--perigo); }
  .hero img { width: 100%; border-radius: var(--raio); border: 1px solid var(--borda); }
  .kpis { display: flex; gap: 14px; margin: 0 0 14px; flex-wrap: wrap; }
  .kpi { background: var(--painel2); border: 1px solid var(--borda); border-radius: 12px;
         padding: 12px 20px; }
  .kpi b { display: block; font-size: 1.6rem; color: var(--realce2); }
  .kpi span { font-size: .75rem; color: var(--texto2); text-transform: uppercase;
              letter-spacing: 1px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
           gap: 12px; margin-top: 12px; }
  .card { background: var(--painel2); border: 1px solid var(--borda);
          border-radius: 12px; padding: 14px; }
  .card h3 { margin: 0 0 8px; font-size: .95rem; }
  .card small { color: var(--texto2); }
  .prob { display: inline-flex; align-items: center; justify-content: center;
          min-width: 26px; height: 26px; border-radius: 8px; margin: 3px 3px 0 0;
          background: var(--realce); color: #fff; font-weight: 700; font-size: .85rem; }
  .chip { display: inline-block; background: var(--borda); border-radius: 999px;
          padding: 2px 10px; margin: 3px 3px 0 0; font-size: .75rem; }
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
  .amostras li { font-size: .85rem; margin: 4px 0; color: var(--texto); }
  .amostras button { background: none; border: 0; color: var(--perigo);
                     cursor: pointer; font-size: .9rem; }
  .vazio { color: var(--texto2); text-align: center; padding: 80px 20px; }
  .vazio div { font-size: 3rem; margin-bottom: 10px; }
  h2.sec { font-size: 1rem; margin: 22px 0 4px; color: var(--texto); }
</style>
</head>
<body>
<header>
  <h1>🎈 Rastreador de <span>Balões</span> — PDI clássico, sem ML</h1>
  <nav>
    <button type="button" id="tab-detectar" class="ativo" onclick="aba('detectar')">🎯 Detectar</button>
    <button type="button" id="tab-calibrar" onclick="aba('calibrar')">🎨 Calibrar cores</button>
  </nav>
</header>

<!-- ============================ DETECTAR ============================ -->
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
    </details>

    <details>
      <summary>Vias de candidatos</summary>
      <input type="hidden" name="vias_enviadas" value="1">
      <div class="chks">
        <label><input type="checkbox" name="via_a" {% if vias.mascaras %}checked{% endif %}> A · máscaras</label>
        <label><input type="checkbox" name="via_b" {% if vias.hough %}checked{% endif %}> B · Hough (arcos)</label>
        <label><input type="checkbox" name="via_c" {% if vias.saturacao %}checked{% endif %}> C · saturação</label>
      </div>
      <label>Confiança de cor mínima (vias A/C) <div class="faixa">
        <input type="range" name="conf_cor" min="0" max="1" step="0.05" value="{{ v.conf_cor }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.conf_cor }}</output></div></label>
      <label>Confiança de cor mínima (via B) <div class="faixa">
        <input type="range" name="conf_cor_hough" min="0" max="1" step="0.05" value="{{ v.conf_cor_hough }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.conf_cor_hough }}</output></div></label>
      <label>Percentil de saturação (via C) <div class="faixa">
        <input type="range" name="sat_percentil" min="50" max="99" step="1" value="{{ v.sat_percentil }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.sat_percentil }}</output></div></label>
      <div class="linha2">
        <div><label>Hough dp</label><input type="number" step="0.1" name="hough_dp" value="{{ v.hough_dp }}"></div>
        <div><label>Hough param2</label><input type="number" name="hough_p2" value="{{ v.hough_p2 }}"></div>
      </div>
      <label>Hough param1</label><input type="number" name="hough_p1" value="{{ v.hough_p1 }}">
    </details>

    <details>
      <summary>Priors de domínio</summary>
      <label>Altura mínima dos balões (fração do topo) <div class="faixa">
        <input type="range" name="altura_min" min="0" max="0.5" step="0.01" value="{{ v.altura_min }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.altura_min }}</output></div></label>
      <label>Altura máxima dos balões <div class="faixa">
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
      </div>
      <div class="linha2">
        <div><label>Kernel morfologia</label>
          <input type="number" name="kernel_morf" value="{{ v.kernel_morf }}" min="3" max="15" step="2"></div>
        <div><label>Kernel máximos (watershed)</label>
          <input type="number" name="ws_kernel" value="{{ v.ws_kernel }}" min="3" max="31" step="2"></div>
      </div>
      <label>Limiar de distância (watershed) <div class="faixa">
        <input type="range" name="ws_limiar" min="0.05" max="0.6" step="0.01" value="{{ v.ws_limiar }}"
               oninput="this.parentNode.querySelector('output').value=this.value">
        <output>{{ v.ws_limiar }}</output></div></label>
      <div class="linha2">
        <div><label>Mean-shift sp</label><input type="number" name="ms_sp" value="{{ v.ms_sp }}"></div>
        <div><label>Mean-shift sr</label><input type="number" name="ms_sr" value="{{ v.ms_sr }}"></div>
      </div>
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
        <div class="kpi"><b>{{ contagem|length }}</b><span>cores</span></div>
      </div>
      <div class="hero"><img src="data:image/jpeg;base64,{{ overlay }}" alt="Detecções"></div>

      <h2 class="sec">Equipes e problemas resolvidos</h2>
      <div class="cards">
        {% for g in grupos %}
        <div class="card">
          <h3>{{ g.equipe }}</h3>
          <div>{% for p in g.problemas|sort %}<span class="prob">{{ p }}</span>{% endfor %}</div>
          <div>{% for c in g.cores|sort %}<span class="chip">{{ c }}</span>{% endfor %}</div>
          <small>centro ({{ "%.0f"|format(g.centro_ret[0]) }}, {{ "%.0f"|format(g.centro_ret[1]) }})</small>
        </div>
        {% endfor %}
      </div>

      <h2 class="sec">Contagem por cor</h2>
      <div class="cards">
        {% for cor, qtd in contagem %}
        <div class="card"><h3>{{ cor }}</h3><b style="font-size:1.4rem;color:var(--realce2)">{{ qtd }}</b></div>
        {% endfor %}
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
        <small>Todos os limiares do pipeline são calibráveis — nada é fixo no código.</small>
      </div>
    {% endif %}
  </section>
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
      normalizada: é exatamente o que o detector vê.</p>
    <div id="alvo-wrap">
      <img id="alvo" src="" alt="" style="display:none">
    </div>
  </div>
</main>

<script>
function aba(qual) {
  document.getElementById('aba-detectar').style.display = qual === 'detectar' ? 'grid' : 'none';
  document.getElementById('aba-calibrar').style.display = qual === 'calibrar' ? 'grid' : 'none';
  document.getElementById('tab-detectar').classList.toggle('ativo', qual === 'detectar');
  document.getElementById('tab-calibrar').classList.toggle('ativo', qual === 'calibrar');
}

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
    dados.erro || ('✔ ' + dados.mensagem + ' Recarregue a aba Detectar para usá-la.');
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
                                      erro="Calibração não encontrada — use a aba Calibrar.",
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

    contagem = sorted(Counter(d.cor for d in resultado["deteccoes"]).items())
    mascaras_b64 = []
    if "mostrar_mascaras" in request.form:
        mascaras_b64 = [(nome, _jpeg_b64(m)) for nome, m in resultado["mascaras"].items()]

    return render_template_string(
        _PAGINA, overlay=_jpeg_b64(resultado["overlay"]),
        grupos=resultado["grupos"], contagem=contagem,
        total=len(resultado["deteccoes"]), mascaras=mascaras_b64,
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
