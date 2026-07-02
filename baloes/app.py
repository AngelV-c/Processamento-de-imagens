"""Interface web do rastreador de balões (Flask).

Sobe um servidor local com uma página única: o usuário envia uma foto,
escolhe os métodos e parâmetros, e recebe o overlay anotado, o relatório
por equipe e (opcionalmente) as máscaras por cor.

Uso:
    python app.py            # http://localhost:5000
    python app.py --porta 8080
"""

import argparse
import base64
import json
import os
import sys
from collections import Counter

import cv2
from flask import Flask, render_template_string, request

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "src"))

from main import executar_pipeline, montar_config

app = Flask(__name__)

_PAGINA = """
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rastreador de Balões</title>
<style>
  :root { --borda: #d0d4da; --fundo: #f6f7f9; --realce: #14532d; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; margin: 0; background: var(--fundo); color: #1a202c; }
  header { background: var(--realce); color: #fff; padding: 14px 24px; }
  header h1 { margin: 0; font-size: 1.25rem; }
  main { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
  form { background: #fff; border: 1px solid var(--borda); border-radius: 8px;
         padding: 16px; display: flex; flex-wrap: wrap; gap: 14px; align-items: end; }
  label { display: flex; flex-direction: column; font-size: .8rem; gap: 4px; color: #4a5568; }
  input, select { padding: 6px 8px; border: 1px solid var(--borda); border-radius: 6px; font-size: .9rem; }
  .chk { flex-direction: row; align-items: center; gap: 6px; padding-bottom: 8px; }
  button { background: var(--realce); color: #fff; border: 0; border-radius: 6px;
           padding: 10px 22px; font-size: .95rem; cursor: pointer; }
  button:hover { opacity: .9; }
  .aviso { background: #fef9c3; border: 1px solid #eab308; border-radius: 6px;
           padding: 10px 14px; margin: 16px 0; font-size: .9rem; }
  .erro  { background: #fee2e2; border-color: #ef4444; }
  section { margin-top: 24px; }
  h2 { font-size: 1.05rem; border-bottom: 2px solid var(--borda); padding-bottom: 6px; }
  img.resultado { max-width: 100%; border: 1px solid var(--borda); border-radius: 8px; }
  table { border-collapse: collapse; width: 100%; background: #fff; font-size: .9rem; }
  th, td { border: 1px solid var(--borda); padding: 8px 12px; text-align: left; }
  th { background: #edf2f7; }
  .badge { display: inline-block; background: #e2e8f0; border-radius: 10px;
           padding: 2px 10px; margin: 2px; font-size: .8rem; }
  .mascaras { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
  .mascaras figure { margin: 0; }
  .mascaras figcaption { font-size: .8rem; color: #4a5568; padding: 4px 0; }
  .mascaras img { width: 100%; border: 1px solid var(--borda); border-radius: 6px; }
</style>
</head>
<body>
<header><h1>🎈 Rastreador de Balões — maratona de programação</h1></header>
<main>
<form method="post" action="/processar" enctype="multipart/form-data">
  <label>Imagem
    <input type="file" name="imagem" accept="image/*" required>
  </label>
  <label>Segmentação
    <select name="segmentacao">
      <option value="meanshift" selected>mean-shift + HSV (recomendado)</option>
      <option value="hsv">HSV puro</option>
    </select>
  </label>
  <label>Detecção
    <select name="metodo">
      <option value="watershed" selected>watershed (recomendado)</option>
      <option value="fourier">watershed + Fourier</option>
      <option value="contorno">contorno simples</option>
      <option value="shape_first">forma primeiro, cor depois</option>
    </select>
  </label>
  <label>Largura máx (px)
    <input type="number" name="largura" value="1200" min="400" max="4000">
  </label>
  <label>eps do DBSCAN (vazio = automático)
    <input type="number" step="0.1" name="eps" placeholder="cm ou px">
  </label>
  <label class="chk"><input type="checkbox" name="clahe"> CLAHE</label>
  <label class="chk"><input type="checkbox" name="wb"> White balance</label>
  <label class="chk"><input type="checkbox" name="mostrar_mascaras"> Mostrar máscaras</label>
  <button type="submit">Processar</button>
</form>

{% if erro %}<div class="aviso erro">{{ erro }}</div>{% endif %}
{% for aviso in avisos %}<div class="aviso">⚠ {{ aviso }}</div>{% endfor %}

{% if overlay %}
<section>
  <h2>Resultado — {{ total }} balão(ões), {{ grupos|length }} equipe(s)</h2>
  <img class="resultado" src="data:image/jpeg;base64,{{ overlay }}" alt="Detecções">
</section>

<section>
  <h2>Relatório por equipe</h2>
  <table>
    <tr><th>Equipe</th><th>Posição</th><th>Cores</th><th>Problemas resolvidos</th></tr>
    {% for g in grupos %}
    <tr>
      <td>{{ g.equipe }}</td>
      <td>({{ "%.0f"|format(g.centro_ret[0]) }}, {{ "%.0f"|format(g.centro_ret[1]) }})</td>
      <td>{% for c in g.cores|sort %}<span class="badge">{{ c }}</span>{% endfor %}</td>
      <td><strong>{{ g.problemas|sort|join(", ") }}</strong></td>
    </tr>
    {% endfor %}
  </table>
</section>

<section>
  <h2>Contagem por cor</h2>
  <table>
    <tr><th>Cor</th><th>Balões</th></tr>
    {% for cor, qtd in contagem %}
    <tr><td>{{ cor }}</td><td>{{ qtd }}</td></tr>
    {% endfor %}
  </table>
</section>

{% if mascaras %}
<section>
  <h2>Máscaras por cor</h2>
  <div class="mascaras">
    {% for nome, b64 in mascaras %}
    <figure>
      <img src="data:image/jpeg;base64,{{ b64 }}" alt="máscara {{ nome }}">
      <figcaption>{{ nome }}</figcaption>
    </figure>
    {% endfor %}
  </div>
</section>
{% endif %}
{% endif %}
</main>
</body>
</html>
"""


def _jpeg_b64(imagem) -> str:
    _, buf = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii")


@app.route("/", methods=["GET"])
def pagina_inicial():
    return render_template_string(_PAGINA, overlay=None, erro=None, avisos=[])


@app.route("/processar", methods=["POST"])
def processar():
    arquivo = request.files.get("imagem")
    if arquivo is None or arquivo.filename == "":
        return render_template_string(_PAGINA, overlay=None, avisos=[],
                                      erro="Nenhuma imagem enviada.")

    dir_upload = os.path.join(_DIR, "output", "uploads")
    os.makedirs(dir_upload, exist_ok=True)
    caminho = os.path.join(dir_upload, os.path.basename(arquivo.filename))
    arquivo.save(caminho)

    config = montar_config(
        os.path.join(_DIR, "config", "cores.json"),
        os.path.join(_DIR, "config", "params.json"),
    )

    eps_txt = request.form.get("eps", "").strip()
    try:
        resultado = executar_pipeline(
            caminho, config,
            segmentacao=request.form.get("segmentacao", "meanshift"),
            metodo=request.form.get("metodo", "watershed"),
            largura=int(request.form.get("largura", 1200)),
            usar_clahe="clahe" in request.form,
            usar_wb="wb" in request.form,
            caminho_homografia=os.path.join(_DIR, "config", "homografia.json"),
            eps=float(eps_txt) if eps_txt else None,
        )
    except Exception as exc:  # superfície de erro única da UI
        return render_template_string(_PAGINA, overlay=None, avisos=[],
                                      erro=f"Falha no processamento: {exc}")

    contagem = sorted(Counter(d.cor for d in resultado["deteccoes"]).items())

    mascaras_b64 = []
    if "mostrar_mascaras" in request.form:
        mascaras_b64 = [(nome, _jpeg_b64(m)) for nome, m in resultado["mascaras"].items()]

    return render_template_string(
        _PAGINA,
        overlay=_jpeg_b64(resultado["overlay"]),
        grupos=resultado["grupos"],
        contagem=contagem,
        total=len(resultado["deteccoes"]),
        mascaras=mascaras_b64,
        avisos=resultado["avisos"],
        erro=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface web do rastreador de balões.")
    parser.add_argument("--porta", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.porta, debug=False)
