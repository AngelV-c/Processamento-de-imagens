# 🎈 Rastreador de Balões — Maratona de Programação

Identifica quais problemas cada equipe resolveu a partir de uma câmera fixa —
cada cor de balão corresponde a um problema. 100% processamento digital de
imagens clássico (OpenCV), **sem machine learning**.

- **Site do projeto (GitHub Pages):** `index.html` na raiz — resultados,
  arquitetura e instruções.
- **Aplicação:** [`baloes_v2/`](baloes_v2/) — pipeline completo (7 etapas),
  interface web com calibração por clique e placar ICPC.
- **Decisões de arquitetura e fontes da pesquisa:**
  [`baloes_v2/ARQUITETURA.md`](baloes_v2/ARQUITETURA.md).

## Rodar localmente

```bash
cd baloes_v2
python -m pip install -r requirements.txt
python app.py        # http://localhost:5000
```
