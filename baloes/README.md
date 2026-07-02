# Rastreio de Balões em Maratonas de Programação

Pipeline clássico de processamento digital de imagens (OpenCV, **sem ML**) que
identifica, a partir de UMA câmera fixa com visão frontal-superior, quais
problemas cada equipe resolveu — cada cor de balão corresponde a um problema.

## Pipeline (etapas 1–7)

1. **Carga** da imagem
2. **Pré-processamento** — redimensionamento, HSV, CLAHE/white-balance opcionais, blur
3. **Segmentação por cor** — limiarização HSV (+ mean-shift) e morfologia
4. **Detecção** — watershed + filtros de forma (circularidade, solidity, aspect ratio, Fourier)
5. **Homografia** — retifica os *centroides* para coordenadas métricas (`cv2.perspectiveTransform`)
6. **Agrupamento** — DBSCAN sobre os centroides retificados agrupa balões por equipe
7. **Relatório** — conjunto de cores por grupo → letras dos problemas por equipe

## Premissas do domínio

- Dentro de uma equipe cada cor aparece **no máximo uma vez** → o relatório é
  presença/ausência, sem contagem de duplicatas.
- **A laranja é a cor das mesas** — é RESERVADA e nunca aparece em
  `config/cores.json` como cor de balão.
- A maior distância entre balões da MESMA equipe é bem menor que a menor
  distância entre equipes DIFERENTES → existe uma faixa de `eps` que separa as
  equipes sem erro (base matemática do DBSCAN aqui).
- Frame único, sem memória entre frames; balão oculto não é contabilizado.

## Estrutura

```
baloes/
  config/
    cores.json           # faixas HSV + letra do problema de cada cor
    params.json          # limiares de detecção, watershed, fourier, eps, minPts
    homografia.json      # gerado pela calibração (matriz H + escala métrica)
    gabaritos.json       # ground truth p/ avaliação automática
  src/
    tipos.py             # dataclasses Deteccao e Grupo (contrato entre etapas)*
    preprocess.py        # etapa 2
    segmentation*.py     # etapa 3 (hsv, meanshift, lab, superpixel)
    detection*.py        # etapa 4 (contorno, watershed, fourier, hough, shape_first)
    homography.py        # etapa 5
    grouping.py          # etapa 6 (DBSCAN)
    reporting.py         # etapa 7
    visualize.py         # overlays (detecções + cascos dos grupos)
  tools/
    calibrate_hsv.py         # calibra faixas HSV das cores (trackbars)
    calibrate_homografia.py  # 4 cliques (ou --pontos) → homografia.json
    diagnostico_eps.py       # curva de vizinho-mais-próximo → eps sugerido
    avaliar.py               # roda o pipeline contra gabaritos.json
  main.py                # CLI — orquestra as etapas 1–7
  app.py                 # interface web (Flask)
  output/                # gerado em runtime (fora do git)
```

\* O arquivo se chama `tipos.py` (e não `types.py`) porque `types` colide com
o módulo homônimo da biblioteca padrão do Python, que numpy/sklearn importam.

## Instalação

```bash
pip install -r requirements.txt
```

## Pipeline v2 (recomendado para locais novos)

O método `--metodo v2` implementa a arquitetura reestruturada:

1. **Calibração por local, não configuração fixa** — os modelos de cor vêm de
   cliques em balões reais da cena (`tools/calibrar_cores.py`), não de faixas
   HSV universais. Cores **acromáticas** (branco/prata) são detectadas
   automaticamente e modeladas por brilho L + croma a-b — impossível por matiz.
   Clique num balão perto E num longe da mesma cor: as amostras são fundidas.
2. **Limiares relativos** — iluminação normalizada na entrada (gray-world com
   ganhos limitados + CLAHE); pisos de S/V/L derivados dos percentis da
   amostra; via de saturação usa percentil da distribuição da própria cena.
3. **Score combinado** — `w1·circ + w2·solidity + w3·fourier + w4·conf_cor ≥
   limiar único`, em vez da cascata de ANDs onde um candidato excelente em
   4 métricas morre por 0.01 na quinta.
4. **Candidatos multi-via** — (A) máscaras calibradas + watershed;
   (B) HoughCircles, que vota com arcos e funciona sob oclusão parcial;
   (C) saturação adaptativa + watershed. União deduplicada por distância.
5. **Contexto de cena** — três priors de domínio configuráveis: faixa de
   altura onde balões podem estar (`fracao_altura_min/max`), isolamento
   (anel ao redor não pode ser da mesma cor) e razão de componente (pedaço
   de teto vem de componente conexo gigante; balão não).

```bash
# 1. calibrar as cores da cena (headless: --amostra "cor:x,y:PROBLEMA")
python tools/calibrar_cores.py --imagem foto_local.jpg --largura 1200 \
    --amostra "vermelho:274,350:A" --amostra "branco:357,207:B"

# 2. rodar
python main.py --imagem foto_local.jpg --metodo v2 --largura 1200
```

Resultado na avaliação automática (`tools/avaliar.py --metodo v2`):
v2 66% global vs 59% do baseline — com a diferença qualitativa de que o
baseline conta rótulos errados como acerto (balões brancos detectados como
"amarelo"), enquanto o v2 suporta branco como cor de verdade.

## Passo a passo de calibração (por local/câmera)

Cada ginásio tem iluminação e câmera diferentes — **não existe configuração
universal**. Com a câmera na posição definitiva:

1. **Cores** — `python tools/calibrate_hsv.py --imagem foto_local.jpg` e ajuste
   as faixas de cada cor; cole o resultado em `config/cores.json`.
2. **Homografia** — clique os 4 cantos de um retângulo de dimensões conhecidas
   sobre o plano das mesas:
   ```bash
   python tools/calibrate_homografia.py --imagem foto_local.jpg \
       --largura-cm 400 --altura-cm 300
   # headless: acrescente --pontos "x1,y1" "x2,y2" "x3,y3" "x4,y4"
   ```
3. **eps do DBSCAN** — `python tools/diagnostico_eps.py --imagem foto_local.jpg`
   mostra a curva de distâncias; escolha um eps dentro do "salto" e registre em
   `params.json` (`agrupamento.eps_cm`).
4. **Validação** — anote o total real de balões da foto em
   `config/gabaritos.json` e rode `python tools/avaliar.py`.

## Execução

```bash
# CLI — pipeline completo
python main.py --imagem foto.jpg --largura 1200 --debug

# Interface web
python app.py            # abra http://localhost:5000
```

Sem `config/homografia.json` o pipeline avisa e agrupa em **pixels**, usando
`agrupamento.eps_px_fallback` — funcional, mas o eps pode não servir para
todas as fileiras de mesas.

### Flags principais do main.py

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--imagem` | (obrigatório) | Imagem de entrada |
| `--segmentacao` | `meanshift` | `hsv`, `meanshift`, `lab`, `superpixel` |
| `--metodo` | `watershed` | `contorno`, `watershed`, `fourier`, `hough`, `shape_first` |
| `--largura` | 1200 | Largura máxima (px) |
| `--eps` | automático | Sobrescreve o raio do DBSCAN (cm com homografia, px sem) |
| `--clahe` / `--wb` | off | Normalização de iluminação |
| `--debug` | off | Salva máscaras por cor e vista retificada |

## Saídas (em `output/`)

- `overlay.jpg` — imagem anotada: círculo/rótulo por balão + casco convexo de
  cada equipe com os problemas resolvidos
- `relatorio.json` — relatório estruturado por equipe
- `mascara_<cor>.jpg` (com `--debug`) e `vista_retificada.jpg` (com homografia)

## Coerência de unidades

Com homografia calibrada, o plano retificado usa `px_por_cm` (padrão 2 px/cm);
o `eps_cm` de `params.json` é convertido internamente. Sem homografia, tudo é
pixel da imagem redimensionada — inclusive `eps_px_fallback`. Áreas de detecção
são sempre **relativas** ao total de pixels da imagem.

## Detalhes técnicos importantes

- OpenCV usa H∈[0,179], S,V∈[0,255] — nunca a escala 0–360.
- O vermelho ocupa as duas pontas do matiz circular → duas faixas com OR.
- Morfologia: abertura (remove ruído) ANTES de fechamento (fecha reflexos).
- Circularidade protegida contra perímetro zero.
- `perspectiveTransform` exige shape `(N,1,2)` e `dtype float32`.
- `min_pts=1` no DBSCAN: equipe com um único balão não pode virar "ruído".
- Filtro de forma tem modo `score` (soma ponderada) além da cascata rígida —
  ver `deteccao.modo_filtro` em `params.json`.
