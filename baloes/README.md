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
