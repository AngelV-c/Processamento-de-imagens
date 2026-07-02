# Rastreador de Balões v2 — reescrita completa

Sistema que identifica, a partir de UMA câmera fixa com visão frontal-superior,
quais problemas cada equipe de uma maratona de programação resolveu — cada cor
de balão corresponde a um problema. 100% processamento digital de imagens
clássico (OpenCV), **sem ML**.

Esta é a reescrita do `baloes/` original em torno das cinco recomendações de
arquitetura que nasceram das limitações encontradas lá — o porquê de cada
decisão está em **[ARQUITETURA.md](ARQUITETURA.md)**. Em resumo:

1. **Calibração por local** — modelos de cor nascem de cliques em balões da
   própria cena, não de faixas HSV fixas; suporta cores acromáticas (branco).
2. **Limiares relativos** — iluminação normalizada na entrada; pisos derivados
   de percentis da amostra e da cena, nunca valores absolutos.
3. **Score combinado** — `w·(circularidade, solidity, fourier, conf_cor)` com
   limiar único, no lugar da cascata de ANDs.
4. **Candidatos multi-via** — máscaras calibradas + HoughCircles (arcos →
   oclusão parcial) + saturação adaptativa, união deduplicada.
5. **Avaliação automática** — gabarito por imagem + `tools/avaliar.py`.

## Estrutura

```
baloes_v2/
  ARQUITETURA.md         # as recomendações e o porquê de cada decisão
  config/
    params.json          # todos os limiares (nenhum valor mágico no código)
    calibracoes/         # calibrações por cena (Foto_Teste_2, Teste_3)
    gabaritos.json       # ground truth para avaliação automática
    homografia.json      # gerado pela calibração de perspectiva (opcional)
  src/
    tipos.py             # Deteccao e Grupo — contrato entre etapas
    preprocess.py        # carga + normalização de iluminação
    calibracao.py        # modelos de cor: construção, casamento, classificação
    formas.py            # circularidade, solidity, Fourier, watershed
    deteccao.py          # multi-via + score combinado + priors de domínio
    homografia.py        # etapa 5 — retificação de centroides
    agrupamento.py       # etapa 6 — DBSCAN (eps em cm, min_pts=1)
    relatorio.py         # etapa 7 — problemas por equipe
    visualizacao.py      # overlays
  tools/
    calibrar_cores.py    # cliques → calibração local (interativo ou headless)
    calibrar_homografia.py
    diagnostico_eps.py   # curva de vizinho-mais-próximo → eps sugerido
    avaliar.py           # mede contra gabaritos.json
  main.py                # CLI — pipeline completo 1–7
  app.py                 # interface web (Flask)
```

## Instalação

```bash
pip install -r requirements.txt
```

## Como rodar

**Interface web:**
```bash
python app.py            # abra http://localhost:5000
```
Envie a foto, escolha a calibração da cena no menu e clique Processar.

**Linha de comando (imagens de teste, calibrações prontas):**
```bash
python main.py --imagem Foto_Teste_2.jpeg \
    --calibracao config/calibracoes/Foto_Teste_2.json --eps 68

python main.py --imagem Teste_3.jpg \
    --calibracao config/calibracoes/Teste_3.json --eps 90
```
Saídas em `output/`: `overlay.jpg` (balões + cascos das equipes) e
`relatorio.json`. Com `--debug`, também as máscaras por cor.

## Fluxo para um local novo (4 passos)

```bash
# 1. Calibrar cores: clique em 1+ balão de cada cor (perto E longe da mesma
#    cor — as amostras são fundidas). Headless: --amostra "cor:x,y:LETRA"
python tools/calibrar_cores.py --imagem foto_local.jpg --largura 1200

# 2. (Opcional) Calibrar perspectiva: 4 cantos de um retângulo conhecido
python tools/calibrar_homografia.py --imagem foto_local.jpg \
    --largura-cm 400 --altura-cm 300

# 3. Escolher o eps: a curva mostra o salto entre intra e intergrupo
python tools/diagnostico_eps.py --imagem foto_local.jpg

# 4. Rodar
python main.py --imagem foto_local.jpg
```

**Importante:** use o mesmo `--largura` (padrão 1200) na calibração e na
execução — as coordenadas dos cliques são relativas à imagem redimensionada.

## Validação

```bash
python tools/avaliar.py
```

| Imagem | Pipeline antigo | Este pipeline |
|---|---|---|
| Foto_Teste_2 (19 balões) | 16 (84%) | **18 (95%)** |
| Teste_3 (25 balões) | 10 (40%)* | 11 (44%) |
| Global | 59% | **66%** |

\* incluía 5 brancos rotulados "amarelo" e 1 "azul" inexistente.

Limitações conhecidas e o caminho para melhorar estão em
[ARQUITETURA.md](ARQUITETURA.md#limitações-conhecidas).
