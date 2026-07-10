# Arquitetura — por que o código foi refeito assim

Este documento registra as cinco recomendações que nasceram das limitações
encontradas durante o desenvolvimento iterativo do `baloes/` original, e que
guiaram a reescrita completa desta versão.

## O diagnóstico

Os algoritmos nunca foram o problema — watershed, descritores de Fourier e
filtros de forma funcionaram bem desde o início. O que quebrou repetidamente
foi sempre a mesma coisa: **valores absolutos calibrados numa imagem sendo
aplicados em outra**. O amarelo do `cores.json` exigia S≥120; os balões do
Teste_3 tinham S entre 55 e 116 e sumiam. A maioria dos balões do Teste_3 é
**branca** — cor que nenhuma segmentação por matiz alcança, então o pipeline
antigo nem tinha como representá-la.

## As cinco recomendações

### 1. Calibração por local, não configuração fixa

Não existe `cores.json` universal: cada ginásio tem iluminação, câmera e
balões diferentes. Em vez de faixas HSV fixas no código, o operador clica em
1+ balão de cada cor na foto do próprio local (2 minutos) e o sistema constrói
os modelos a partir dali: mediana circular do matiz, percentis de S/V/L e
distribuição LAB. Cores acromáticas (branco/prata) são detectadas
automaticamente e modeladas por brilho + croma a-b. A configuração vira um
artefato por evento (`config/calibracoes/<evento>.json`), não código.

### 2. Limiares relativos à imagem, não absolutos

- Iluminação normalizada na entrada (gray-world com ganhos limitados + CLAHE
  no V), para que o resto do pipeline veja imagens parecidas.
- Pisos de S/V/L derivados de percentis da amostra calibrada.
- A via de saturação usa percentil da distribuição da própria cena — o fundo
  neutro domina o histograma, os balões são os outliers do topo.
- O matiz (H) é a única grandeza quase invariante entre cenas; S e V nunca
  têm valores fixos no pipeline.

### 3. Pontuação combinada em vez de cascata de filtros rígidos

O pipeline antigo era uma corrente de ANDs (área → circularidade → solidity →
aspect_ratio → cor): um candidato excelente em quatro métricas morria por 0.01
na quinta. Agora:

    score = 0.20·circularidade + 0.15·solidity + 0.25·fourier + 0.40·conf_cor
    aceita se score ≥ limiar único (0.75)

Um balão parcialmente ocluído com circularidade 0.45 mas espectro e cor
perfeitos sobrevive. O aspect_ratio vira penalidade suave, não porta.

### 4. Geração de candidatos redundante (multi-via)

Detectores fracos diferentes falham em lugares diferentes:

- **Via A** — máscaras de cor calibradas + watershed (precisa, cor conhecida)
- **Via B** — HoughCircles: o acumulador vota com ARCOS de borda, então um
  balão 40% ocluído ainda gera pico onde nenhum blob fechado existe. Como a
  forma de um círculo sintético é perfeita por construção, esta via exige
  confiança de cor bem mais alta.
- **Via C** — saturação adaptativa (percentil da cena) + watershed, pega
  cromáticos que escaparam dos modelos.

A união é deduplicada por distância entre centros; o maior score vence.
Foi a via B que recuperou os balões do cluster denso da Foto_Teste_2 que o
pipeline antigo nunca alcançou.

### 5. Ground truth e avaliação automática desde o dia 1

`config/gabaritos.json` guarda o total (e contagens por cor quando
confirmadas) de cada imagem de teste; `tools/avaliar.py` roda o pipeline
contra TODAS elas e reporta a taxa por imagem e global. Qualquer mudança de
parâmetro mostra na hora se regrediu em outra cena — o overfitting da fase
inicial aconteceu exatamente porque o feedback era manual e uma imagem por
vez.

## Pesquisa de técnicas e substituições (2ª geração)

Uma revisão da literatura clássica de PDI orientou a troca de três
componentes por técnicas com fundamento melhor para as dificuldades
específicas que encontramos:

| Dificuldade | Antes | Agora | Base |
|---|---|---|---|
| Oclusão parcial (cluster denso) | HoughCircles | **Círculos por segmentos de arco**: Canny → segmentos conectados → ajuste algébrico de Kåsa → validação por resíduo mediano e cobertura angular. Cada arco gera no máximo um círculo com medida direta de qualidade. | Família [EDCircles](https://www.sciencedirect.com/science/article/abs/pii/S003132031500446X) (detecção por arcos de segmentos de borda); ajuste robusto a [dados parciais com outliers](https://arxiv.org/pdf/2508.03720); [detecção bottom-up com parametrização adaptativa](https://pmc.ncbi.nlm.nih.gov/articles/PMC12031632/) |
| Limiar de saturação dependente da cena | percentil global de S | **MSER** no canal S: regiões maximamente estáveis através de limiares — invariante a transformações monotônicas de iluminação, baixo custo. O canal L foi testado e removido (propunha camisetas idênticas ao branco em cor). | [Matas et al., BMVC 2002](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/matas_bmvc2002.pdf); [visão geral MSER](https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions) |
| Constância de cor entre câmeras | gray-world (média, p=1) | **Shades-of-Gray**: norma de Minkowski p=6 por canal — pixels claros pesam mais, aproxima o white-patch sem a fragilidade a um pixel estourado. Ganhos continuam limitados. | [Finlayson & Trezzi, Shades of Gray and Colour Constancy](https://www.researchgate.net/publication/221502067_Shades_of_Gray_and_Colour_Constancy) |
| Balão branco vs superfície branca | só portões de L | + **brilho especular** como bônus de score: balão de látex é brilhante e mostra um pequeno reflexo das luzes; paredes/camisetas são foscas. Nunca é porta, só bônus. | física do material (reflexão especular vs difusa) |

E uma restrição do DOMÍNIO virou estrutura: nas regras de maratona cada
equipe tem no máximo **um balão por cor** (um por problema resolvido). O
agrupamento usa isso em dois níveis: cluster com cor duplicada é re-separado
com eps menor (caso típico: duas equipes fundidas por eps grande); se nem o
eps mínimo separa, a duplicata de menor score é descartada — pela regra, uma
delas é necessariamente um falso positivo.

## Priors de domínio que emergiram dos testes

Cor e forma não separam balão branco de teto branco, luminária ou camiseta
branca — são fisicamente a mesma cor. Três critérios de CONTEXTO resolvem:

- **Faixa de altura** (`fracao_altura_min/max`): balões flutuam presos às
  mesas — teto acima da faixa, pessoas e mesas abaixo.
- **Isolamento**: o anel ao redor de um balão é fundo; o anel de um pedaço
  de teto retalhado pelo watershed é mais teto.
- **Contexto de componente**: um recorte de parede vem de um componente
  conexo gigante; um cacho real de balões tem componente ≤ ~6 balões.

## Resultados (avaliação automática)

| Imagem | Pipeline antigo | 1ª geração v2 | 2ª geração (arcos+MSER+SoG) |
|---|---|---|---|
| Foto_Teste_2 (19 balões) | 16 (84%) | 18 (95%) | **19 (100%)** |
| Teste_3 (25 balões) | 10 (40%)* | 11 (44%) | **15 (60%)** |
| Global | 59% | 66% | **77%** |

Na 2ª geração, a avaliação por posição anotada no Teste_3 (12 balões
confirmados) reporta precisão 60%, recall 75% e acurácia de cor 89% —
e dois dos "falsos positivos" apontados são balões reais fora da anotação
parcial.

\* Os 10 antigos incluíam 5 balões brancos rotulados como "amarelo" e 1
"azul" inexistente — o número comparável real era menor.

## Limitações conhecidas

- Luminárias: brancas, brilhantes e redondas — 2 ainda passam no Teste_3.
- Rosa pálido vs branco sob luz quente é ambíguo até para o olho humano.
- Rosa e vermelho se sobrepõem na fronteira magenta do matiz.
- Se a restrição de PDI clássico cair um dia: um detector pequeno (YOLO)
  treinado com ~200 fotos anotadas resolve oclusão e iluminação de uma vez.
