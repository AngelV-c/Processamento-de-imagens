# Detecção de Balões em Maratonas de Programação

Pipeline clássico de processamento digital de imagens (OpenCV, sem ML) que
identifica quais cores de balão cada equipe possui sobre a mesa, a partir de
uma câmera com visão frontal-superior.

## Estrutura

```
baloes/
  config/cores.json        # faixas HSV de cada cor de balão
  src/preprocess.py        # carregamento e pré-processamento
  src/segmentation.py      # máscaras por cor + morfologia
  src/detection.py         # contornos + circularidade → Deteccao
  src/visualize.py         # overlays de debug
  calibrate_hsv.py         # ferramenta interativa de calibração
  main.py                  # CLI – etapas 1-4
  output/                  # gerado em runtime
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python main.py --imagem caminho/para/foto.jpg
```

Opções completas:

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--imagem` | (obrigatório) | Caminho da imagem de entrada |
| `--config` | `config/cores.json` | Arquivo de configuração de cores |
| `--saida` | `output/` | Diretório de saída |
| `--largura` | sem redimensionamento | Largura máxima em pixels |
| `--clahe` | desligado | Equalização CLAHE no canal V |
| `--wb` | desligado | Gray-world white balance |
| `--blur` | 5 | Kernel do blur gaussiano (valor ímpar) |

Exemplo com normalização de iluminação:

```bash
python main.py --imagem foto.jpg --largura 1280 --clahe --wb --blur 7
```

## Saída

Após a execução, o diretório `output/` conterá:

- `overlay.jpg` — imagem com círculos e rótulos sobre cada balão detectado
- `mascara_<cor>.jpg` — máscara binária de cada cor (útil para ajustar faixas)

No terminal é impresso um resumo com a contagem de balões por cor.

## Calibração de cores

Use `calibrate_hsv.py` para encontrar as faixas HSV corretas para uma nova cor
ou para ajustar as existentes:

```bash
python calibrate_hsv.py --imagem foto.jpg --cor azul
```

- Ajuste os sliders em tempo real e observe a máscara resultante.
- Pressione **s** para salvar a faixa atual (use duas vezes para o vermelho).
- Pressione **q** para sair: o trecho JSON pronto será impresso no terminal.
- Cole o resultado dentro do array `"cores"` em `config/cores.json`.

### Vermelho e o matiz circular

O vermelho no espaço HSV do OpenCV (H: 0–179) ocupa **duas faixas** nas pontas
da escala: aproximadamente H ∈ [0, 10] e H ∈ [170, 179]. Por isso, em
`cores.json`, a cor `"vermelho"` tem **dois objetos** em `"faixas"`. Ao calibrar,
salve a primeira faixa (s), ajuste os sliders para a segunda faixa, salve de
novo (s), e depois saia (q).

## Cor reservada: laranja das mesas

**A laranja das mesas não está em `config/cores.json` e não deve ser adicionada.**
Ela é intencionalmente excluída para evitar falsos positivos nas superfícies das
mesas. Se um novo conjunto de mesas usar cor diferente, revise as faixas HSV das
cores de balão que mais se aproximarem da nova cor das mesas.

## Contrato de saída (para etapas futuras)

A função `detectar_baloes()` em `src/detection.py` retorna `list[Deteccao]`:

```python
@dataclass
class Deteccao:
    cor: str        # nome da cor
    cx: float       # centroide x (pixels)
    cy: float       # centroide y (pixels)
    area: float     # área do contorno (pixels²)
    circularidade: float  # 4π·A/P²
    raio: float     # raio aproximado (pixels)
```

Essa lista alimentará, em etapas futuras:
- **Etapa 5**: correção de perspectiva por homografia
- **Etapa 6**: agrupamento de balões por equipe via DBSCAN

## Ajuste de parâmetros

Todos os limiares ficam em `config/cores.json`, seção `"deteccao"`:

| Parâmetro | Padrão | Significado |
|-----------|--------|-------------|
| `area_minima_relativa` | 0.0005 | Fração mínima da área total da imagem |
| `area_maxima_relativa` | 0.15 | Fração máxima da área total da imagem |
| `circularidade_minima` | 0.7 | Limiar de 4π·A/P² |
| `kernel_morfologia` | 7 | Tamanho do kernel de abertura+fechamento |
