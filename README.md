# Cubify Anything - Ferramentas de Inferência e Visualização

Este repositório é um fork do [apple/ml-cubifyanything](https://github.com/apple/ml-cubifyanything) com ferramentas adicionais para exportação de predições, inferência em imagens arbitrárias e visualização offline.

Instalar o repositorio a partir do wsl
--------------------------------------------------------

git clone https://github.com/NishinoTSK/ml-cubifyanything
cd ml_cubifyanythin
virtualenv ambiente
source ambiente/bin/activate
Criar pasta models na pasta ml-cubifyanything e colocar os pesos dentro dela cutr_rgb.pth baixados daqui (RGB https://github.com/apple/ml-cubifyanything?tab=readme-ov-file.)


--------------------------------------------------------
## Novas Funcionalidades

### 1. Exportação de Predições do Demo (`tools/demo.py`)

Agora é possível salvar as predições do modelo em arquivos JSON para análise posterior ou visualização offline.

```bash
python tools/demo.py data/val.txt \
    --video-ids 42898570 \
    --model-path models/cutr_rgb.pth \
    --save-preds-dir outputs/preds \
    --device cuda
```

**Parâmetro novo:** `--save-preds-dir outputs/preds`

Para cada frame processado, será criado um arquivo JSON com:
- `video_id` e `timestamp` do frame
- `image_size_hw`: dimensões da imagem
- `detections`: lista de detecções com:
  - `bbox_xyxy`: bounding box 2D em formato [x1, y1, x2, y2]
  - `score`: confiança da detecção (0-1)
  - `class_id`: classe (foreground/background)
- `boxes_3d` (se disponível): caixas 3D com centro, dimensões e rotação

**Exemplo de JSON salvo:**
```json
{
  "video_id": 48458427,
  "timestamp": 116505.351959,
  "image_size_hw": [768, 1024],
  "detections": [
    {
      "bbox_xyxy": [100.5, 200.3, 300.7, 400.2],
      "score": 0.854,
      "class_id": 1
    }
  ],
  "boxes_3d": {
    "gravity_center_xyz": [[1.2, 0.5, 2.1]],
    "dims_lhw": [[0.3, 0.4, 0.2]],
    "R_3x3": [[[...]]]
  }
}
```

---

### 2. Inferência em Imagens Arbitrárias (`tools/infer_image.py`)

Execute o modelo em qualquer imagem PNG/JPG, mesmo fora do dataset CA-1M.

#### Modelo RGB (apenas imagem)

```bash
python tools/infer_image.py \
    --image "minha_foto.png" \
    --model-path models/cutr_rgb.pth \
    --device cuda \
    --score-thresh 0.25 \
    --max-edge 1024
```

#### Modelo RGB-D (imagem + profundidade)

```bash
python tools/infer_image.py \
    --image "minha_foto.png" \
    --depth "minha_profundidade.png" \
    --model-path models/cutr_rgbd.pth \
    --device cuda
```

**Parâmetros:**

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--image` | Caminho para imagem RGB (obrigatório) | - |
| `--model-path` | Caminho para o checkpoint .pth (obrigatório) | - |
| `--device` | Dispositivo: cpu, cuda ou mps | cpu |
| `--score-thresh` | Limiar de confiança para detecções | 0.25 |
| `--max-edge` | Redimensiona imagem se o lado maior exceder este valor. Use 0 para desativar | 1024 |
| `--out-json` | Caminho de saída do JSON. Padrão: `<imagem>_inf.json` | - |
| `--fx`, `--fy`, `--cx`, `--cy` | Parâmetros intrínsecos da câmera (opcional) | estimado |
| `--depth` | Imagem de profundidade UInt16 em mm (apenas para modelo RGB-D) | - |

**Sobre os intrínsecos:**
- Se não informados, são estimados com base nas dimensões da imagem
- O modelo pode funcionar sem intrínsecos precisos, mas resultados 3D serão aproximados
- Se você redimensionar a imagem (`--max-edge`) e informar intrínsecos, eles serão ajustados automaticamente

---

### 3. Visualização 2D Offline (`tools/visualize_preds.py`)

Desenha bounding boxes 2D sobre a imagem e salva como PNG.

```bash
python tools/visualize_preds.py \
    --image "minha_foto.png" \
    --pred-json "minha_foto_inf.json" \
    --score-thresh 0.25 \
    --line-width 3
```

**Parâmetros:**

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--image` | Imagem original | - |
| `--pred-json` | JSON de predições | - |
| `--out` | Caminho de saída. Padrão: `<imagem>_inf.png` | - |
| `--score-thresh` | Filtra detecções abaixo deste score | 0.0 |
| `--no-labels` | Não mostrar labels nas caixas | - |
| `--line-width` | Espessura das linhas das caixas | 3 |

---

### 4. Visualização 3D Offline com Rerun (`tools/rerun_visualize_saved_preds.py`)

Visualiza predições no Rerun com suporte a 2D e 3D.

```bash
python tools/rerun_visualize_saved_preds.py \
    --image "minha_foto.png" \
    --pred-json "minha_foto_inf.json" \
    --application-id "minha_cena"
```

**Funcionalidades:**
- Mostra imagem RGB
- Sobrepõe bounding boxes 2D
- Mostra caixas 3D no espaço (se disponível no JSON)
- Salva automaticamente um arquivo `.rrd` para visualização posterior

**Arquivo .rrd gerado:**
- Padrão: `<imagem>_inf.rrd`
- Pode ser aberto posteriormente no Rerun Viewer

---

## Fluxo de Trabalho Completo

### Exemplo 1: Dataset CA-1M com exportação

```bash
# 1. Processar dataset e salvar predições
python tools/demo.py data/val.txt \
    --video-ids 48458427 \
    --model-path models/cutr_rgb.pth \
    --save-preds-dir outputs/preds \
    --device cuda

# 2. Visualizar frame específico em 2D
python tools/visualize_preds.py \
    --image "ca1m-val-48458427/48458427/116505351959250.wide/image.png" \
    --pred-json "outputs/preds/48458427_116505p351959.json" \
    --no-labels

# 3. Visualizar em 3D com Rerun
python tools/rerun_visualize_saved_preds.py \
    --image "ca1m-val-48458427/48458427/116505351959250.wide/image.png" \
    --pred-json "outputs/preds/48458427_116505p351959.json"
```

### Exemplo 2: Imagem própria

```bash
# 1. Rodar inferência
python tools/infer_image.py \
    --image "foto_do_quarto.jpg" \
    --model-path models/cutr_rgb.pth \
    --device cuda \
    --max-edge 1024

# 2. Gerou: foto_do_quarto_inf.json

# 3. Visualizar 2D
python tools/visualize_preds.py \
    --image "foto_do_quarto.jpg" \
    --pred-json "foto_do_quarto_inf.json" \
    --score-thresh 0.3

# 4. Visualizar 3D
python tools/rerun_visualize_saved_preds.py \
    --image "foto_do_quarto.jpg" \
    --pred-json "foto_do_quarto_inf.json"
```

---

## Capturando Imagens com Intrínsecos

### Opção 1: NeRF Capture (iOS, recomendado)

Aplicativo oficial que fornece intrínsecos + pose + profundidade (se LiDAR disponível).

1. Instale [NeRF Capture](https://apps.apple.com/au/app/nerfcapture/id6446518379)
2. Use com streaming do `demo.py`:
```bash
python tools/demo.py stream --model-path models/cutr_rgbd.pth --device mps
```

### Opção 2: ARCore (Android)

Apps com ARCore podem exportar intrínsecos. Procure por:
- "Camera intrinsics"
- "Calibration"
- "fx, fy, cx, cy"

Valores típicos para referência:
- iPhone 12/13/14 (wide): fx ≈ fy ≈ 1200-1500 (varia com resolução)
- Câmeras Android: consulte documentação do fabricante

---

## Requisitos

Mesmos do repositório original:
- Python 3.10+
- PyTorch 2.x
- Ver `requirements.txt`

Instalação:
```bash
pip install torch torchvision
pip install -r requirements.txt
pip install -e .
```

---

## Licença

- Código original: Apple Sample Code License
- Modificações: mantidas sob mesma licença
- Modelos: Apple ML Research Model Terms of Use em [LICENSE_MODEL](LICENSE_MODEL)

---

## Créditos

Ferramentas adicionais desenvolvidas por [NishinoTSK](https://github.com/NishinoTSK).

Baseado no trabalho original:
- **Cubify Anything**: Justin Lazarow, David Griffiths, Gefen Kohavi, Francisco Crespo, Afshin Dehghan (Apple)
- Paper: [arXiv:2412.04458](https://arxiv.org/abs/2412.04458)
