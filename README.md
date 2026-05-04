# Cubify Anything - Ferramentas de Inferência e Visualização

Este repositório é um fork do [apple/ml-cubifyanything](https://github.com/apple/ml-cubifyanything) com ferramentas adicionais para exportação de predições, inferência em imagens arbitrárias e visualização offline.

## Instalar o repositorio a partir do wsl

git clone [https://github.com/NishinoTSK/ml-cubifyanything](https://github.com/NishinoTSK/ml-cubifyanything)

cd ml_cubifyanythin

virtualenv ambiente

source ambiente/bin/activate

Criar pasta models na pasta ml-cubifyanything e colocar os pesos dentro dela cutr_rgb.pth baixados daqui (RGB [https://github.com/apple/ml-cubifyanything?tab=readme-ov-file](https://github.com/apple/ml-cubifyanything?tab=readme-ov-file).)

pip install torch torchvision 

pip install scipy

pip install timm

pip install webdataset==0.2.86

pip install Pillow

pip install tifffile

pip install cyclonedds-nightly

pip install rerun-sdk==0.19.1

pip install -e . --no-build-isolation

pip install transformers accelerate

pip install ultralytics   # necessário apenas para YOLO-World

Rodar inferencia na sua imagem

python tools/infer_image.py \
    --image teste/passthrough_20260503_140304.png \
    --meta-json teste/passthrough_20260503_140304.json \
    --model-path models/cutr_rgb.pth --device cuda \
    --max-edge 0 --score-thresh 0.35 --label --label-backend all

python tools/visualize_preds.py --image teste/passthrough_20260503_140304.png --pred-json teste/passthrough_20260503_140304_inf.json --category-from label

python tools/rerun_visualize_saved_preds.py --image "teste/passthrough_20260503_140304.png" --pred-json "teste/passthrough_20260503_140304_inf.json" --no-labels

---

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


| Parâmetro                      | Descrição                                                                    | Padrão   |
| ------------------------------ | ---------------------------------------------------------------------------- | -------- |
| `--image`                      | Caminho para imagem RGB (obrigatório)                                        | -        |
| `--model-path`                 | Caminho para o checkpoint .pth (obrigatório)                                 | -        |
| `--device`                     | Dispositivo: cpu, cuda ou mps                                                | cpu      |
| `--score-thresh`               | Limiar de confiança para detecções                                           | 0.25     |
| `--max-edge`                   | Redimensiona imagem se o lado maior exceder este valor. Use 0 para desativar | 1024     |
| `--out-json`                   | Caminho de saída do JSON. Padrão: `<imagem>_inf.json`                        | -        |
| `--fx`, `--fy`, `--cx`, `--cy` | Parâmetros intrínsecos da câmera (opcional)                                  | estimado |
| `--meta-json`                  | Caminho para passthrough JSON com `fx/fy/cx/cy` (Quest/Unity)                | -        |
| `--depth`                      | Imagem de profundidade UInt16 em mm (apenas para modelo RGB-D)               | -        |


**Sobre os intrínsecos:**

- Se não informados, são estimados com base nas dimensões da imagem
- O modelo pode funcionar sem intrínsecos precisos, mas resultados 3D serão aproximados
- Se você redimensionar a imagem (`--max-edge`) e informar intrínsecos, eles serão ajustados automaticamente
- Com `--meta-json passthrough_xxx.json`, os campos `fx/fy/cx/cy` são lidos automaticamente do JSON do Quest/Unity. Flags explícitas (`--fx` etc.) ainda têm precedência. JSONs com vírgula decimal (locale pt-BR) são corrigidos em memória durante a leitura.

```bash
python tools/infer_image.py \
    --image teste/passthrough_20260503_140608.png \
    --meta-json teste/passthrough_20260503_140608.json \
    --model-path models/cutr_rgb.pth \
    --device cuda --max-edge 0 --score-thresh 0.25 --label
```

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


| Parâmetro        | Descrição                                    | Padrão |
| ---------------- | -------------------------------------------- | ------ |
| `--image`        | Imagem original                              | -      |
| `--pred-json`    | JSON de predições                            | -      |
| `--out`          | Caminho de saída. Padrão: `<imagem>_inf.png` | -      |
| `--score-thresh` | Filtra detecções abaixo deste score          | 0.0    |
| `--no-labels`    | Não mostrar labels nas caixas                | -      |
| `--line-width`   | Espessura das linhas das caixas              | 3      |


---

### 4. Visualização 3D Offline com Rerun (`tools/rerun_visualize_saved_preds.py`)

Visualiza predições no Rerun com suporte a 2D e 3D.

```bash
python tools/rerun_visualize_saved_preds.py \
    --image "minha_foto.png" \
    --pred-json "minha_foto_inf.json" \
    --application-id "minha_cena"
```

Use `--no-labels` para esconder texto nas caixas 2D/3D (só geometria). O padrão é `--labels` (rótulos visíveis quando existirem no JSON).

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

## Etiquetagem de Objetos (BLIP + Grounding-DINO / OWL-ViT v2 / YOLO-World)

O CuTR só devolve `class_id` numérico. Para responder "onde está o sofá marrom?" você precisa de **rótulos semânticos**. O etiquetador opcional suporta **quatro backends**, todos no mesmo módulo (`tools/labeler.py`):

| Backend | Tipo | Saída | Quando usar |
|---|---|---|---|
| **BLIP** (`Salesforce/blip-image-captioning-base`) | Caption por crop | `label`: `"a wooden dining chair"` | Respostas ricas em linguagem natural |
| **Grounding-DINO** (`IDEA-Research/grounding-dino-tiny`) | Detecção open-vocab na imagem inteira + IoU match | `category` + `category_score` | Categorias padronizadas com confiança |
| **OWL-ViT v2** (`google/owlv2-base-patch16-ensemble`) | Detecção open-vocab na imagem inteira + IoU match | `category` + `category_score` | Alternativa ao DINO; geralmente mais preciso em bboxes pequenas |
| **YOLO-World** (`yolov8l-worldv2.pt`) | Detecção open-vocab na imagem inteira + IoU match | `category` + `category_score` | Muito mais rápido que DINO/OWL (~52 FPS no V100); ideal para batch/lote |

Todos são plugados nos mesmos dois pontos: `infer_image.py --label` e `label_room.py`.

Você pode rodar **vários backends de categoria ao mesmo tempo** (ex.: `both_yolo` roda BLIP + DINO + YOLO). Cada detector escreve suas próprias chaves no JSON (`category_dino`, `category_yolo`, etc.) e a chave legada `category` espelha o primeiro que retornar um match. Nos visualizadores, use `--category-from` para escolher qual campo exibir.

### Dependências

```bash
pip install transformers accelerate
pip install ultralytics   # necessário apenas para YOLO-World
```

Primeira execução baixa modelos para `~/.cache/huggingface` (e `~/.cache/ultralytics` para o YOLO):
- BLIP base + DINO tiny: ~1-2 GB
- OWL-ViT v2: ~1.5 GB extras
- YOLO-World large: ~170 MB extras (CNN, mais leve e rápido)

### Caminho 1 — Inferência única com etiquetagem

Adicione `--label` ao comando de sempre:

```bash
WGPU_BACKEND=vulkan python tools/infer_image.py \
    --image teste/19.jpeg \
    --model-path models/cutr_rgb.pth \
    --device cuda \
    --fx 1120 --fy 1120 --cx 800 --cy 600 \
    --max-edge 0 --score-thresh 0.25 \
    --label
```

Flags disponíveis:

| Flag              | Descrição                                          | Default                                 |
| ----------------- | -------------------------------------------------- | --------------------------------------- |
| `--label`         | Liga a etiquetagem (sem isso, comportamento atual) | off                                     |
| `--label-backend` | `blip` \| `dino` \| `owlv2` \| `yolo` \| `both` \| `both_owl` \| `both_yolo` \| `all` \| `none` | `both` |
| `--vocab`         | Arquivo com classes (uma por linha) para DINO/OWL/YOLO | `tools/labeling_vocab_default.txt`     |
| `--blip-model`    | HF model id para BLIP                              | `Salesforce/blip-image-captioning-base` |
| `--dino-model`    | HF model id para Grounding-DINO                    | `IDEA-Research/grounding-dino-tiny`     |
| `--owlv2-model`   | HF model id para OWL-ViT v2                        | `google/owlv2-base-patch16-ensemble`    |
| `--yolo-model`    | Ultralytics checkpoint para YOLO-World             | `yolov8l-worldv2.pt`                    |
| `--iou-min`       | IoU mínima do match detector contra a bbox do CuTR | `0.3`                                   |

**Backends disponíveis:**
- `blip` — só BLIP (caption livre)
- `dino` — só Grounding-DINO (categoria padronizada + score)
- `owlv2` — só OWL-ViT v2 (categoria padronizada + score; geralmente mais preciso que DINO em bboxes pequenas)
- `yolo` — só YOLO-World (categoria padronizada + score; **mais rápido**, útil para batch/lote)
- `both` — BLIP + DINO (padrão)
- `both_owl` — BLIP + OWL-ViT v2
- `both_yolo` — BLIP + DINO + YOLO-World (mais rico, todos os campos prefixados no JSON)
- `all` — BLIP + DINO + OWL-ViT v2 + YOLO-World (todos os backends de uma vez)
- `none` — desabilita etiquetagem


O `*_inf.json` ganha os campos por detecção. Quando um único detector é usado (`dino`, `owlv2` ou `yolo`), a estrutura é simples:

```json
{
  "source_image": "teste/19.jpeg",
  "image_size_hw": [768, 1024],
  "detections": [
    {
      "bbox_xyxy": [120.5, 230.0, 340.7, 510.9],
      "score": 0.85,
      "class_id": 1,
      "label": "a wooden dining chair",
      "category": "chair",
      "category_score": 0.78
    }
  ],
  "boxes_3d": { ... }
}
```

Quando **múltiplos backends** rodam (ex.: `both_yolo`), cada detector escreve suas próprias chaves para que você possa comparar posteriormente:

```json
{
  "detections": [
    {
      "label": "a wooden dining chair",
      "category": "chair",
      "category_score": 0.78,
      "category_dino": "chair",
      "category_score_dino": 0.78,
      "category_owlv2": "armchair",
      "category_score_owlv2": 0.61,
      "category_yolo": "chair",
      "category_score_yolo": 0.91
    }
  ]
}
```

- `category`/`category_score` espelham o primeiro backend (na ordem: DINO → OWL → YOLO) que devolveu match.
- `category_*prefixed*`/`category_score_*prefixed*` ficam `null` quando o detector respectivo não encontrou correspondência IoU.

`label` é sempre `null` quando o BLIP não devolve caption; `category`/`category_score` ficam `null` quando nenhum detector atinge `iou-min`.

Nos visualizadores (`visualize_preds.py`, `rerun_visualize_saved_preds.py`, `rerun_visualize_merged_world.py`) o texto da caixa usa **só** `category` por padrão. Sem categoria, mostra `cls=N` + score; o `label` permanece no JSON (ex.: busca semântica no app) mas **não** é usado como fallback na tela.

Use `--category-from` para escolher qual campo exibir quando múltiplos backends foram rodados:

```bash
python tools/visualize_preds.py --image teste/19.jpeg --pred-json teste/19_inf.json --category-from category_yolo
python tools/rerun_visualize_saved_preds.py --image teste/19.jpeg --pred-json teste/19_inf.json --category-from category_dino
python tools/rerun_visualize_merged_world.py --room-json teste/room.json --category-from category_owlv2
```

Valores aceitos: `category` (padrão), `category_dino`, `category_owlv2`, `category_yolo`, `label` (exibe o caption do BLIP ao invés da classe).

```bash
python tools/visualize_preds.py --image teste/19.jpeg --pred-json teste/19_inf.json
python tools/rerun_visualize_saved_preds.py --image teste/19.jpeg --pred-json teste/19_inf.json
```

### Caminho 2 — Etiquetar um `room.json` existente

Use `label_room.py` depois que o `scan_pipeline.py` gerou seu mapa do quarto:

```bash
python tools/label_room.py \
    --room teste/recontruct2/room.json \
    --captures teste/recontruct2 \
    --label-backend both \
    --device cuda
```

Para cada `obj` em `objects[*]`:

1. Abre `<captures>/<evidence.best_frame>`.
2. Cria um *crop* a partir de `evidence.best_bbox` (gravado pelo `scan_pipeline.py`).
3. Chama `Labeler.label_detections` — o DINO roda **uma vez por imagem distinta** e os objetos do mesmo frame são todos atribuídos via IoU contra essa única passada.
4. Escreve `label`, `category`, `category_score` no objeto, mais os campos prefixados quando múltiplos backends são usados (`category_dino`, `category_yolo`, etc.).

Resultado em `objects[*]`:

```json
{
  "id": "obj_0003",
  "label": "a brown sofa",
  "category": "sofa",
  "category_score": 0.71,
  "category_dino": "sofa",
  "category_score_dino": 0.71,
  "category_yolo": "couch",
  "category_score_yolo": 0.88,
  "score": 0.83,
  "center_xyz": [...],
  "dims_lhw": [...],
  "R_3x3": [...],
  "evidence": {
    "n_frames": 4,
    "best_frame": "passthrough_20260503_140312.png",
    "best_bbox": [410.1, 220.7, 760.5, 612.3],
    "frames": [...]
  }
}
```

### Tempos esperados (GPU)

| Backend | Cenário | Tempo aproximado |
|---|---|---|
| DINO | Inferência única (~5-15 detecções) | 5-10 s |
| OWL-ViT v2 | Inferência única (~5-15 detecções) | 6-12 s |
| YOLO-World | Inferência única (~5-15 detecções) | 1-3 s |
| BLIP + DINO | `room.json` com ~25 objetos / ~10 imagens | 30-60 s |
| BLIP + OWL-ViT v2 | `room.json` com ~25 objetos / ~10 imagens | 35-70 s |
| BLIP + DINO + YOLO | `room.json` com ~25 objetos / ~10 imagens | 40-90 s |
| Qualquer | Primeira execução (download de modelos) | +30-90 s extras |

CPU funciona, mas é ~5-10× mais lento. Use `--label-backend yolo` quando velocidade for prioridade (é ~5× mais rápido que DINO), `--label-backend owlv2` quando precisão de categoria em bboxes pequenas for mais importante, ou `--label-backend both_yolo` para ter todos os campos prefixados no JSON e escolher o melhor via `--category-from`.

### Compatibilidade

- Sem `--label`, `infer_image.py` produz exatamente o mesmo JSON que produzia antes.
- Os campos `label`/`category`/`category_score` são opcionais em ambos os schemas; na tela, sem `category`, os visualizadores caem para `cls=N {score}` (não usam `label` como fallback).

---

## Pipeline de Reconstrução Persistente do Quarto (Quest 3 + Unity)

Cenário-alvo: app Unity rodando no Meta Quest 3 que precisa responder "onde está X?" sem chamar o modelo a cada pergunta. A solução é **escanear o quarto uma vez**, gerar um `room.json` com todas as caixas 3D já no frame de mundo do Quest, e em runtime fazer só lookup local.

### Visão geral

```
[ETAPA 1: SCAN UNICO no Quest 3]
Unity (saver) -> N capturas: PNG + JSON{fx,fy,cx,cy,pose_R_wc,pose_t_wc,anchor_uuid}
                                                    |
                                                    v
[ETAPA 2: PIPELINE no PC]
tools/scan_pipeline.py -> CuTR por frame -> transforma cam->mundo Quest -> dedup -> room.json

[ETAPA 3: RUNTIME]
Unity carrega room.json (1x), resolve anchor pelo uuid, faz busca local. Zero inferencia.
```

### O que o saver Unity precisa salvar (extensão do que ele já salva)

Hoje o saver provavelmente escreve algo assim:

```json
{ "fx": 869.55, "fy": 869.55, "cx": 640.27, "cy": 642.13,
  "width": 1280, "height": 1280, "timestamp": "..." }
```

Adicionar dois campos novos (e idealmente um terceiro):

```json
{
  "fx": 869.55, "fy": 869.55, "cx": 640.27, "cy": 642.13,
  "width": 1280, "height": 1280,
  "timestamp": "2026-04-22T14:53:23Z",

  "pose_R_wc": [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]],
  "pose_t_wc": [tx, ty, tz],
  "anchor_uuid": "<OVRSpatialAnchor uuid (opcional)>"
}
```

`pose_R_wc` e `pose_t_wc` representam a transformada **camera → mundo Quest** (em metros), exatamente o `Transform` da câmera Passthrough no momento do snapshot. Snippet C# pra gerar a matriz a partir do quaternion do `Transform.rotation`:

```csharp
static float[][] Quat3x3(Quaternion q)
{
    var m = Matrix4x4.Rotate(q);
    return new[]
    {
        new[] { m.m00, m.m01, m.m02 },
        new[] { m.m10, m.m11, m.m12 },
        new[] { m.m20, m.m21, m.m22 },
    };
}
```

### Estrutura de pastas esperada pelo pipeline

```
rooms/
  quarto_bianca/
    captures/
      cap_001.png
      cap_001.json        # contem pose + intrinsics
      cap_001_depth.png   # opcional, UInt16 mm (so para cutr_rgbd.pth)
      cap_002.png
      cap_002.json
      ...
    room.json             # gerado pelo pipeline
```

### Rodar a reconstrução

```bash
WGPU_BACKEND=vulkan python tools/scan_pipeline.py \
    --captures rooms/quarto_bianca/captures \
    --out      rooms/quarto_bianca/room.json \
    --model-path models/cutr_rgb.pth \
    --device cuda \
    --score-thresh 0.25 \
    --convention unity
```

Saída no terminal:

```
Loading CuTR model: models/cutr_rgb.pth
  OK   cap_001: 4 detections
  OK   cap_002: 6 detections
  ...
Aggregating 87 detections from 18 captures...
After dedup: 23 unique objects
Saved room map: rooms/quarto_bianca/room.json
```

**Parâmetros chave:**


| Flag                   | Descrição                                              | Default     |
| ---------------------- | ------------------------------------------------------ | ----------- |
| `--captures`           | Pasta com `<stem>.png` + `<stem>.json`                 | obrigatório |
| `--out`                | Caminho de saída do `room.json`                        | obrigatório |
| `--model-path`         | `cutr_rgb.pth` ou `cutr_rgbd.pth`                      | obrigatório |
| `--convention`         | `unity` (default, aplica `M=diag(1,-1,1)`) ou `opencv` | `unity`     |
| `--score-thresh`       | Filtro de detecções                                    | `0.25`      |
| `--max-edge`           | Resize máximo do lado maior (0 = off)                  | `1024`      |
| `--iou-thresh`         | 3D IoU pra fundir caixas (menor = funde mais)          | `0.2`       |
| `--containment-thresh` | Funde se um box está N% dentro do outro                | `0.5`       |
| `--center-fuse-ratio`  | Funde se centros estão a `ratio*min(diag)`             | `0.5`       |
| `--min-evidence`       | Descarta clusters de menos de N frames                 | `1`         |
| `--min-volume`         | Descarta caixas com volume menor que X m³              | `0.0`       |


**Receita recomendada para cenas com ruído** (ex: scan de quarto com ~10 fotos):

```bash
WGPU_BACKEND=vulkan python tools/scan_pipeline.py \
    --captures rooms/quarto_bianca/captures \
    --out      rooms/quarto_bianca/room.json \
    --model-path models/cutr_rgb.pth \
    --device cuda --convention unity \
    --score-thresh 0.5 \
    --min-evidence 2 \
    --min-volume 0.005
```

`--min-evidence 2` (precisa aparecer em ≥2 frames) e `--min-volume 0.005` (≥5 L, ~17 cm cúbico) eliminam ruído de detecção. Com 10 fotos do quarto, isso normalmente reduz de ~80 falsos positivos para ~25 objetos reais.

### Schema do `room.json`

```json
{
  "room_id": "quarto_bianca",
  "world_frame": {
    "anchor_uuid": "<OVRSpatialAnchor uuid>",
    "units": "meters",
    "convention": "unity_left_handed_y_up"
  },
  "created_at": "2026-04-28T18:00:00Z",
  "n_captures_used": 18,
  "n_captures_skipped": 0,
  "objects": [
    {
      "id": "obj_0001",
      "label": null,
      "category": null,
      "category_score": null,
      "score": 0.83,
      "center_xyz": [1.2, 0.5, -2.1],
      "dims_lhw": [0.6, 1.1, 0.4],
      "R_3x3": [[...]],
      "evidence": {
        "n_frames": 4,
        "best_frame": "cap_007.png",
        "best_bbox": [410.1, 220.7, 760.5, 612.3],
        "frames": [...]
      }
    }
  ]
}
```

`label`/`category`/`category_score` ficam `null` propositalmente; preencha depois com `tools/label_room.py` (BLIP + Grounding-DINO) ou um pipeline manual sem mudar o resto. O `evidence.best_bbox` é a bbox 2D da detecção de maior score que originou o cluster — é o que `label_room.py` usa para pegar o crop correto na imagem.

### Visualizar o `room.json`

```bash
python tools/rerun_visualize_merged_world.py --room-json rooms/quarto_bianca/room.json
```

O visualizador detecta o schema automaticamente (também lê o legacy `merged_boxes_3d` dos JSONs antigos do `merge_preds_world.py`).

### Alinhamento entre sessões (OVRSpatialAnchor)

A pose do Quest é estável dentro de uma sessão, mas ao reabrir o app a origem pode escorregar. Pra que o `room.json` continue alinhado:

1. **Antes do scan**: criar um `OVRSpatialAnchor` na posição inicial e salvar com `SaveAnchorAsync()`. Guardar o `uuid`.
2. **Em cada captura**: incluir o mesmo `anchor_uuid` no JSON.
3. **Em runtime**: carregar o anchor pelo `uuid` (`OVRSpatialAnchor.LoadUnboundAnchorsAsync`). O sistema realinha automaticamente o frame do mundo, e as `center_xyz` do `room.json` ficam corretas.

Se isso for trabalhoso pra MVP, ignorar — basta fazer scan e usar o `room.json` na mesma sessão.

### Por que isso funciona (e os scripts antigos não)

O `merge_preds_world.py` e o `video_reconstruct_rgb.py` precisavam **estimar** a pose entre fotos com heurísticas frágeis (dHash, ORB, COLMAP monocular) — todas sem escala métrica confiável. O Quest 3 já entrega pose `(R_wc, t_wc)` em metros via Insight Tracking, então o "merge" vira um simples produto de matriz: `world = R_wc · M · cam + t_wc`. Sem SfM, sem RANSAC, sem chute de escala.

Esses dois scripts ficam como **legacy** — funcionam só em casos específicos (panorama 360 do mesmo ponto). Para reconstrução real do quarto, usar `scan_pipeline.py`.

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

