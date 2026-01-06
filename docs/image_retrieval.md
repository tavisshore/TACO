# Image Retrieval for CVGL Localization

This document describes the PyTorch Lightning-based image retrieval model used for CVGL (Computer Vision Global Localization) in the TACO system.

## Overview

The image retrieval model uses a pre-trained ConvNeXt-Tiny backbone to encode images into low-dimensional descriptors (default 512-D). These descriptors enable efficient visual place recognition for global localization.

## Model Architecture

```
Input Image (3 x H x W)
    ↓
ConvNeXt-Tiny Features (convolutional backbone)
    ↓
Adaptive Average Pooling
    ↓
Flatten + LayerNorm → (768-D)
    ↓
Projection Head (Linear → BN → ReLU → Dropout → Linear)
    ↓
L2-Normalized Embedding (512-D)
```

### Key Components

1. **Backbone**: ConvNeXt-Tiny (pretrained)
   - Modern CNN architecture
   - Better than ResNet for visual features
   - ~28M parameters

2. **Projection Head**:
   - Linear(768 → 1024) → BatchNorm → ReLU → Dropout(0.1) → Linear(1024 → 512)
   - Maps backbone features to embedding space

3. **Loss Functions**:
   - **Triplet Loss**: Ensures anchor is closer to positive than negative
   - **Contrastive Loss**: InfoNCE loss for better separation

## Training

### Data Format

The model expects triplet data in the following format:

**Triplets file** (`triplets.txt`):
```
anchor.jpg,positive1.jpg,negative1.jpg,0
anchor.jpg,positive2.jpg,negative2.jpg,0
anchor2.jpg,positive3.jpg,negative2.jpg,1
...
```

Each line contains:
- Anchor image path
- Positive image path (same place as anchor)
- Negative image path (different place)
- Place label (integer)

### Training Command

```bash
python -m taco.localization.cvgl.train \
    --train-data-dir /path/to/train/images \
    --train-triplets /path/to/train/triplets.txt \
    --val-data-dir /path/to/val/images \
    --val-triplets /path/to/val/triplets.txt \
    --output-dir outputs/retrieval \
    --embedding-dim 512 \
    --batch-size 32 \
    --max-epochs 100 \
    --learning-rate 1e-4 \
    --gpus 1
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding-dim` | 512 | Output embedding dimension |
| `--batch-size` | 32 | Training batch size |
| `--max-epochs` | 100 | Maximum training epochs |
| `--learning-rate` | 1e-4 | Learning rate |
| `--temperature` | 0.07 | Temperature for contrastive loss |
| `--margin` | 0.2 | Margin for triplet loss |
| `--freeze-backbone` | False | Freeze ConvNeXt weights |
| `--no-pretrained` | False | Don't use pretrained weights |
| `--gpus` | None | Number of GPUs (None=CPU) |

### Fine-tuning Strategy

For best results:

1. **Stage 1**: Train with frozen backbone
   ```bash
   --freeze-backbone --learning-rate 1e-3
   ```

2. **Stage 2**: Fine-tune entire network
   ```bash
   --learning-rate 1e-4 --resume /path/to/checkpoint
   ```

## Inference

### Load Trained Model

```python
from taco.localization.cvgl import ImageRetrievalModel
import torch

# Load from checkpoint
model = ImageRetrievalModel.load_from_checkpoint(
    "outputs/retrieval/checkpoints/best.ckpt"
)
model.eval()
model = model.to("cuda")  # or "cpu"
```

### Encode Single Image

```python
import numpy as np
from PIL import Image

# Load image
image = np.array(Image.open("query.jpg"))  # (H, W, 3) RGB

# Encode to embedding
embedding = model.encode_image(image)  # (512,)
```

### Build Image Database

```python
import numpy as np
from pathlib import Path

# Encode all database images
database_embeddings = []
database_paths = []

for img_path in Path("database").glob("*.jpg"):
    image = np.array(Image.open(img_path))
    embedding = model.encode_image(image)
    database_embeddings.append(embedding)
    database_paths.append(str(img_path))

# Stack into array
database_embeddings = np.stack(database_embeddings)  # (N, 512)
```

### Retrieve Similar Images

```python
# Query for similar images
query_image = np.array(Image.open("query.jpg"))
query_embedding = model.encode_image(query_image)

# Get top-10 matches
indices, similarities = model.retrieve_similar(
    query_embedding,
    database_embeddings,
    top_k=10
)

print("Top matches:")
for idx, sim in zip(indices, similarities):
    print(f"  {database_paths[idx]}: {sim:.3f}")
```

## Integration with CVGL Localizer

The trained model can be integrated into the `CVGLLocalizer`:

```python
from taco.localization.cvgl import CVGLLocalizer, ImageRetrievalModel
import numpy as np

# Load model
model = ImageRetrievalModel.load_from_checkpoint("best.ckpt")

# Create localizer with model
localizer = CVGLLocalizer()
localizer.model = model
localizer.load_map("map_database.npz")

# Localize image
image = np.array(...)  # (H, W, 3)
measurement = localizer.localize(image, timestamp=0.0)
```

## Performance Tips

### Training Speed
- Use mixed precision: `--precision 16-mixed` (automatic with `--gpus`)
- Increase batch size if GPU memory allows
- Use more workers: `--num-workers 8`
- Use persistent workers for faster data loading

### Inference Speed
- Use batch inference when possible
- Compile model with `torch.compile()` (PyTorch 2.0+)
- Use TensorRT or ONNX for deployment
- Use half precision: `model.half()`

### Memory Optimization
- Reduce embedding dimension: `--embedding-dim 256`
- Use gradient accumulation for large batch sizes
- Use gradient checkpointing for deeper models

## Evaluation Metrics

The model is evaluated using:

1. **Triplet Accuracy**: % of triplets where `d(anchor, positive) < d(anchor, negative)`
2. **Recall@K**: % of queries where correct place is in top-K retrievals
3. **Mean Average Precision (mAP)**: Average precision across all queries

## Data Preparation

### Creating Triplets

Use the following script to generate triplets from GPS-tagged images:

```python
from pathlib import Path
import numpy as np

def create_triplets(image_dir, gps_file, output_file, pos_threshold=10.0, neg_threshold=50.0):
    """Create triplets from GPS-tagged images.

    Args:
        image_dir: Directory with images
        gps_file: CSV with format: image_path,lat,lon
        output_file: Output triplets file
        pos_threshold: Distance threshold for positives (meters)
        neg_threshold: Distance threshold for negatives (meters)
    """
    # Load GPS data
    data = []
    with open(gps_file) as f:
        for line in f:
            path, lat, lon = line.strip().split(',')
            data.append((path, float(lat), float(lon)))

    # Compute pairwise distances (simplified)
    triplets = []
    for i, (anchor_path, anchor_lat, anchor_lon) in enumerate(data):
        for j, (pos_path, pos_lat, pos_lon) in enumerate(data):
            if i == j:
                continue

            # Check if positive (same place)
            dist = np.sqrt((anchor_lat - pos_lat)**2 + (anchor_lon - pos_lon)**2) * 111000
            if dist < pos_threshold:
                # Find negative
                for k, (neg_path, neg_lat, neg_lon) in enumerate(data):
                    neg_dist = np.sqrt((anchor_lat - neg_lat)**2 + (anchor_lon - neg_lon)**2) * 111000
                    if neg_dist > neg_threshold:
                        triplets.append(f"{anchor_path},{pos_path},{neg_path},0\n")
                        break

    # Save triplets
    with open(output_file, 'w') as f:
        f.writelines(triplets)

# Usage
create_triplets(
    image_dir="images",
    gps_file="gps_data.csv",
    output_file="triplets.txt"
)
```

## Advanced Features

### Hard Negative Mining

For improved training, implement online hard negative mining:

```python
# In training loop
with torch.no_grad():
    # Compute all embeddings
    all_embeddings = model(all_images)

    # For each anchor, find hardest negative
    # (closest negative that's still different)
    hard_negatives = find_hard_negatives(
        anchors, all_embeddings, labels
    )
```

### Multi-Scale Features

Extract features at multiple scales:

```python
# Modify forward pass
def forward_multiscale(self, images):
    features = []
    for scale in [0.5, 1.0, 1.5]:
        scaled = F.interpolate(images, scale_factor=scale)
        feat = self.backbone(scaled)
        features.append(feat)

    # Aggregate features
    combined = torch.cat(features, dim=1)
    embedding = self.projection_head(combined)
    return embedding
```

## Citation

If you use this model in your research, please cite:

```bibtex
@article{taco2024,
  title={TACO: Tight IMU-Visual Localization with Pose Graph Optimization},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- ConvNeXt: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- Triplet Loss: [FaceNet: A Unified Embedding](https://arxiv.org/abs/1503.03832)
- InfoNCE: [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
