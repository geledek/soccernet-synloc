# SoccerNet SynLoc Baseline

A standalone, Colab-friendly baseline for the [SoccerNet SynLoc Challenge](https://www.soccer-net.org/tasks/synloc) - synthetic athlete localization in soccer broadcasts.

## Challenge Overview

**Task:** Detect athletes in synthetic soccer images and project their positions to a Bird's Eye View (BEV) of the pitch.

**Metric:** mAP-LocSim (Mean Average Precision using Localization Similarity instead of IoU)

**Dataset:** ~70,000 synthetic images with 2D keypoint annotations (pelvis, pelvis_ground)

## Quick Start (Google Colab)

### Step 1: Setup Environment

Open a new Colab notebook and run:

```python
# Check GPU
!nvidia-smi

# Clone repository
!git clone https://github.com/YOUR_USERNAME/soccernet-synloc.git
%cd soccernet-synloc

# Install package
!pip install -e .[dev] -q

# Install SoccerNet downloader
!pip install SoccerNet -q
```

### Step 2: Mount Google Drive & Download Data

```python
from google.colab import drive
drive.mount('/content/drive')

# Create data directory on Drive (persists across sessions)
!mkdir -p /content/drive/MyDrive/SoccerNet/synloc

# Download dataset (~20GB, takes 10-20 minutes)
from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory='/content/drive/MyDrive/SoccerNet/synloc')
downloader.downloadDataTask(
    task="synloc",
    split=["train", "valid", "test", "challenge"]
)
```

> **Note:** You'll need SoccerNet credentials. Register at [soccer-net.org](https://www.soccer-net.org/)

### Step 3: Organize Data

```python
# The downloaded structure needs reorganization
# Run this once after download

import os
from pathlib import Path

DATA_ROOT = Path('/content/drive/MyDrive/SoccerNet/synloc')

# Extract annotations if needed
if (DATA_ROOT / 'annotations.zip').exists():
    !unzip -q {DATA_ROOT}/annotations.zip -d {DATA_ROOT}

# Create organized structure
for split, ann_name in [('train', 'train'), ('valid', 'val'), ('test', 'test'), ('challenge', 'challenge_public')]:
    split_dir = DATA_ROOT / split
    split_dir.mkdir(exist_ok=True)

    # Move/link annotations
    ann_src = DATA_ROOT / 'annotations' / f'{ann_name}.json'
    ann_dst = split_dir / 'annotations.json'
    if ann_src.exists() and not ann_dst.exists():
        !cp {ann_src} {ann_dst}

    # Link images (assuming fullhd structure)
    img_src = DATA_ROOT / 'fullhd' / (split if split != 'valid' else 'val')
    img_dst = split_dir / 'images'
    if img_src.exists() and not img_dst.exists():
        !ln -s {img_src} {img_dst}

print("Data organization complete!")
```

### Step 4: Copy Data to Local SSD (Faster Training)

```python
# Copy to Colab's local SSD for 3-5x faster I/O
!mkdir -p /content/synloc
!cp -r /content/drive/MyDrive/SoccerNet/synloc/train /content/synloc/
!cp -r /content/drive/MyDrive/SoccerNet/synloc/valid /content/synloc/

# Use local path for training
DATA_ROOT = '/content/synloc'

# Keep checkpoints on Drive (persistent)
CHECKPOINT_DIR = '/content/drive/MyDrive/SoccerNet/checkpoints'
!mkdir -p {CHECKPOINT_DIR}
```

### Step 5: Train Model

```python
import torch
from torch.utils.data import DataLoader
from synloc.models import YOLOXPose
from synloc.data import SynLocDataset, get_train_transforms, get_val_transforms
from synloc.training import SynLocTrainer

# Configuration
config = {
    'model_variant': 'tiny',  # tiny, s, m, l
    'input_size': (640, 640),
    'batch_size': 16,         # Reduce if OOM
    'epochs': 100,
    'lr': 1e-3,
}

# Create model
model = YOLOXPose(
    variant=config['model_variant'],
    num_keypoints=2,
    input_size=config['input_size']
)

# Create datasets
train_dataset = SynLocDataset(
    ann_file=f'{DATA_ROOT}/train/annotations.json',
    img_dir=f'{DATA_ROOT}/train/images',
    transforms=get_train_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

val_dataset = SynLocDataset(
    ann_file=f'{DATA_ROOT}/valid/annotations.json',
    img_dir=f'{DATA_ROOT}/valid/images',
    transforms=get_val_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True,
    num_workers=2, collate_fn=SynLocDataset.collate_fn, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=False,
    num_workers=2, collate_fn=SynLocDataset.collate_fn, pin_memory=True
)

# Train
trainer = SynLocTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    lr=config['lr'],
    epochs=config['epochs'],
    checkpoint_dir=CHECKPOINT_DIR,
    use_amp=True
)

history = trainer.train()
```

### Step 6: Evaluate

```python
from synloc.evaluation import run_inference, evaluate_predictions

# Load best checkpoint
checkpoint = torch.load(f'{CHECKPOINT_DIR}/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create test loader
test_dataset = SynLocDataset(
    ann_file=f'{DATA_ROOT}/test/annotations.json',
    img_dir=f'{DATA_ROOT}/test/images',
    transforms=get_val_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False,
    num_workers=2, collate_fn=SynLocDataset.collate_fn
)

# Run inference
results = run_inference(model, test_loader, device='cuda')

# Evaluate with mAP-LocSim
metrics = evaluate_predictions(
    gt_file=f'{DATA_ROOT}/test/annotations.json',
    results=results,
    position_from_keypoint_index=1  # pelvis_ground
)

print(f"\n{'='*50}")
print(f"mAP-LocSim: {metrics['mAP_locsim']:.4f}")
print(f"Precision:  {metrics['precision']:.4f}")
print(f"Recall:     {metrics['recall']:.4f}")
print(f"F1 Score:   {metrics['f1']:.4f}")
print(f"{'='*50}")
```

### Step 7: Generate Submission

```python
from synloc.evaluation import format_results_for_submission, create_submission_zip

# Run inference on challenge set
challenge_dataset = SynLocDataset(
    ann_file='/content/drive/MyDrive/SoccerNet/synloc/challenge/annotations.json',
    img_dir='/content/drive/MyDrive/SoccerNet/synloc/challenge/images',
    transforms=get_val_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

challenge_loader = DataLoader(
    challenge_dataset, batch_size=config['batch_size'], shuffle=False,
    num_workers=2, collate_fn=SynLocDataset.collate_fn
)

results = run_inference(model, challenge_loader, device='cuda')

# Filter by optimal threshold (from validation)
filtered_results = [r for r in results if r['score'] >= metrics['score_threshold']]

# Create submission
SUBMISSION_DIR = '/content/drive/MyDrive/SoccerNet/submissions'
!mkdir -p {SUBMISSION_DIR}

results_path, metadata_path = format_results_for_submission(
    results=filtered_results,
    score_threshold=metrics['score_threshold'],
    position_from_keypoint_index=1,
    output_dir=SUBMISSION_DIR
)

zip_path = create_submission_zip(results_path, metadata_path, f'{SUBMISSION_DIR}/submission.zip')
print(f"Submission ready: {zip_path}")
```

---

## Project Structure

```
soccernet-synloc/
├── notebooks/                    # Jupyter notebooks
│   ├── 01_setup_and_data.ipynb  # Environment & data exploration
│   ├── 02_training.ipynb        # Training with progress tracking
│   ├── 03_evaluation.ipynb      # mAP-LocSim & error analysis
│   └── 04_submission.ipynb      # Challenge submission
├── synloc/                       # Main package
│   ├── models/
│   │   ├── layers.py            # ConvBNAct, CSPLayer, Focus, etc.
│   │   ├── backbone.py          # CSPDarknet
│   │   ├── neck.py              # YOLOXPAFPN
│   │   ├── head.py              # YOLOXPoseHead
│   │   ├── head_simcc.py        # SimCC head (improvement)
│   │   └── yoloxpose.py         # Full model assembly
│   ├── data/
│   │   ├── camera.py            # Camera projection utilities
│   │   ├── dataset.py           # SynLocDataset (COCO format)
│   │   └── transforms.py        # Albumentations augmentations
│   ├── training/
│   │   ├── losses.py            # OKSLoss, IoULoss, BCE
│   │   └── trainer.py           # Training loop with AMP
│   ├── evaluation/
│   │   ├── locsim.py            # mAP-LocSim metric
│   │   └── inference.py         # Inference utilities
│   ├── visualization/
│   │   └── pitch.py             # Pitch drawing, BEV visualization
│   └── utils/
│       └── config.py            # YAML config loading
├── configs/
│   ├── base.yaml                # Base configuration
│   └── experiments/             # Ablation experiment configs
├── docs/
│   └── GOOGLE_DRIVE_GUIDE.md    # Drive management guide
├── requirements.txt
├── setup.py
└── README.md
```

---

## Model Variants

| Variant | Parameters | Recommended Batch Size | Recommended GPU |
|---------|------------|------------------------|-----------------|
| tiny    | 3.9M       | 16-32                  | T4 (16GB)       |
| s       | 8.9M       | 8-16                   | T4/V100         |
| m       | 25.3M      | 4-8                    | V100/A100       |
| l       | 54.2M      | 2-4                    | A100            |

---

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config['batch_size'] = 8  # or 4

# Use smaller model
config['model_variant'] = 'tiny'

# Reduce input size
config['input_size'] = (512, 512)
```

### Slow Data Loading

```python
# Copy data to local SSD (see Step 4)
# Increase workers (but not too many on Colab)
num_workers = 2  # 2-4 is usually optimal
```

### Session Timeout

```python
# Save checkpoints frequently to Drive
trainer = SynLocTrainer(..., save_interval=5)

# Resume from checkpoint
checkpoint = torch.load(f'{CHECKPOINT_DIR}/epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
trainer.start_epoch = checkpoint['epoch']
```

---

## Improvement Ideas

1. **Higher Resolution:** 960x960 or 1280x1280 input (requires more VRAM)
2. **Larger Model:** Use 's' or 'm' variant
3. **SimCC Head:** Use `synloc/models/head_simcc.py` for better keypoint localization
4. **Stronger Augmentation:** Enable mosaic in config
5. **Test-Time Augmentation:** Multi-scale inference with flip

---

## References

- [SoccerNet Challenge 2024](https://www.soccer-net.org/challenges/2024)
- [YOLOX-Pose Paper](https://arxiv.org/abs/2212.05637)
- [SimCC Paper](https://arxiv.org/abs/2107.03332)

## License

MIT License

## Citation

If you use this baseline, please cite:

```bibtex
@inproceedings{soccernet2024,
  title={SoccerNet 2024 Challenges},
  author={SoccerNet Team},
  year={2024}
}
```
