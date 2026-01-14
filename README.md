# SoccerNet SynLoc Baseline

A standalone, Colab-friendly baseline for the [SoccerNet SynLoc Challenge](https://www.soccer-net.org/tasks/synloc) - synthetic athlete localization in soccer broadcasts.

## Challenge Overview

**Task:** Detect athletes in synthetic soccer images and project their positions to a Bird's Eye View (BEV) of the pitch.

**Metric:** mAP-LocSim (Mean Average Precision using Localization Similarity instead of IoU)

**Dataset:** ~70,000 synthetic images with 2D keypoint annotations (pelvis, pelvis_ground)

## Quick Start

> **For Colab Enterprise users:** See the [Colab Enterprise Quickstart Guide](docs/COLAB_ENTERPRISE_QUICKSTART.md) for a streamlined, single-notebook workflow.

### Standard Colab / Colab Enterprise Setup

#### Step 1: Setup Environment

Open a new Colab notebook and run:

```python
# Check GPU
!nvidia-smi

# Clone repository
!git clone https://github.com/geledek/soccernet-synloc.git
%cd soccernet-synloc

# Install package
!pip install -e .[dev] -q
```

### Step 2: Download Data from GCS

**For Colab Enterprise (recommended):** Data is stored in Google Cloud Storage.

> **First time?** You need to upload data to your GCS bucket first. See the [Colab Enterprise Quickstart Guide - Step 0](docs/COLAB_ENTERPRISE_QUICKSTART.md#step-0-upload-data-to-gcs-bucket-one-time) for instructions.

```python
# GCS Configuration - REPLACE WITH YOUR VALUES
GCS_BUCKET_NAME = 'your-bucket-name'      # Your GCS bucket
GCS_FOLDER_PATH = 'soccernet/synloc'      # Path within bucket
GCS_DATA_PATH = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/SpiideoSynLoc'

# Verify data exists
!gsutil ls {GCS_DATA_PATH}/

# Copy data to local SSD (faster training)
!mkdir -p /content/synloc

# Copy annotations
!gsutil -m cp -r {GCS_DATA_PATH}/annotations /content/synloc/

# Copy images (this takes a few minutes)
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/train /content/synloc/
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/val /content/synloc/
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/test /content/synloc/
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/challenge /content/synloc/
```

### Step 3: Organize Data Structure

```python
from pathlib import Path

DATA_ROOT = Path('/content/synloc')

# Create organized structure expected by the package
for split, ann_name, img_folder in [
    ('train', 'train', 'train'),
    ('valid', 'val', 'val'),
    ('test', 'test', 'test'),
    ('challenge', 'challenge_public', 'challenge')
]:
    split_dir = DATA_ROOT / split
    split_dir.mkdir(exist_ok=True)

    # Copy/link annotation
    ann_src = DATA_ROOT / 'annotations' / f'{ann_name}.json'
    ann_dst = split_dir / 'annotations.json'
    if ann_src.exists() and not ann_dst.exists():
        !cp {ann_src} {ann_dst}

    # Create images symlink
    img_src = DATA_ROOT / img_folder
    img_dst = split_dir / 'images'
    if img_src.exists() and not img_dst.exists():
        !ln -s {img_src} {img_dst}

print("Data organization complete!")

# Verify structure
!ls -la /content/synloc/train/
```

### Step 4: Set Paths

```python
# Data path (local SSD - fast)
DATA_ROOT = '/content/synloc'

# Checkpoint path (GCS - persistent)
CHECKPOINT_BUCKET = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/checkpoints'
CHECKPOINT_DIR = '/content/checkpoints'
!mkdir -p {CHECKPOINT_DIR}
```

<details>
<summary><b>Alternative: Google Drive (Standard Colab)</b></summary>

```python
from google.colab import drive
drive.mount('/content/drive')

# Download with SoccerNet
!pip install SoccerNet -q
from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory='/content/drive/MyDrive/SoccerNet/synloc')
downloader.downloadDataTask(task="synloc", split=["train", "valid", "test", "challenge"])

DATA_ROOT = '/content/drive/MyDrive/SoccerNet/synloc'
CHECKPOINT_DIR = '/content/drive/MyDrive/SoccerNet/checkpoints'
```

</details>

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

# Save final checkpoint to GCS (persistent)
!gsutil cp {CHECKPOINT_DIR}/best_model.pth {CHECKPOINT_BUCKET}/
!gsutil cp {CHECKPOINT_DIR}/final_model.pth {CHECKPOINT_BUCKET}/
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
    ann_file=f'{DATA_ROOT}/challenge/annotations.json',
    img_dir=f'{DATA_ROOT}/challenge/images',
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
SUBMISSION_DIR = '/content/submissions'
!mkdir -p {SUBMISSION_DIR}

results_path, metadata_path = format_results_for_submission(
    results=filtered_results,
    score_threshold=metrics['score_threshold'],
    position_from_keypoint_index=1,
    output_dir=SUBMISSION_DIR
)

zip_path = create_submission_zip(results_path, metadata_path, f'{SUBMISSION_DIR}/submission.zip')
print(f"Submission ready: {zip_path}")

# Upload to GCS for persistence
SUBMISSION_BUCKET = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/submissions'
!gsutil cp {zip_path} {SUBMISSION_BUCKET}/
print(f"Uploaded to: {SUBMISSION_BUCKET}/submission.zip")
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
# Save checkpoints frequently
trainer = SynLocTrainer(..., save_interval=5)

# Upload checkpoints to GCS periodically (in training loop or callback)
!gsutil cp {CHECKPOINT_DIR}/*.pth {CHECKPOINT_BUCKET}/

# Resume from GCS checkpoint
!gsutil cp {CHECKPOINT_BUCKET}/epoch_50.pth {CHECKPOINT_DIR}/
checkpoint = torch.load(f'{CHECKPOINT_DIR}/epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
trainer.start_epoch = checkpoint['epoch']
```

### GCS Data Transfer Slow

```python
# Use parallel transfers (-m flag)
!gsutil -m cp -r gs://bucket/path /content/

# For very large datasets, copy only what you need
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/train /content/synloc/  # Train only first
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
