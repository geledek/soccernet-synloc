# Colab Enterprise Quickstart Guide

Complete guide for running SoccerNet SynLoc baseline on Google Colab Enterprise.

## Prerequisites

- Access to Google Colab Enterprise
- GCS bucket (you'll upload data to it)
- SoccerNet account (register at [soccer-net.org](https://www.soccer-net.org/))

---

## Step 0: Upload Data to GCS Bucket (One Time)

Choose one method to get SoccerNet data into your GCS bucket:

### Method A: Download in Colab → Upload to GCS (Recommended)

```python
# In a Colab Enterprise notebook
!pip install SoccerNet -q

from SoccerNet.Downloader import SoccerNetDownloader

# Download to local (Colab has fast network)
downloader = SoccerNetDownloader(LocalDirectory="/content/soccernet_download")
downloader.downloadDataTask(
    task="synloc",
    split=["train", "valid", "test", "challenge"]
)

# Set your bucket details
YOUR_BUCKET_NAME = "your-bucket-name"      # Replace with your bucket
YOUR_FOLDER_PATH = "soccernet/synloc"      # Adjust as needed

# Upload to GCS (takes 10-15 minutes)
!gsutil -m cp -r /content/soccernet_download/SpiideoSynLoc gs://{YOUR_BUCKET_NAME}/{YOUR_FOLDER_PATH}/

# Verify upload
!gsutil ls gs://{YOUR_BUCKET_NAME}/{YOUR_FOLDER_PATH}/SpiideoSynLoc/
```

### Method B: Upload from Local Machine

```bash
# From your Mac/PC terminal (if you already downloaded locally)
cd /path/to/your/data

gsutil -m cp -r SpiideoSynLoc gs://YOUR_BUCKET_NAME/soccernet/synloc/
```

### Verify Data Structure

Your bucket should have:
```
gs://YOUR_BUCKET_NAME/soccernet/synloc/SpiideoSynLoc/
├── annotations/
│   ├── train.json (280 MB)
│   ├── val.json (45 MB)
│   ├── test.json (62 MB)
│   └── challenge_public.json (8.4 MB)
└── fullhd/
    ├── train/ (12 GB - 42,504 images)
    ├── val/ (1.9 GB - 6,777 images)
    ├── test/ (2.7 GB - 9,309 images)
    └── challenge/ (3.3 GB - 11,352 images)
```

---

## Step 1: Create New Notebook

1. Open Colab Enterprise: https://colab.cloud.google.com
2. Click **File → New notebook**
3. Rename to "SoccerNet Training"
4. Select runtime: **GPU (T4, V100, or A100)**

---

## Step 2: Environment Setup

Copy and run this cell:

```python
# Check GPU availability
!nvidia-smi

# Clone repository
!git clone https://github.com/geledek/soccernet-synloc.git
%cd soccernet-synloc

# Install package (takes ~1 min)
!pip install -e . -q

print("✓ Setup complete!")
```

---

## Step 3: Configure GCS Access

```python
# GCS Configuration - REPLACE WITH YOUR VALUES
GCS_BUCKET_NAME = 'your-bucket-name'      # Your GCS bucket
GCS_FOLDER_PATH = 'soccernet/synloc'      # Path within bucket
GCS_DATA_PATH = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/SpiideoSynLoc'

# Verify data exists
!gsutil ls {GCS_DATA_PATH}/

# Expected output:
# gs://your-bucket-name/soccernet/synloc/SpiideoSynLoc/annotations/
# gs://your-bucket-name/soccernet/synloc/SpiideoSynLoc/fullhd/
```

---

## Step 4: Download Data to Local SSD

**Important:** This copies ~14GB to local SSD (5-10 minutes)

```python
!mkdir -p /content/synloc

# Download annotations (~400MB)
!gsutil -m cp -r {GCS_DATA_PATH}/annotations /content/synloc/

# Download training images (~12GB)
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/train /content/synloc/

# Download validation images (~2GB)
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/val /content/synloc/

print("✓ Data download complete!")
```

> **Tip:** Use `-m` flag for parallel transfers (much faster)

---

## Step 5: Organize Data Structure

```python
from pathlib import Path

DATA_ROOT = Path('/content/synloc')

# Create expected directory structure
splits = [
    ('train', 'train', 'train'),
    ('valid', 'val', 'val'),
]

for split_name, ann_file, img_folder in splits:
    split_dir = DATA_ROOT / split_name
    split_dir.mkdir(exist_ok=True)

    # Copy annotation
    !cp {DATA_ROOT}/annotations/{ann_file}.json {split_dir}/annotations.json

    # Create symlink to images
    !ln -s {DATA_ROOT}/{img_folder} {split_dir}/images

# Verify structure
!ls -la /content/synloc/train/
!ls -la /content/synloc/valid/

# Check image counts
print(f"Train images: {len(list(Path('/content/synloc/train/images').glob('*.jpg')))}")
print(f"Valid images: {len(list(Path('/content/synloc/valid/images').glob('*.jpg')))}")
```

Expected output:
```
Train images: 42504
Valid images: 6777
```

---

## Step 6: Training Configuration

```python
import torch
from torch.utils.data import DataLoader
from synloc.models import YOLOXPose
from synloc.data import SynLocDataset, get_train_transforms, get_val_transforms
from synloc.training import SynLocTrainer

# Training configuration
config = {
    'model_variant': 'tiny',    # Options: tiny, s, m, l
    'input_size': (640, 640),   # Image size
    'batch_size': 16,           # Reduce if OOM (8, 4)
    'epochs': 100,              # Training epochs
    'lr': 1e-3,                 # Learning rate
}

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")
```

---

## Step 7: Create Model & Datasets

```python
# Create model
model = YOLOXPose(
    variant=config['model_variant'],
    num_keypoints=2,
    input_size=config['input_size']
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Create training dataset
train_dataset = SynLocDataset(
    ann_file='/content/synloc/train/annotations.json',
    img_dir='/content/synloc/train/images',
    transforms=get_train_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

# Create validation dataset
val_dataset = SynLocDataset(
    ann_file='/content/synloc/valid/annotations.json',
    img_dir='/content/synloc/valid/images',
    transforms=get_val_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

print(f"Train dataset: {len(train_dataset)} images")
print(f"Val dataset: {len(val_dataset)} images")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=2,
    collate_fn=SynLocDataset.collate_fn,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=2,
    collate_fn=SynLocDataset.collate_fn,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

---

## Step 8: Setup Checkpoint Persistence

```python
# Local checkpoint directory (ephemeral)
CHECKPOINT_DIR = '/content/checkpoints'
!mkdir -p {CHECKPOINT_DIR}

# GCS checkpoint bucket (persistent)
CHECKPOINT_BUCKET = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/checkpoints'

print(f"Local checkpoints: {CHECKPOINT_DIR}")
print(f"GCS checkpoints: {CHECKPOINT_BUCKET}")
```

---

## Step 9: Train Model

```python
# Create trainer
trainer = SynLocTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    lr=config['lr'],
    epochs=config['epochs'],
    checkpoint_dir=CHECKPOINT_DIR,
    use_amp=True,
    save_interval=10  # Save every 10 epochs
)

# Start training (takes several hours)
history = trainer.train()

print("✓ Training complete!")
```

---

## Step 10: Save Checkpoints to GCS

**Important:** Save to GCS before session ends!

```python
# Upload final checkpoints
!gsutil -m cp {CHECKPOINT_DIR}/*.pth {CHECKPOINT_BUCKET}/

# Upload training config
import json
config_path = f'{CHECKPOINT_DIR}/config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

!gsutil cp {config_path} {CHECKPOINT_BUCKET}/

print(f"✓ Checkpoints saved to {CHECKPOINT_BUCKET}")
```

---

## Step 11: Evaluation (Optional)

```python
from synloc.evaluation import run_inference, evaluate_predictions

# Load best checkpoint
checkpoint = torch.load(f'{CHECKPOINT_DIR}/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Download test data if needed
if not Path('/content/synloc/test').exists():
    !gsutil -m cp -r {GCS_DATA_PATH}/fullhd/test /content/synloc/
    !cp {DATA_ROOT}/annotations/test.json /content/synloc/test/annotations.json
    !ln -s /content/synloc/test /content/synloc/test/images

# Create test dataset
test_dataset = SynLocDataset(
    ann_file='/content/synloc/test/annotations.json',
    img_dir='/content/synloc/test/images',
    transforms=get_val_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=2,
    collate_fn=SynLocDataset.collate_fn
)

# Run inference
results = run_inference(model, test_loader, device='cuda')

# Evaluate with mAP-LocSim
metrics = evaluate_predictions(
    gt_file='/content/synloc/test/annotations.json',
    results=results,
    position_from_keypoint_index=1
)

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"mAP-LocSim:     {metrics['mAP_locsim']:.4f}")
print(f"Precision:      {metrics['precision']:.4f}")
print(f"Recall:         {metrics['recall']:.4f}")
print(f"F1 Score:       {metrics['f1']:.4f}")
print(f"Score Threshold: {metrics['score_threshold']:.4f}")
print(f"Frame Accuracy: {metrics['frame_accuracy']:.4f}")
print("="*50)
```

---

## Step 12: Generate Submission

```python
from synloc.evaluation import format_results_for_submission, create_submission_zip

# Download challenge data
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/challenge /content/synloc/
!cp {DATA_ROOT}/annotations/challenge_public.json /content/synloc/challenge/annotations.json
!ln -s /content/synloc/challenge /content/synloc/challenge/images

# Create challenge dataset
challenge_dataset = SynLocDataset(
    ann_file='/content/synloc/challenge/annotations.json',
    img_dir='/content/synloc/challenge/images',
    transforms=get_val_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

challenge_loader = DataLoader(
    challenge_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=2,
    collate_fn=SynLocDataset.collate_fn
)

# Run inference
results = run_inference(model, challenge_loader, device='cuda')

# Filter by optimal threshold
filtered_results = [r for r in results if r['score'] >= metrics['score_threshold']]

print(f"Total detections: {len(filtered_results)}")

# Create submission
SUBMISSION_DIR = '/content/submissions'
!mkdir -p {SUBMISSION_DIR}

results_path, metadata_path = format_results_for_submission(
    results=filtered_results,
    score_threshold=metrics['score_threshold'],
    position_from_keypoint_index=1,
    output_dir=SUBMISSION_DIR
)

zip_path = create_submission_zip(
    results_path,
    metadata_path,
    f'{SUBMISSION_DIR}/submission.zip'
)

print(f"✓ Submission created: {zip_path}")

# Upload to GCS
SUBMISSION_BUCKET = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/submissions'
!gsutil cp {zip_path} {SUBMISSION_BUCKET}/submission.zip

print(f"✓ Uploaded to: {SUBMISSION_BUCKET}/submission.zip")

# Download to local machine
from google.colab import files
files.download(zip_path)
```

---

## Resume Training (After Session Timeout)

If your session times out, resume from checkpoint:

```python
# Step 1-7: Same as above (setup, data, model)

# Download checkpoint from GCS
!gsutil cp {CHECKPOINT_BUCKET}/epoch_50.pth {CHECKPOINT_DIR}/

# Load checkpoint
checkpoint = torch.load(f'{CHECKPOINT_DIR}/epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create trainer with starting epoch
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

# Set starting epoch
trainer.start_epoch = checkpoint['epoch']
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training
history = trainer.train()
```

---

## Quick Reference

### Data Locations

```python
# GCS (persistent) - Replace with your values
GCS_DATA = 'gs://YOUR_BUCKET_NAME/soccernet/synloc/SpiideoSynLoc'
GCS_CHECKPOINTS = 'gs://YOUR_BUCKET_NAME/soccernet/synloc/checkpoints'

# Local (ephemeral, fast)
DATA_ROOT = '/content/synloc'
CHECKPOINT_DIR = '/content/checkpoints'
```

### Common Commands

```bash
# List GCS bucket
!gsutil ls gs://YOUR_BUCKET_NAME/soccernet/synloc/

# Copy from GCS to local
!gsutil -m cp -r gs://YOUR_BUCKET_NAME/path /content/

# Copy from local to GCS
!gsutil -m cp -r /content/path gs://YOUR_BUCKET_NAME/

# Check disk space
!df -h /content

# Check GPU usage
!nvidia-smi
```

### Troubleshooting

#### Out of Memory
```python
config['batch_size'] = 8  # Reduce from 16
config['model_variant'] = 'tiny'  # Use smaller model
```

#### Data download slow
```python
# Use -m flag for parallel transfers
!gsutil -m cp -r ...

# Copy only what you need
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/train /content/synloc/  # Train first
```

#### Session timeout
```python
# Save checkpoints frequently
trainer = SynLocTrainer(..., save_interval=5)

# Upload to GCS periodically
!gsutil -m cp {CHECKPOINT_DIR}/*.pth {CHECKPOINT_BUCKET}/
```

---

## Complete Training Script (Single Cell)

For convenience, here's everything in one cell:

<details>
<summary>Click to expand full training script</summary>

```python
# ============================================================
# SOCCERNET SYNLOC - COMPLETE TRAINING SCRIPT
# ============================================================

# === 1. SETUP ===
!git clone https://github.com/geledek/soccernet-synloc.git
%cd soccernet-synloc
!pip install -e . -q

# === 2. GCS CONFIG (REPLACE WITH YOUR VALUES) ===
GCS_BUCKET_NAME = 'your-bucket-name'
GCS_FOLDER_PATH = 'soccernet/synloc'
GCS_DATA_PATH = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/SpiideoSynLoc'

# === 3. DOWNLOAD DATA ===
!mkdir -p /content/synloc
!gsutil -m cp -r {GCS_DATA_PATH}/annotations /content/synloc/
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/train /content/synloc/
!gsutil -m cp -r {GCS_DATA_PATH}/fullhd/val /content/synloc/

# === 4. ORGANIZE DATA ===
from pathlib import Path
DATA_ROOT = Path('/content/synloc')

for split_name, ann_file, img_folder in [('train', 'train', 'train'), ('valid', 'val', 'val')]:
    split_dir = DATA_ROOT / split_name
    split_dir.mkdir(exist_ok=True)
    !cp {DATA_ROOT}/annotations/{ann_file}.json {split_dir}/annotations.json
    !ln -s {DATA_ROOT}/{img_folder} {split_dir}/images

# === 5. CREATE MODEL ===
import torch
from torch.utils.data import DataLoader
from synloc.models import YOLOXPose
from synloc.data import SynLocDataset, get_train_transforms, get_val_transforms
from synloc.training import SynLocTrainer

config = {'model_variant': 'tiny', 'input_size': (640, 640), 'batch_size': 16, 'epochs': 100, 'lr': 1e-3}

model = YOLOXPose(variant=config['model_variant'], num_keypoints=2, input_size=config['input_size'])

train_dataset = SynLocDataset(
    ann_file='/content/synloc/train/annotations.json',
    img_dir='/content/synloc/train/images',
    transforms=get_train_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

val_dataset = SynLocDataset(
    ann_file='/content/synloc/valid/annotations.json',
    img_dir='/content/synloc/valid/images',
    transforms=get_val_transforms(config['input_size'][0]),
    input_size=config['input_size']
)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, collate_fn=SynLocDataset.collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, collate_fn=SynLocDataset.collate_fn, pin_memory=True)

# === 6. TRAIN ===
CHECKPOINT_DIR = '/content/checkpoints'
CHECKPOINT_BUCKET = f'gs://{GCS_BUCKET_NAME}/{GCS_FOLDER_PATH}/checkpoints'
!mkdir -p {CHECKPOINT_DIR}

trainer = SynLocTrainer(model=model, train_loader=train_loader, val_loader=val_loader, device='cuda', lr=config['lr'], epochs=config['epochs'], checkpoint_dir=CHECKPOINT_DIR, use_amp=True)

history = trainer.train()

# === 7. SAVE TO GCS ===
!gsutil -m cp {CHECKPOINT_DIR}/*.pth {CHECKPOINT_BUCKET}/
print(f"✓ Training complete! Checkpoints saved to {CHECKPOINT_BUCKET}")
```

</details>

---

## Next Steps

- Monitor training in TensorBoard (if enabled)
- Run evaluation on test set
- Generate submission for challenge
- Experiment with larger models (s, m, l)
- Try higher resolution (960x960)

## Support

- **Repository:** https://github.com/geledek/soccernet-synloc
- **Issues:** https://github.com/geledek/soccernet-synloc/issues
- **SoccerNet:** https://www.soccer-net.org/
