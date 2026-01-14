# Managing Large Datasets with Google Drive for Colab

## Overview

The SoccerNet SynLoc dataset is ~20GB (images only), which requires careful management in Google Colab. This guide covers strategies for efficient data handling.

## Dataset Sizes

| Split     | Images | Size  |
|-----------|--------|-------|
| train     | 42,504 | 12 GB |
| valid     | 6,777  | 1.9 GB|
| test      | 9,309  | 2.7 GB|
| challenge | 11,352 | 3.3 GB|
| mini      | 7      | 2 MB  |
| **Total** | 69,949 | ~20 GB|

## Strategy 1: Store Everything on Drive (Recommended)

Best for: Colab Pro users with 100GB+ Drive storage.

### Setup (One Time)

```python
from google.colab import drive
drive.mount('/content/drive')

# Create data directory
!mkdir -p /content/drive/MyDrive/SoccerNet/synloc

# Upload from local machine or download directly
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory='/content/drive/MyDrive/SoccerNet/synloc')
downloader.downloadDataTask(task="synloc", split=["train", "valid", "test", "challenge"])
```

### Usage in Training

```python
DATA_ROOT = '/content/drive/MyDrive/SoccerNet/synloc'

# Reading directly from Drive (slower but persistent)
train_dataset = SynLocDataset(
    ann_file=f'{DATA_ROOT}/train/annotations.json',
    img_dir=f'{DATA_ROOT}/train/images',
    ...
)
```

**Pros:** Data persists across sessions
**Cons:** Slower I/O (~2-5x slower than local SSD)

---

## Strategy 2: Hybrid - Annotations on Drive, Images Cached Locally

Best for: Users with limited Drive space or wanting faster I/O.

### Setup

```python
from google.colab import drive
drive.mount('/content/drive')

# Keep annotations on Drive (small, ~400MB total)
ANN_ROOT = '/content/drive/MyDrive/SoccerNet/synloc/annotations'

# Copy images to local SSD (faster I/O)
!mkdir -p /content/synloc_images
!cp -r /content/drive/MyDrive/SoccerNet/synloc/train/images /content/synloc_images/train

# Or download directly to local
# Note: Will be lost when session ends
```

### Usage

```python
train_dataset = SynLocDataset(
    ann_file='/content/drive/MyDrive/SoccerNet/synloc/train/annotations.json',
    img_dir='/content/synloc_images/train',  # Local SSD
    ...
)
```

**Pros:** Fast training, annotations persist
**Cons:** Need to re-copy images each session

---

## Strategy 3: Use Zip Files for Transfer

Best for: Moving data between local machine and Colab.

### Upload Workflow

```bash
# On local machine: Create optimized zip
cd /path/to/synloc/data
zip -r synloc_train.zip train/ -x "*.DS_Store"

# Upload to Drive via browser or rclone
```

### In Colab

```python
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

# Extract to local SSD (much faster than working from Drive)
!unzip -q /content/drive/MyDrive/synloc_train.zip -d /content/

# Now use local path
DATA_ROOT = '/content/train'
```

---

## Strategy 4: Mini Dataset for Development

Best for: Initial development, debugging, testing pipeline.

```python
# Use mini split (only 7 images, 2MB)
DATA_ROOT = '/content/drive/MyDrive/SoccerNet/synloc/mini'

# Or create your own mini split
import shutil
import random

# Sample 100 images from train
train_images = list(Path('train/images').glob('*.jpg'))
sample = random.sample(train_images, 100)

for img in sample:
    shutil.copy(img, 'mini_train/images/')
```

---

## Performance Tips

### 1. Use WebDataset Format (Advanced)

Convert to WebDataset for streaming:

```python
import webdataset as wds

# Create tar shards
# Each shard ~1GB, can stream efficiently
```

### 2. Prefetch Data

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,      # Parallel loading
    prefetch_factor=2,  # Prefetch batches
    pin_memory=True,    # Faster GPU transfer
)
```

### 3. Cache Transforms

```python
# For expensive transforms, cache the result
from functools import lru_cache

@lru_cache(maxsize=1000)
def load_and_transform(img_path):
    ...
```

### 4. Use Memory-Mapped Files

```python
import numpy as np

# Pre-convert images to numpy and memory-map
data = np.load('train_images.npy', mmap_mode='r')
```

---

## Recommended Directory Structure on Drive

```
MyDrive/
└── SoccerNet/
    ├── synloc/
    │   ├── train/
    │   │   ├── annotations.json
    │   │   └── images/
    │   ├── valid/
    │   ├── test/
    │   └── challenge/
    ├── checkpoints/
    │   ├── yolox_tiny_epoch_100.pth
    │   └── config.json
    └── submissions/
        └── submission_20240115.zip
```

---

## Colab Storage Limits

| Tier       | RAM   | Disk  | GPU Time | Drive |
|------------|-------|-------|----------|-------|
| Free       | 12GB  | 78GB  | ~4hr     | 15GB  |
| Pro        | 25GB  | 166GB | ~24hr    | 100GB |
| Pro+       | 52GB  | 225GB | ~24hr    | 2TB   |

---

## Quick Reference

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Check available space
!df -h /content
!df -h /content/drive

# Copy to local (fast I/O)
!cp -r /content/drive/MyDrive/data /content/

# Check GPU
!nvidia-smi

# Save checkpoint to Drive
torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoints/model.pth')
```
