"""
Plain PyTorch training loop for YOLOX-Pose.

Colab-friendly implementation with:
- AMP (automatic mixed precision) support
- Checkpoint saving/loading
- Progress tracking with tqdm
- Optional wandb logging
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Optional, Callable
from pathlib import Path
import json

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class SynLocTrainer:
    """Training loop for YOLOX-Pose on SynLoc dataset.

    Features:
    - Mixed precision training (AMP)
    - Learning rate scheduling
    - Checkpoint management (Google Drive compatible)
    - Progress tracking with tqdm
    - Optional Weights & Biases logging

    Args:
        model: YOLOX-Pose model.
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler (optional).
        device: Training device ('cuda' or 'cpu').
        use_amp: Whether to use automatic mixed precision.
        checkpoint_dir: Directory for saving checkpoints.
        log_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.

    Example:
        >>> model = YOLOXPose(variant='s')
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> trainer = SynLocTrainer(model, train_loader, optimizer=optimizer)
        >>> trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        checkpoint_dir: str = 'checkpoints',
        log_wandb: bool = False,
        wandb_project: str = 'synloc'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_wandb = log_wandb and HAS_WANDB

        # Optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.05
            )
        self.optimizer = optimizer
        self.scheduler = scheduler

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Initialize wandb
        if self.log_wandb:
            wandb.init(project=wandb_project, config={
                'model': model.__class__.__name__,
                'optimizer': optimizer.__class__.__name__,
                'lr': optimizer.param_groups[0]['lr'],
                'use_amp': self.use_amp,
            })

    def train(
        self,
        num_epochs: int,
        eval_interval: int = 1,
        save_interval: int = 10,
        eval_fn: Optional[Callable] = None
    ) -> Dict:
        """Run training loop.

        Args:
            num_epochs: Number of epochs to train.
            eval_interval: Evaluate every N epochs.
            save_interval: Save checkpoint every N epochs.
            eval_fn: Custom evaluation function (model, loader) -> dict.

        Returns:
            Training history dict.
        """
        history = {'train_loss': [], 'val_metric': []}

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Training epoch
            train_loss = self._train_epoch()
            history['train_loss'].append(train_loss)

            # Evaluation
            if self.val_loader is not None and (epoch + 1) % eval_interval == 0:
                if eval_fn is not None:
                    metrics = eval_fn(self.model, self.val_loader)
                else:
                    metrics = self._validate()
                history['val_metric'].append(metrics)

                # Track best
                main_metric = metrics.get('mAP', metrics.get('map_locsim', 0))
                if main_metric > self.best_metric:
                    self.best_metric = main_metric
                    self.save_checkpoint('best.pth')

            # Save periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pth')

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final checkpoint
        self.save_checkpoint('final.pth')

        return history

    def _train_epoch(self) -> float:
        """Run single training epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                # Get model outputs
                outputs = self.model(images)

                # For now, just track that forward pass works
                # Full loss computation requires target assignment
                # which would need the SimOTA assigner
                loss = self._compute_simple_loss(outputs, batch)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Log to wandb
            if self.log_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

        return total_loss / num_batches

    def _compute_simple_loss(self, outputs, batch) -> torch.Tensor:
        """Compute simplified loss for initial training setup.

        This is a placeholder - full training requires SimOTA assignment.
        For now, we just ensure the forward pass works and gradients flow.
        """
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = outputs

        # Simple objectness loss as placeholder
        # In full implementation, need target assignment via SimOTA
        total_loss = 0.0
        for obj in objectnesses:
            # Dummy target: assume mostly background
            target = torch.zeros_like(obj)
            loss = nn.functional.binary_cross_entropy_with_logits(obj, target)
            total_loss = total_loss + loss

        return total_loss / len(objectnesses)

    def _validate(self) -> Dict:
        """Run validation.

        Returns:
            Validation metrics dict.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                loss = self._compute_simple_loss(outputs, batch)
                total_loss += loss.item()

        return {'val_loss': total_loss / len(self.val_loader)}

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f'Saved checkpoint: {path}')

    def load_checkpoint(self, filename: str):
        """Load training checkpoint.

        Args:
            filename: Checkpoint filename or path.
        """
        path = Path(filename)
        if not path.exists():
            path = self.checkpoint_dir / filename

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', 0.0)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f'Loaded checkpoint from epoch {self.epoch}')

    def resume(self, checkpoint_dir: Optional[str] = None):
        """Resume from latest checkpoint.

        Args:
            checkpoint_dir: Directory to search for checkpoints.
        """
        search_dir = Path(checkpoint_dir) if checkpoint_dir else self.checkpoint_dir

        # Find latest checkpoint
        checkpoints = list(search_dir.glob('epoch_*.pth'))
        if not checkpoints:
            print('No checkpoints found, starting from scratch')
            return

        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
        self.load_checkpoint(latest)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """Create cosine learning rate schedule with warmup.

    Args:
        optimizer: Optimizer instance.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        min_lr_ratio: Minimum LR ratio at end.

    Returns:
        LambdaLR scheduler.
    """
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
