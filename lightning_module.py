#!/usr/bin/env python3
"""
PyTorch Lightning Module for Building Graph Prediction

A Lightning module that trains models to predict building graphs from aerial imagery.
Supports different model architectures and training strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from typing import Dict, Any



class BuildingGraphLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for building graph prediction.
    """
    
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 scheduler_type: str = "cosine",  # "cosine", "plateau", or "none"
                 max_corners_per_building: int = 50):
        super().__init__()
        
        self.save_hyperparameters()

        self.model = model
        
        # Loss weights
        self.building_count_weight = 1.0
        self.corner_count_weight = 1.0
        self.corner_coord_weight = 1.0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        return self.model(x)


    def _compute_loss(self, predictions, targets):
        """Compute multi-task loss."""
        
        # Building count loss (classification)
        building_count_loss = F.cross_entropy(
            predictions['building_counts'], 
            targets['building_counts']
        )
        
        # Corner count loss (per building, multi-class)
        corner_count_loss = F.cross_entropy(
            predictions['corner_counts'].view(-1, self.hparams.max_corners_per_building + 1),
            targets['corner_counts'].view(-1, self.hparams.max_corners_per_building + 1).argmax(dim=-1)
        )
        
        # Corner coordinate loss (MSE for existing corners)
        coord_loss = F.mse_loss(
            predictions['corner_coords'],
            targets['corner_coords'],
            reduction='mean'
        )
        
        # Combine losses
        total_loss = (
            self.building_count_weight * building_count_loss +
            self.corner_count_weight * corner_count_loss +
            self.corner_coord_weight * coord_loss
        )
        
        return {
            'total_loss': total_loss,
            'building_count_loss': building_count_loss,
            'corner_count_loss': corner_count_loss,
            'coord_loss': coord_loss
        }
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        
        # Forward pass
        predictions = self(images)
        
        # Prepare targets
        targets = self._prepare_targets(batch)
        
        # Compute loss
        losses = self._compute_loss(predictions, targets)
        
        # Log metrics
        self.log('train_loss', losses['total_loss'], prog_bar=True)
        self.log('train_building_count_loss', losses['building_count_loss'])
        self.log('train_corner_count_loss', losses['corner_count_loss'])
        self.log('train_coord_loss', losses['coord_loss'])
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        
        # Forward pass
        predictions = self(images)
        
        # Prepare targets
        targets = self._prepare_targets(batch)
        
        # Compute loss
        losses = self._compute_loss(predictions, targets)
        
        # Log metrics
        self.log('val_loss', losses['total_loss'], prog_bar=True)
        self.log('val_building_count_loss', losses['building_count_loss'])
        self.log('val_corner_count_loss', losses['corner_count_loss'])
        self.log('val_coord_loss', losses['coord_loss'])
        
        # Compute accuracy metrics
        building_count_acc = (
            predictions['building_counts'].argmax(dim=1) == targets['building_counts']
        ).float().mean()
        
        self.log('val_building_count_acc', building_count_acc)
        
        return losses['total_loss']
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Get logged metrics
        metrics = self.trainer.callback_metrics
        
        # Print aggregate results
        print(f"\n=== Training Epoch {self.current_epoch} Complete ===")
        if 'train_loss' in metrics:
            print(f"Train Loss: {metrics['train_loss']:.4f}")
        if 'train_building_count_loss' in metrics:
            print(f"Train Building Count Loss: {metrics['train_building_count_loss']:.4f}")
        if 'train_corner_count_loss' in metrics:
            print(f"Train Corner Count Loss: {metrics['train_corner_count_loss']:.4f}")
        if 'train_coord_loss' in metrics:
            print(f"Train Coord Loss: {metrics['train_coord_loss']:.4f}")
        print("=" * 50)
        
        # Metrics are automatically logged to wandb through self.log() calls
    
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        # Get logged metrics
        metrics = self.trainer.callback_metrics
        
        # Print aggregate results
        print(f"\n=== Validation Epoch {self.current_epoch} Complete ===")
        if 'val_loss' in metrics:
            print(f"Val Loss: {metrics['val_loss']:.4f}")
        if 'val_building_count_loss' in metrics:
            print(f"Val Building Count Loss: {metrics['val_building_count_loss']:.4f}")
        if 'val_corner_count_loss' in metrics:
            print(f"Val Corner Count Loss: {metrics['val_corner_count_loss']:.4f}")
        if 'val_coord_loss' in metrics:
            print(f"Val Coord Loss: {metrics['val_coord_loss']:.4f}")
        if 'val_building_count_acc' in metrics:
            print(f"Val Building Count Accuracy: {metrics['val_building_count_acc']:.4f}")
        print("=" * 50)
        
        # Metrics are automatically logged to wandb through self.log() calls
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


# Example usage and testing
if __name__ == "__main__":
    print("Testing BuildingGraphLightningModule...")
    
    # Create model
    model = BuildingGraphLightningModule(
        learning_rate=1e-3,
        max_buildings=5,
        max_corners_per_building=20
    )
    
    # Create dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 512, 512)
    
    # Test forward pass
    with torch.no_grad():
        predictions = model(dummy_images)
        print(f"Predictions shapes:")
        print(f"  Building counts: {predictions['building_counts'].shape}")
        print(f"  Corner counts: {predictions['corner_counts'].shape}")
        print(f"  Corner coords: {predictions['corner_coords'].shape}")
    
    print("\nLightning module created successfully!")
