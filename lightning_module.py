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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from building_graph_dataset import BuildingGraphDataset, create_data_loaders


class SimpleCNN(nn.Module):
    """Simple CNN backbone for feature extraction from images."""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Modify first layer if input channels != 3
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace classifier with feature extractor
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)
        
    def forward(self, x):
        return self.backbone(x)


class BuildingGraphPredictor(nn.Module):
    """
    Model that predicts building graphs from images.
    
    This is a simple baseline that predicts:
    - Number of buildings in the tile
    - For each building: number of corners and their coordinates
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 feature_dim: int = 256,
                 max_buildings: int = 10,
                 max_corners_per_building: int = 50):
        super().__init__()
        
        self.max_buildings = max_buildings
        self.max_corners_per_building = max_corners_per_building
        
        # Image feature extractor
        self.feature_extractor = SimpleCNN(input_channels, feature_dim)
        
        # Building count predictor
        self.building_count_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_buildings + 1)  # 0 to max_buildings
        )
        
        # Corner count predictor (per building)
        self.corner_count_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_buildings * (max_corners_per_building + 1))
        )
        
        # Corner coordinate predictor
        self.corner_coords_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, max_buildings * max_corners_per_building * 2),  # x, y coords
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, images):
        batch_size = images.shape[0]
        
        # Extract features
        features = self.feature_extractor(images)
        
        # Predict building counts
        building_counts = self.building_count_head(features)
        
        # Predict corner counts (reshape to [batch, max_buildings, max_corners+1])
        corner_counts = self.corner_count_head(features)
        corner_counts = corner_counts.view(batch_size, self.max_buildings, -1)
        
        # Predict corner coordinates (reshape to [batch, max_buildings, max_corners, 2])
        corner_coords = self.corner_coords_head(features)
        corner_coords = corner_coords.view(
            batch_size, self.max_buildings, self.max_corners_per_building, 2
        )
        
        return {
            'building_counts': building_counts,
            'corner_counts': corner_counts,
            'corner_coords': corner_coords
        }


class BuildingGraphLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for building graph prediction.
    """
    
    def __init__(self,
                 model_config: Dict[str, Any] = None,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 scheduler_type: str = "cosine",  # "cosine", "plateau", or "none"
                 max_buildings: int = 10,
                 max_corners_per_building: int = 50):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model configuration
        if model_config is None:
            model_config = {
                'input_channels': 3,
                'feature_dim': 256,
                'max_buildings': max_buildings,
                'max_corners_per_building': max_corners_per_building
            }
        
        # Initialize model
        self.model = BuildingGraphPredictor(**model_config)
        
        # Loss weights
        self.building_count_weight = 1.0
        self.corner_count_weight = 1.0
        self.corner_coord_weight = 1.0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        return self.model(x)
    
    def _prepare_targets(self, batch):
        """Convert batch data to model targets."""
        batch_size = len(batch['graphs']['corners'])
        device = self.device
        
        # Initialize target tensors
        building_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        corner_counts = torch.zeros(
            batch_size, self.hparams.max_buildings, self.hparams.max_corners_per_building + 1, 
            device=device
        )
        corner_coords = torch.zeros(
            batch_size, self.hparams.max_buildings, self.hparams.max_corners_per_building, 2,
            device=device
        )
        
        for i in range(batch_size):
            num_buildings = batch['graphs']['num_buildings'][i].item()
            building_counts[i] = min(num_buildings, self.hparams.max_buildings)
            
            if num_buildings > 0:
                corners = batch['graphs']['corners'][i]
                num_corners_per_building = batch['graphs']['num_corners_per_building'][i]
                
                corner_offset = 0
                for b in range(min(num_buildings, self.hparams.max_buildings)):
                    num_corners = num_corners_per_building[b].item()
                    num_corners = min(num_corners, self.hparams.max_corners_per_building)
                    
                    # One-hot encode corner count
                    corner_counts[i, b, num_corners] = 1.0
                    
                    # Copy corner coordinates
                    if num_corners > 0:
                        end_idx = corner_offset + num_corners
                        building_corners = corners[corner_offset:end_idx]
                        corner_coords[i, b, :num_corners] = building_corners
                    
                    corner_offset += num_corners_per_building[b].item()
        
        return {
            'building_counts': building_counts,
            'corner_counts': corner_counts,
            'corner_coords': corner_coords
        }
    
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
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            return [optimizer], [scheduler]
        elif self.hparams.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
            }
        else:
            return optimizer


def create_lightning_data_module(data_path: str,
                                areas: List[str] = None,
                                batch_size: int = 4,
                                num_workers: int = 2,
                                train_split: float = 0.8,
                                **dataset_kwargs):
    """
    Create PyTorch Lightning DataModule.
    """
    
    class BuildingGraphDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.data_path = data_path
            self.areas = areas
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.train_split = train_split
            self.dataset_kwargs = dataset_kwargs
            
        def setup(self, stage=None):
            # Create data loaders
            self.train_loader, self.val_loader = create_data_loaders(
                data_path=self.data_path,
                areas=self.areas,
                train_split=self.train_split,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                **self.dataset_kwargs
            )
            
        def train_dataloader(self):
            return self.train_loader
            
        def val_dataloader(self):
            return self.val_loader
    
    return BuildingGraphDataModule()


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
