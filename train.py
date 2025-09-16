#!/usr/bin/env python3
"""
Training script for Building Graph Prediction using PyTorch Lightning
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from lightning_module import BuildingGraphLightningModule, create_lightning_data_module


def main():
    parser = argparse.ArgumentParser(description='Train Building Graph Prediction Model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to data directory')
    parser.add_argument('--areas', nargs='+', default=['bergen'], 
                       help='Areas to include in training')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Tile size for images')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    
    # Model arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--max_buildings', type=int, default=10,
                       help='Maximum number of buildings per tile')
    parser.add_argument('--max_corners_per_building', type=int, default=50,
                       help='Maximum number of corners per building')
    
    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0,
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32],
                       help='Training precision')
    
    # Logging arguments
    parser.add_argument('--experiment_name', type=str, default='building_graph_prediction',
                       help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='lightning_logs',
                       help='Directory for logs')
    
    args = parser.parse_args()
    
    print("=== Building Graph Prediction Training ===")
    print(f"Data path: {args.data_path}")
    print(f"Areas: {args.areas}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Using {'GPU' if args.gpus > 0 else 'CPU'}")
    
    # Create data module
    data_module = create_lightning_data_module(
        data_path=args.data_path,
        areas=args.areas,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        tile_size=args.tile_size,
        normalize_coords=True
    )
    
    # Create model
    model_config = {
        'input_channels': 3,
        'feature_dim': 256,
        'max_buildings': args.max_buildings,
        'max_corners_per_building': args.max_corners_per_building
    }
    
    model = BuildingGraphLightningModule(
        model_config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        max_buildings=args.max_buildings,
        max_corners_per_building=args.max_corners_per_building
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            filename='best-{epoch:02d}-{val_loss:.3f}'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min'
        )
    ]
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Check validation every half epoch
    )
    
    # Train model
    print("\nStarting training...")
    trainer.fit(model, data_module)
    
    # Test best model
    print("\nTesting best model...")
    trainer.test(ckpt_path='best')
    
    print(f"\nTraining completed! Logs saved to: {args.log_dir}")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
