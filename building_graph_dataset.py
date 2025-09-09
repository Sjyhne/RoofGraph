#!/usr/bin/env python3
"""
Building Graph Dataset for Machine Learning

A PyTorch Dataset class that loads image tiles and their corresponding building graphs,
converting them into a format suitable for training graph prediction models.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pathlib
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import rasterio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BuildingGraph:
    """Container for a single building's graph data within a tile."""
    corners: torch.Tensor          # Shape: [num_corners, 2] - (x, y) coordinates in tile space
    edges: torch.Tensor            # Shape: [num_edges, 2] - edge connections as corner indices
    building_id: int               # Unique building identifier
    num_corners: int               # Number of corners in this building

class BuildingGraphDataset(Dataset):
    """
    PyTorch Dataset for building graph prediction from image tiles.
    
    This dataset loads image tiles and their corresponding building graphs,
    preserving all buildings and their complete geometry within each tile.None 
    
    Args:
        data_path: Path to the data directory containing area subdirectories
        areas: List of area names to include (e.g., ['bergen', 'tromso'])
        tile_size: Size of tiles (assumes square tiles)
        transform: Optional torchvision transforms to apply to images
        normalize_coords: If True, normalize corner coordinates to [0, 1] range
    """
    
    def __init__(self, 
                 data_path: str,
                 areas: List[str] = None,
                 tile_size: int = 512,
                 transform: Optional[transforms.Compose] = None,
                 normalize_coords: bool = True):
        
        self.data_path = pathlib.Path(data_path)
        self.tile_size = tile_size
        self.transform = transform
        self.normalize_coords = normalize_coords
        
        # Default to all areas if none specified
        if areas is None:
            areas = ['bergen', 'kristiansand', 'rana', 'sandvika', 'stavanger', 'tromso']
        self.areas = areas
        
        # Load all tile data
        self.tiles = self._load_all_tiles()
        
        print(f"Loaded {len(self.tiles)} tiles from {len(areas)} areas")
        print(f"Preserving all buildings and complete geometry per tile")
    
    def _load_all_tiles(self) -> List[Dict]:
        """Load all tiles from all specified areas."""
        tiles = []
        
        for area in self.areas:
            area_path = self.data_path / area
            mapping_file = area_path / "tile_to_buildings.json"
            
            if not mapping_file.exists():
                print(f"Warning: No tile mapping found for {area}")
                continue
            
            with open(mapping_file, 'r') as f:
                area_tiles = json.load(f)
            
            for tile_key, building_ids in area_tiles.items():
                tiles.append({
                    'tile_key': f"{area}_{tile_key}",
                    'area': area,
                    'building_ids': building_ids,
                    'original_tile_key': tile_key
                })
        
        return tiles
    
    def _load_tile_image(self, area: str, tile_key: str) -> torch.Tensor:
        """Load the actual tile image as a tensor."""
        # Parse tile key to extract image info
        parts = tile_key.split('_')
        if len(parts) < 6:
            raise ValueError(f"Invalid tile key format: {tile_key}")
        
        image_id = '_'.join(parts[:4])
        tile_x, tile_y, tile_size = int(parts[4]), int(parts[5]), int(parts[6])
        
        # Load image
        image_path = self.data_path / area / "COG" / f"{image_id}.tif"
        
        try:
            with rasterio.open(image_path) as src:
                # Read the specific tile window
                window = rasterio.windows.Window(tile_x, tile_y, tile_size, tile_size)
                tile_data = src.read(window=window)
                
                # Handle different band configurations
                if tile_data.shape[0] == 1:  # Grayscale
                    image = tile_data[0]
                    # Convert to 3-channel by repeating
                    image = np.stack([image, image, image], axis=-1)
                elif tile_data.shape[0] == 3:  # RGB
                    image = np.transpose(tile_data, (1, 2, 0))
                elif tile_data.shape[0] == 4:  # RGBA
                    image = np.transpose(tile_data, (1, 2, 0))[:, :, :3]
                else:
                    image = np.transpose(tile_data, (1, 2, 0))
                
                # Convert to tensor and normalize
                image = torch.from_numpy(image).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC -> CHW
                
                return image
                
        except Exception as e:
            print(f"Warning: Could not load tile image {tile_key}: {e}")
            # Return a blank image
            return torch.zeros((3, tile_size, tile_size), dtype=torch.float32)
    
    def _load_building_data(self, building_id: int, area: str) -> Optional[Dict]:
        """Load building data from JSON file."""
        try:
            building_file = self.data_path / area / "buildings_transformed" / f"{building_id}.json"
            with open(building_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load building {building_id}: {e}")
            return None
    
    def _extract_building_graphs(self, tile_info: Dict) -> List[BuildingGraph]:
        """Extract building graphs that intersect with the tile."""
        area = tile_info['area']
        tile_key = tile_info['original_tile_key']
        building_ids = tile_info['building_ids']
        
        # Parse tile coordinates
        parts = tile_key.split('_')
        image_id = '_'.join(parts[:4])
        tile_x, tile_y, tile_size = int(parts[4]), int(parts[5]), int(parts[6])
        
        graphs = []
        
        for building_id in building_ids:
            building = self._load_building_data(building_id, area)
            if not building:
                continue
            
            # Get corners in this specific image
            if image_id not in building.get('corners', {}):
                continue
            
            corners = building['corners'][image_id]
            if len(corners) == 0:
                continue
            
            # Convert to tile-relative coordinates
            tile_corners = []
            for corner in corners:
                rel_x = corner[0] - tile_x
                rel_y = corner[1] - tile_y
                tile_corners.append([rel_x, rel_y])
            
            # Normalize coordinates to [0, 1] if requested
            if self.normalize_coords:
                tile_corners = np.array(tile_corners) / tile_size
            else:
                tile_corners = np.array(tile_corners)
            
            # Get edges
            edges = building.get('edges', [])
            if len(edges) == 0:
                continue
            
            # Convert edges to tensor format
            edge_list = []
            for corner_idx, connected_corners in enumerate(edges):
                for connected_idx in connected_corners:
                    if connected_idx < len(tile_corners):
                        edge_list.append([corner_idx, connected_idx])
            
            if len(edge_list) == 0:
                continue
            
            # Create building graph
            graph = BuildingGraph(
                corners=torch.tensor(tile_corners, dtype=torch.float32),
                edges=torch.tensor(edge_list, dtype=torch.long),
                building_id=building_id,
                num_corners=len(tile_corners)
            )
            graphs.append(graph)
        
        return graphs
    
    def _prepare_graphs_for_training(self, graphs: List[BuildingGraph]) -> Dict[str, torch.Tensor]:
        """Prepare graphs for training without padding - preserve all data."""
        if not graphs:
            return {
                'corners': torch.zeros((0, 2), dtype=torch.float32),
                'edges': torch.zeros((0, 2), dtype=torch.long),
                'building_ids': torch.zeros(0, dtype=torch.long),
                'num_corners_per_building': torch.zeros(0, dtype=torch.long),
                'num_buildings': torch.tensor(0, dtype=torch.long)
            }
        
        # Collect all corners and edges from all buildings
        all_corners = []
        all_edges = []
        building_ids = []
        num_corners_per_building = []
        corner_offset = 0
        
        for graph in graphs:
            # Add corners
            all_corners.append(graph.corners)
            num_corners_per_building.append(graph.num_corners)
            
            # Add edges with global corner indexing
            for edge in graph.edges:
                global_edge = torch.tensor([
                    corner_offset + edge[0], 
                    corner_offset + edge[1]
                ], dtype=torch.long)
                all_edges.append(global_edge)
            
            building_ids.append(graph.building_id)
            corner_offset += graph.num_corners
        
        return {
            'corners': torch.cat(all_corners, dim=0) if all_corners else torch.zeros((0, 2), dtype=torch.float32),
            'edges': torch.stack(all_edges, dim=0) if all_edges else torch.zeros((0, 2), dtype=torch.long),
            'building_ids': torch.tensor(building_ids, dtype=torch.long),
            'num_corners_per_building': torch.tensor(num_corners_per_building, dtype=torch.long),
            'num_buildings': torch.tensor(len(graphs), dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        
        # Load tile image
        image = self._load_tile_image(tile_info['area'], tile_info['original_tile_key'])
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Extract building graphs (preserves all buildings and geometry)
        graphs = self._extract_building_graphs(tile_info)
        
        # Prepare graphs for training (no padding, all data preserved)
        graph_data = self._prepare_graphs_for_training(graphs)
        
        return {
            'image': image,
            'graphs': graph_data,
            'tile_key': tile_info['tile_key'],
            'area': tile_info['area']
        }
    
    def visualize_batch(self, batch, save_path=None, show_building_ids=True, max_samples=None):
        """
        Visualize a batch of tiles with their building graphs.
        
        Args:
            batch: Batch data from the dataset
            save_path: Optional path to save the visualization
            show_building_ids: Whether to show building IDs as labels
            max_samples: Maximum number of samples to visualize (None for all)
        """
        images = batch['image']
        graphs = batch['graphs']
        tile_keys = batch['tile_keys']
        areas = batch['areas']
        
        num_samples = len(images)
        if max_samples is not None:
            num_samples = min(num_samples, max_samples)
        
        # Create subplot grid
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        
        # Ensure axes is always a 2D array for consistent indexing
        if num_samples == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Define colors for buildings
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 
                 'magenta', 'yellow', 'lime', 'navy', 'teal', 'maroon', 'silver', 'gold', 'coral', 'indigo']
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Get image and convert to numpy
            image = images[i].permute(1, 2, 0).numpy()  # CHW -> HWC
            image = np.clip(image, 0, 1)  # Ensure valid range
            
            # Display image
            ax.imshow(image, origin='upper')
            
            # Get graph data for this sample
            corners = graphs['corners'][i].numpy()
            edges = graphs['edges'][i].numpy()
            building_ids = graphs['building_ids'][i].numpy()
            num_corners_per_building = graphs['num_corners_per_building'][i].numpy()
            num_buildings = graphs['num_buildings'][i].item()
            
            # Convert normalized coordinates back to pixel coordinates for visualization
            if self.normalize_coords:
                corners = corners * self.tile_size
            
            if num_buildings > 0:
                # Draw building graphs
                corner_offset = 0
                for building_idx in range(num_buildings):
                    # Use building ID to get consistent color, but ensure it's within bounds
                    building_id = building_ids[building_idx]
                    color_idx = int(building_id) % len(colors)
                    color = colors[color_idx]
                    
                    num_corners = num_corners_per_building[building_idx]
                    
                    # Get corners for this building
                    building_corners = corners[corner_offset:corner_offset + num_corners]
                    
                    # Draw corners
                    ax.scatter(building_corners[:, 0], building_corners[:, 1], 
                              c=color, s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
                    
                    # Draw edges for this building
                    building_edges = []
                    for edge in edges:
                        if (corner_offset <= edge[0] < corner_offset + num_corners and 
                            corner_offset <= edge[1] < corner_offset + num_corners):
                            # Convert to local indices
                            local_edge = [edge[0] - corner_offset, edge[1] - corner_offset]
                            building_edges.append([
                                [building_corners[local_edge[0], 0], building_corners[local_edge[0], 1]],
                                [building_corners[local_edge[1], 0], building_corners[local_edge[1], 1]]
                            ])
                    
                    if building_edges:
                        edge_lines = LineCollection(building_edges, colors=color, linewidths=1.5, alpha=0.8)
                        ax.add_collection(edge_lines)
                    
                    # Add building ID label
                    if show_building_ids and len(building_corners) > 0:
                        centroid = np.mean(building_corners, axis=0)
                        ax.text(centroid[0], centroid[1], str(building_id), 
                               ha='center', va='center', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       edgecolor=color, alpha=0.9))
                    
                    corner_offset += num_corners
            
            # Set title and labels
            ax.set_title(f'{areas[i]}\n{tile_keys[i]}\n{num_buildings} buildings', fontsize=10)
            ax.set_xlabel('X (pixels)', fontsize=8)
            ax.set_ylabel('Y (pixels)', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def visualize_sample(self, sample, save_path=None, show_building_ids=True):
        """
        Visualize a single sample from the dataset.
        
        Args:
            sample: Single sample data from the dataset
            save_path: Optional path to save the visualization
            show_building_ids: Whether to show building IDs as labels
        """
        # Convert single sample to batch format for visualization
        batch = {
            'image': sample['image'].unsqueeze(0),  # Add batch dimension
            'graphs': {
                'corners': [sample['graphs']['corners']],
                'edges': [sample['graphs']['edges']],
                'building_ids': [sample['graphs']['building_ids']],
                'num_corners_per_building': [sample['graphs']['num_corners_per_building']],
                'num_buildings': sample['graphs']['num_buildings'].unsqueeze(0)
            },
            'tile_keys': [sample['tile_key']],
            'areas': [sample['area']]
        }
        
        self.visualize_batch(batch, save_path=save_path, show_building_ids=show_building_ids)

def create_data_loaders(data_path: str,
                       areas: List[str] = None,
                       train_split: float = 0.8,
                       batch_size: int = 4,
                       num_workers: int = 2,
                       **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_path: Path to the data directory
        areas: List of areas to include
        train_split: Fraction of data to use for training
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for BuildingGraphDataset
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = BuildingGraphDataset(data_path, areas=areas, **dataset_kwargs)
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn  # Use custom collate function for variable-sized data
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """
    Custom collate function for batching variable-sized graph data.
    
    Since we preserve all buildings and geometry, we need to handle variable sizes.
    """
    images = torch.stack([item['image'] for item in batch])
    
    # For variable-sized data, we return lists instead of stacked tensors
    graphs = {
        'corners': [item['graphs']['corners'] for item in batch],
        'edges': [item['graphs']['edges'] for item in batch],
        'building_ids': [item['graphs']['building_ids'] for item in batch],
        'num_corners_per_building': [item['graphs']['num_corners_per_building'] for item in batch],
        'num_buildings': torch.stack([item['graphs']['num_buildings'] for item in batch])
    }
    
    return {
        'image': images,
        'graphs': graphs,
        'tile_keys': [item['tile_key'] for item in batch],
        'areas': [item['area'] for item in batch]
    }

# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    print("Testing BuildingGraphDataset...")
    
    # Create dataset
    dataset = BuildingGraphDataset(
        data_path="data",
        areas=['bergen'],  # Test with just one area
        tile_size=512,
        normalize_coords=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\\nSample data:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Number of buildings: {sample['graphs']['num_buildings'].item()}")
        print(f"  Total corners: {sample['graphs']['corners'].shape[0]}")
        print(f"  Total edges: {sample['graphs']['edges'].shape[0]}")
        print(f"  Corners per building: {sample['graphs']['num_corners_per_building']}")
        print(f"  Building IDs: {sample['graphs']['building_ids']}")
        print(f"  Tile key: {sample['tile_key']}")
    
    # Test data loaders
    print("\\nTesting data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_path="data",
        areas=['bergen'],
        batch_size=2,
        num_workers=0  # Use 0 for debugging
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"\\nBatch data:")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Number of samples in batch: {len(batch['graphs']['corners'])}")
        print(f"  Sample 0 - Corners: {batch['graphs']['corners'][0].shape}, Edges: {batch['graphs']['edges'][0].shape}")
        print(f"  Sample 1 - Corners: {batch['graphs']['corners'][1].shape}, Edges: {batch['graphs']['edges'][1].shape}")
        print(f"  Number of buildings per sample: {batch['graphs']['num_buildings']}")
        
        # Visualize the batch
        print("\\nVisualizing batch...")
        dataset.visualize_batch(batch, save_path='batch_visualization.png', max_samples=4)
        break  # Just test first batch
    
    # Test single sample visualization
    print("\\nTesting single sample visualization...")
    sample = dataset[0]
    dataset.visualize_sample(sample, save_path='single_sample_visualization.png')
    
    print("\\nDataset test completed!")
