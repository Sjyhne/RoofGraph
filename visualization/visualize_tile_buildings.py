#!/usr/bin/env python3
"""
Visualize tile-specific buildings to show the clipping effect.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
import pathlib
import argparse
import rasterio

def load_original_building(building_id, area, data_path):
    """Load original building data."""
    building_file = data_path / area / "buildings_transformed" / f"{building_id}.json"
    with open(building_file, 'r') as f:
        return json.load(f)

def load_tile_image(image_id, area, data_path, tile_x, tile_y, tile_size):
    """Load the actual tile image from the COG file."""
    image_path = data_path / area / "COG" / f"{image_id}.tif"
    
    if not image_path.exists():
        print(f"Warning: Image file not found: {image_path}")
        return None
    
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
                # For other band counts, take first 3 bands
                image = np.transpose(tile_data[:3], (1, 2, 0))
            
            # Normalize to [0, 1] range
            image = image.astype(np.float32) / 255.0
            
            return image
            
    except Exception as e:
        print(f"Warning: Could not load tile image: {e}")
        return None

def visualize_tile_comparison(tile_key, area, data_path, save_path=None):
    """Visualize original vs tile-specific building geometry."""
    # Load tile-specific data
    tile_buildings_file = data_path / area / "tile_buildings_data.json"
    with open(tile_buildings_file, 'r') as f:
        tile_data = json.load(f)
    
    if tile_key not in tile_data:
        print(f"Tile {tile_key} not found in tile data")
        return
    
    tile_buildings = tile_data[tile_key]
    if not tile_buildings:
        print(f"No buildings in tile {tile_key}")
        return
    
    # Parse tile info
    parts = tile_key.split('_')
    image_id = '_'.join(parts[:4])
    tile_x, tile_y, tile_size = int(parts[4]), int(parts[5]), int(parts[6])
    
    # Load the tile image
    tile_image = load_tile_image(image_id, area, data_path, tile_x, tile_y, tile_size)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Colors for different buildings
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot 1: Original buildings (full extent) with aerial image
    ax1.set_title(f'Original Buildings in Image\n{image_id}')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Show aerial image if available
    if tile_image is not None:
        # For the first plot, we need to show a larger area around the tile
        # Let's load a larger area to show context
        try:
            with rasterio.open(data_path / area / "COG" / f"{image_id}.tif") as src:
                # Load a larger area around the tile for context
                context_size = tile_size * 2
                context_x = max(0, tile_x - tile_size // 2)
                context_y = max(0, tile_y - tile_size // 2)
                context_width = min(context_size, src.width - context_x)
                context_height = min(context_size, src.height - context_y)
                
                window = rasterio.windows.Window(context_x, context_y, context_width, context_height)
                context_data = src.read(window=window)
                
                # Handle different band configurations
                if context_data.shape[0] == 1:  # Grayscale
                    context_image = context_data[0]
                    context_image = np.stack([context_image, context_image, context_image], axis=-1)
                elif context_data.shape[0] == 3:  # RGB
                    context_image = np.transpose(context_data, (1, 2, 0))
                elif context_data.shape[0] == 4:  # RGBA
                    context_image = np.transpose(context_data, (1, 2, 0))[:, :, :3]
                else:
                    context_image = np.transpose(context_data[:3], (1, 2, 0))
                
                context_image = context_image.astype(np.float32) / 255.0
                
                # Display the context image
                ax1.imshow(context_image, extent=[context_x, context_x + context_width, 
                                                context_y + context_height, context_y], 
                          origin='upper', alpha=0.7)
        except Exception as e:
            print(f"Warning: Could not load context image: {e}")
    
    ax1.grid(True, alpha=0.3)
    
    # Add tile boundary rectangle
    tile_rect = patches.Rectangle((tile_x, tile_y), tile_size, tile_size, 
                                linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
    ax1.add_patch(tile_rect)
    
    for i, tile_building in enumerate(tile_buildings):
        building_id = tile_building['original_building_id']
        original_building = load_original_building(building_id, area, data_path)
        
        if image_id in original_building['corners']:
            corners = np.array(original_building['corners'][image_id])
            color = colors[i % len(colors)]
            
            # Plot corners
            ax1.scatter(corners[:, 0], corners[:, 1], c=color, s=40, alpha=0.9, 
                       edgecolors='white', linewidth=1.0, label=f'Building {building_id}')
            
            # Plot edges
            if 'edges' in original_building:
                edges = []
                for corner_idx, connected_corners in enumerate(original_building['edges']):
                    if corner_idx < len(corners):
                        for connected_idx in connected_corners:
                            if connected_idx < len(corners):
                                edges.append([
                                    [corners[corner_idx, 0], corners[corner_idx, 1]],
                                    [corners[connected_idx, 0], corners[connected_idx, 1]]
                                ])
                
                if edges:
                    edge_lines = LineCollection(edges, colors=color, linewidths=2.0, alpha=0.9)
                    ax1.add_collection(edge_lines)
    
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot 2: Tile-specific buildings (clipped) with aerial image
    ax2.set_title(f'Tile-Specific Buildings\n{tile_key}')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    # Show the actual tile image as background
    if tile_image is not None:
        ax2.imshow(tile_image, extent=[0, tile_size, tile_size, 0], 
                  origin='upper', alpha=0.8)
    else:
        # Fallback: add tile boundary rectangle
        tile_rect2 = patches.Rectangle((0, 0), tile_size, tile_size, 
                                     linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax2.add_patch(tile_rect2)
    
    ax2.grid(True, alpha=0.3)
    
    for i, tile_building in enumerate(tile_buildings):
        corners = np.array(tile_building['corners'])
        node_types = tile_building.get('node_types', ['original'] * len(corners))
        color = colors[i % len(colors)]
        
        # Separate original and boundary nodes
        original_mask = np.array([t == 'original' for t in node_types])
        boundary_mask = np.array([t == 'boundary' for t in node_types])
        
        # Plot original corners
        if np.any(original_mask):
            ax2.scatter(corners[original_mask, 0], corners[original_mask, 1], 
                       c=color, s=60, alpha=0.9, marker='o',
                       edgecolors='white', linewidth=1.0, 
                       label=f'Building {tile_building["id"]} (original)')
        
        # Plot boundary corners
        if np.any(boundary_mask):
            ax2.scatter(corners[boundary_mask, 0], corners[boundary_mask, 1], 
                       c=color, s=40, alpha=0.8, marker='s',
                       edgecolors='black', linewidth=1.5,
                       label=f'Building {tile_building["id"]} (boundary)')
        
        # Plot edges
        if tile_building['edges']:
            edges = []
            for corner_idx, connected_corners in enumerate(tile_building['edges']):
                if corner_idx < len(corners):
                    for connected_idx in connected_corners:
                        if connected_idx < len(corners):
                            edges.append([
                                [corners[corner_idx, 0], corners[corner_idx, 1]],
                                [corners[connected_idx, 0], corners[connected_idx, 1]]
                            ])
            
            if edges:
                edge_lines = LineCollection(edges, colors=color, linewidths=2.0, alpha=0.9)
                ax2.add_collection(edge_lines)
    
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.set_xlim(-50, tile_size + 50)
    ax2.set_ylim(-50, tile_size + 50)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize tile-specific buildings")
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--area", type=str, default="rana")
    parser.add_argument("--tile_key", type=str, help="Specific tile key to visualize")
    parser.add_argument("--save_path", type=str, help="Path to save visualization")
    
    args = parser.parse_args()
    
    data_path = pathlib.Path(args.data_path)
    
    if args.tile_key:
        # Visualize specific tile
        visualize_tile_comparison(args.tile_key, args.area, data_path, args.save_path)
    else:
        # Show available tiles
        tile_buildings_file = data_path / args.area / "tile_buildings_data.json"
        with open(tile_buildings_file, 'r') as f:
            tile_data = json.load(f)
        
        print("Available tiles:")
        for i, tile_key in enumerate(list(tile_data.keys())[:10]):  # Show first 10
            num_buildings = len(tile_data[tile_key])
            print(f"  {i}: {tile_key} ({num_buildings} buildings)")
        
        if len(tile_data) > 10:
            print(f"  ... and {len(tile_data) - 10} more tiles")

if __name__ == "__main__":
    main()
