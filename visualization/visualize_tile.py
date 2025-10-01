#!/usr/bin/env python3
"""
Simple tile visualization script for RoofGraph dataset.

This script visualizes a specific tile with its buildings drawn on it.
"""

import json
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import sys

# Optional imports
try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

if not RASTERIO_AVAILABLE and not PIL_AVAILABLE:
    print("Warning: Neither rasterio nor PIL available. Image background will not be shown.")


def load_tile_mapping(data_path):
    """Load the tile to buildings mapping."""
    tile_mapping_path = data_path / "tile_to_buildings.json"
    if not tile_mapping_path.exists():
        print(f"Error: Tile mapping file not found: {tile_mapping_path}")
        return None
    
    with open(tile_mapping_path, 'r') as f:
        return json.load(f)


def load_building_data(buildings_path, building_ids):
    """Load building data for specific building IDs."""
    buildings = {}
    for building_id in building_ids:
        building_file = buildings_path / f"{building_id}.json"
        if building_file.exists():
            try:
                with open(building_file, 'r') as f:
                    buildings[building_id] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load building {building_id}: {e}")
        else:
            print(f"Warning: Building file not found: {building_file}")
    
    return buildings


def parse_tile_key(tile_key):
    """Parse tile key to extract image_id, x, y, and tile_size."""
    # Format: image_id_x_y_tile_size
    parts = tile_key.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid tile key format: {tile_key}")
    
    # Find the last 3 parts (x, y, tile_size)
    x = int(parts[-3])
    y = int(parts[-2])
    tile_size = int(parts[-1])
    
    # Everything before the last 3 parts is the image_id
    image_id = '_'.join(parts[:-3])
    
    return image_id, x, y, tile_size


def load_tile_image(image_path, x, y, tile_size):
    """Load a tile from a GeoTIFF image."""
    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(image_path) as src:
                # Create window for the tile
                window = Window(x, y, tile_size, tile_size)
                
                # Read the tile
                tile_data = src.read(window=window)
                
                # Handle different band configurations
                if tile_data.shape[0] == 1:
                    # Grayscale to RGB
                    tile_data = np.stack([tile_data[0]] * 3, axis=0)
                elif tile_data.shape[0] == 3:
                    # Already RGB
                    pass
                elif tile_data.shape[0] == 4:
                    # RGBA to RGB (drop alpha channel)
                    tile_data = tile_data[:3]
                else:
                    # Take first 3 bands
                    tile_data = tile_data[:3]
                
                # Transpose to (height, width, channels)
                tile_data = np.transpose(tile_data, (1, 2, 0))
                
                # Normalize to 0-1 range
                tile_data = tile_data.astype(np.float32)
                if tile_data.max() > 1.0:
                    tile_data = tile_data / 255.0
                
                return tile_data
        except Exception as e:
            print(f"Warning: Could not load tile image with rasterio: {e}")
    
    # Fallback to PIL if rasterio is not available
    if PIL_AVAILABLE:
        try:
            # Load the full image with PIL
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Crop to the tile region
                tile_img = img.crop((x, y, x + tile_size, y + tile_size))
                
                # Convert to numpy array
                tile_data = np.array(tile_img, dtype=np.float32) / 255.0
                
                return tile_data
        except Exception as e:
            print(f"Warning: Could not load tile image with PIL: {e}")
    
    return None


def get_building_corners_in_tile(building, image_id, tile_x, tile_y, tile_size):
    """Get building corners for visualization, using any available image data."""
    # Try to get corners from the specific image first
    corners = None
    if image_id in building.get('corners', {}):
        corners = building['corners'][image_id]
    
    # If no corners in this image, try to find corners from any other image
    if not corners or len(corners) == 0:
        for other_image_id, other_corners in building.get('corners', {}).items():
            if len(other_corners) > 0:
                corners = other_corners
                print(f"Using corners from image {other_image_id} for building {building.get('id', 'unknown')}")
                break
    
    if not corners or len(corners) == 0:
        return []
    
    # Convert to numpy array for easier processing
    corners_array = np.array(corners)
    
    # Convert all corners to tile-relative coordinates
    # We'll show the full building even if it extends outside the tile
    relative_corners = []
    for corner in corners_array:
        # Convert to tile-relative coordinates
        rel_x = corner[0] - tile_x
        rel_y = corner[1] - tile_y
        
        # Don't clip to tile bounds - show the full building even if it extends outside
        relative_corners.append([rel_x, rel_y])
    
    return relative_corners


def get_building_edges_in_tile(building, image_id, tile_x, tile_y, tile_size):
    """Get building edges for visualization, using any available image data."""
    # Try to get corners from the specific image first
    corners = None
    if image_id in building.get('corners', {}):
        corners = building['corners'][image_id]
    
    # If no corners in this image, try to find corners from any other image
    if not corners or len(corners) == 0:
        for other_image_id, other_corners in building.get('corners', {}).items():
            if len(other_corners) > 0:
                corners = other_corners
                break
    
    if not corners or len(corners) == 0 or 'edges' not in building:
        return []
    
    # Convert to numpy array for easier processing
    corners_array = np.array(corners)
    
    # Get all edges and convert to tile-relative coordinates
    edges = []
    for i, connected_corners in enumerate(building['edges']):
        for connected_idx in connected_corners:
            if connected_idx < len(corners_array):
                # Convert to tile-relative coordinates
                corner1 = corners_array[i]
                corner2 = corners_array[connected_idx]
                
                rel_corner1 = [
                    corner1[0] - tile_x,
                    corner1[1] - tile_y
                ]
                rel_corner2 = [
                    corner2[0] - tile_x,
                    corner2[1] - tile_y
                ]
                
                edges.append([rel_corner1, rel_corner2])
    
    return edges


def visualize_tile(tile_key, data_path, show_image=True, save_path=None):
    """Visualize a tile with its buildings."""
    print(f"Visualizing tile: {tile_key}")
    
    # Parse tile key
    try:
        image_id, tile_x, tile_y, tile_size = parse_tile_key(tile_key)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"Image ID: {image_id}")
    print(f"Tile position: ({tile_x}, {tile_y})")
    print(f"Tile size: {tile_size}x{tile_size}")
    
    # Load tile mapping
    tile_mapping = load_tile_mapping(data_path)
    if tile_mapping is None:
        return
    
    if tile_key not in tile_mapping:
        print(f"Error: Tile {tile_key} not found in mapping")
        return
    
    building_ids = tile_mapping[tile_key]
    print(f"Found {len(building_ids)} buildings in this tile")
    
    # Load building data
    buildings_path = data_path / "buildings_transformed"
    buildings = load_building_data(buildings_path, building_ids)
    
    if not buildings:
        print("No building data loaded")
        return
    
    # Load tile image if available
    tile_image = None
    if show_image and RASTERIO_AVAILABLE:
        image_path = data_path / "COG" / f"{image_id}.tif"
        if image_path.exists():
            tile_image = load_tile_image(image_path, tile_x, tile_y, tile_size)
        else:
            print(f"Warning: Image file not found: {image_path}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Show tile image as background if available
    if tile_image is not None:
        ax.imshow(tile_image, extent=[0, tile_size, tile_size, 0], origin='upper')
        print(f"Loaded tile image with shape: {tile_image.shape}")
    else:
        # Create a white background
        ax.set_facecolor('white')
        print("No background image available")
    
    # Set up the plot - first collect all building coordinates to determine full extent
    all_x_coords = []
    all_y_coords = []
    
    for building_id, building in buildings.items():
        corners = get_building_corners_in_tile(building, image_id, tile_x, tile_y, tile_size)
        if corners:
            corners_array = np.array(corners)
            all_x_coords.extend(corners_array[:, 0])
            all_y_coords.extend(corners_array[:, 1])
    
    if all_x_coords and all_y_coords:
        # Add some padding around the buildings
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        padding = 50
        
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_max + padding, y_min - padding)  # Flip Y axis to match image coordinates
    else:
        # Fallback to tile bounds if no buildings found
        ax.set_xlim(0, tile_size)
        ax.set_ylim(tile_size, 0)  # Flip Y axis to match image coordinates
    
    ax.set_aspect('equal')
    ax.set_title(f'Tile: {tile_key}\nBuildings: {len(buildings)}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Draw buildings with more visible colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (building_id, building) in enumerate(buildings.items()):
        color = colors[i % len(colors)]
        
        # Get building corners in tile coordinates
        corners = get_building_corners_in_tile(building, image_id, tile_x, tile_y, tile_size)
        
        # Always draw buildings that are connected to this tile, regardless of corner count
        if len(corners) > 0:
            # Draw building edges only (no filled polygon)
            edges = get_building_edges_in_tile(building, image_id, tile_x, tile_y, tile_size)
            if edges:
                edge_lines = LineCollection(edges, colors=color, linewidths=2, alpha=0.9)
                ax.add_collection(edge_lines)
            
            # Draw small keypoints at corners
            for corner in corners:
                ax.plot(corner[0], corner[1], 'o', color=color, markersize=4, alpha=0.8)
            
            # Add building ID label at centroid with better visibility
            if len(corners) >= 2:
                centroid = np.mean(corners, axis=0)
                
                # Clip centroid to tile bounds for label positioning
                label_x = max(20, min(tile_size - 20, centroid[0]))
                label_y = max(20, min(tile_size - 20, centroid[1]))
                
                ax.text(label_x, label_y, str(building_id), 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               edgecolor=color, alpha=0.9))
            else:
                # For single corner buildings, place label near the corner
                corner = corners[0]
                label_x = max(20, min(tile_size - 20, corner[0] + 10))
                label_y = max(20, min(tile_size - 20, corner[1] + 10))
                
                ax.text(label_x, label_y, str(building_id), 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               edgecolor=color, alpha=0.9))
    
    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Add tile boundary with more prominent styling
    tile_boundary = patches.Rectangle((0, 0), tile_size, tile_size, 
                                    linewidth=4, edgecolor='red', 
                                    facecolor='none', linestyle='-')
    ax.add_patch(tile_boundary)
    
    # Add tile boundary label
    ax.text(tile_size/2, -20, 'Tile Boundary', ha='center', va='top', 
           fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def list_available_tiles(data_path, limit=10):
    """List available tiles in the dataset."""
    tile_mapping = load_tile_mapping(data_path)
    if tile_mapping is None:
        return
    
    tiles = list(tile_mapping.keys())
    print(f"Found {len(tiles)} tiles in the dataset")
    print(f"Showing first {min(limit, len(tiles))} tiles:")
    
    for i, tile_key in enumerate(tiles[:limit]):
        building_count = len(tile_mapping[tile_key])
        print(f"  {i+1:2d}. {tile_key} ({building_count} buildings)")


def main():
    parser = argparse.ArgumentParser(description="Visualize a tile with its buildings")
    parser.add_argument("--data_path", type=str, default="data/bergen", 
                       help="Path to data directory")
    parser.add_argument("--tile_key", type=str, 
                       help="Tile key to visualize (e.g., '14544_20_018_00761_3584_0_512')")
    parser.add_argument("--list_tiles", action="store_true", 
                       help="List available tiles")
    parser.add_argument("--no_image", action="store_true", 
                       help="Don't show background image")
    parser.add_argument("--save", type=str, 
                       help="Save visualization to file")
    parser.add_argument("--limit", type=int, default=10, 
                       help="Limit number of tiles to list")
    
    args = parser.parse_args()
    
    data_path = pathlib.Path(args.data_path)
    
    if args.list_tiles:
        list_available_tiles(data_path, args.limit)
        return
    
    if not args.tile_key:
        print("Error: Please provide a tile_key or use --list_tiles to see available tiles")
        print("Example: python visualize_tile.py --tile_key '14544_20_018_00761_3584_0_512'")
        return
    
    visualize_tile(args.tile_key, data_path, 
                  show_image=not args.no_image, 
                  save_path=args.save)


if __name__ == "__main__":
    main()