#!/usr/bin/env python3
"""
Simple script to visualize a random tile with buildings overlaid.
"""

import json
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

def load_building(building_path, building_id):
    """Load a single building JSON file."""
    building_file = building_path / f"{building_id}.json"
    with open(building_file, 'r') as f:
        return json.load(f)

def visualize_tile(tile_key, buildings_in_tile, data_path):
    """Visualize a single tile with its buildings."""
    
    # Parse tile key: "image_id_x_y_tile_size"
    parts = tile_key.split("_")
    image_id = "_".join(parts[:-3])  # Everything except last 3 parts
    x = int(parts[-3])
    y = int(parts[-2])
    tile_size = int(parts[-1])
    
    print(f"Visualizing tile: {tile_key}")
    print(f"Image: {image_id}")
    print(f"Position: ({x}, {y})")
    print(f"Size: {tile_size}x{tile_size}")
    print(f"Buildings in tile: {len(buildings_in_tile)}")
    
    # Paths
    buildings_path = data_path / "buildings_transformed"
    cog_path = data_path / "COG"
    
    # Load the image tile
    image_file = cog_path / f"{image_id}.tif"
    print(f"Loading image: {image_file}")
    
    with rasterio.open(image_file) as src:
        # Create window for this tile
        window = Window(x, y, tile_size, tile_size)
        
        # Read the tile
        tile_image = src.read(window=window, boundless=True, fill_value=0)
        # Convert from (channels, height, width) to (height, width, channels)
        tile_image = np.moveaxis(tile_image, 0, -1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show the image
    ax.imshow(tile_image, extent=[0, tile_size, tile_size, 0])
    
    # Plot buildings
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, building_id in enumerate(buildings_in_tile):
        building = load_building(buildings_path, building_id)
        
        if image_id not in building["corners"]:
            print(f"Warning: Building {building_id} has no corners for image {image_id}")
            continue
        
        corners = np.array(building["corners"][image_id])
        if len(corners) == 0:
            continue
        
        # Adjust corners relative to the tile position
        adjusted_corners = corners - [x, y]
        
        # Choose color for this building
        color = colors[i % len(colors)]
        
        # Plot building edges
        if "edges" in building:
            for corner_idx, connected_corners in enumerate(building["edges"]):
                if corner_idx < len(adjusted_corners):
                    corner1 = adjusted_corners[corner_idx]
                    for connected_idx in connected_corners:
                        if connected_idx < len(adjusted_corners):
                            corner2 = adjusted_corners[connected_idx]
                            ax.plot([corner1[0], corner2[0]], [corner1[1], corner2[1]], 
                                   color=color, linewidth=2, alpha=0.8)
        
        # Plot building corners
        ax.scatter(adjusted_corners[:, 0], adjusted_corners[:, 1], 
                  color=color, s=50, alpha=0.8, label=f'Building {building_id}')
    
    # Set up the plot
    ax.set_xlim(0, tile_size)
    ax.set_ylim(tile_size, 0)  # Flip y-axis to match image coordinates
    ax.set_title(f"Tile: {tile_key}\nBuildings: {len(buildings_in_tile)}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    data_path = pathlib.Path("data/tromso")
    tile_file = data_path / "tile_to_buildings.json"
    
    # Check if tile file exists
    if not tile_file.exists():
        print(f"Error: Tile file not found at {tile_file}")
        print("Please run simple_precompute_tiles.py first!")
        return
    
    # Load the tile-to-buildings mapping
    print(f"Loading tile mapping from: {tile_file}")
    with open(tile_file, 'r') as f:
        tile_to_buildings = json.load(f)
    
    print(f"Found {len(tile_to_buildings)} tiles")
    
    # Filter tiles that have buildings
    tiles_with_buildings = {k: v for k, v in tile_to_buildings.items() if len(v) > 0}
    print(f"Tiles with buildings: {len(tiles_with_buildings)}")
    
    if len(tiles_with_buildings) == 0:
        print("No tiles with buildings found!")
        return
    
    # Choose a random tile
    tile_key = random.choice(list(tiles_with_buildings.keys()))
    buildings_in_tile = tiles_with_buildings[tile_key]
    
    # Visualize it
    visualize_tile(tile_key, buildings_in_tile, data_path)

if __name__ == "__main__":
    main()
