#!/usr/bin/env python3
"""
Simple script to visualize all buildings in a single image.
Shows all building edges overlaid on the image.
"""

import json
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def load_building(building_path, building_id):
    """Load a single building JSON file."""
    building_file = building_path / f"{building_id}.json"
    with open(building_file, 'r') as f:
        return json.load(f)

def get_all_building_ids(buildings_path):
    """Get list of all building IDs from the buildings_transformed folder."""
    building_files = list(buildings_path.glob("*.json"))
    return [f.stem for f in building_files]

def get_all_image_ids(cog_path):
    """Get list of all image IDs from the COG folder."""
    image_files = list(cog_path.glob("*.tif"))
    return [f.stem for f in image_files]

def visualize_image_buildings(image_id, data_path):
    """Visualize all buildings in a single image."""
    
    print(f"Visualizing all buildings in image: {image_id}")
    
    # Paths
    buildings_path = data_path / "buildings_transformed"
    cog_path = data_path / "COG"
    
    # Load the full image
    image_file = cog_path / f"{image_id}.tif"
    print(f"Loading image: {image_file}")
    
    with rasterio.open(image_file) as src:
        # Read the full image
        image = src.read()
        # Convert from (channels, height, width) to (height, width, channels)
        image = np.moveaxis(image, 0, -1)
        image_height, image_width = image.shape[:2]
        print(f"Image size: {image_width} x {image_height}")
    
    # Get all building IDs
    building_ids = get_all_building_ids(buildings_path)
    print(f"Found {len(building_ids)} total buildings")
    
    # Find buildings that appear in this image
    buildings_in_image = []
    for building_id in building_ids:
        building = load_building(buildings_path, building_id)
        if image_id in building.get("corners", {}):
            buildings_in_image.append(building)
    
    print(f"Found {len(buildings_in_image)} buildings in this image")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Show the image
    ax.imshow(image, extent=[0, image_width, image_height, 0])
    
    # Plot all buildings
    for i, building in enumerate(buildings_in_image):
        if image_id not in building.get("corners", {}):
            continue
        
        corners = np.array(building["corners"][image_id])
        if len(corners) == 0:
            continue
        
        # Use red for all buildings
        color = 'red'
        
        # Plot building edges
        if "edges" in building:
            for corner_idx, connected_corners in enumerate(building["edges"]):
                if corner_idx < len(corners):
                    corner1 = corners[corner_idx]
                    for connected_idx in connected_corners:
                        if connected_idx < len(corners):
                            corner2 = corners[connected_idx]
                            ax.plot([corner1[0], corner2[0]], [corner1[1], corner2[1]], 
                                   color=color, linewidth=1, alpha=0.8)
        
        # Plot building corners (smaller dots)
        ax.scatter(corners[:, 0], corners[:, 1], 
                  color=color, s=10, alpha=0.6)
    
    # Set up the plot
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)  # Flip y-axis to match image coordinates
    ax.set_title(f"All Buildings in Image: {image_id}\nTotal Buildings: {len(buildings_in_image)}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    data_path = pathlib.Path("data/bergen")
    
    # Paths
    buildings_path = data_path / "buildings_transformed"
    cog_path = data_path / "COG"
    
    # Get all image IDs
    image_ids = get_all_image_ids(cog_path)
    print(f"Found {len(image_ids)} images")
    
    if len(image_ids) == 0:
        print("No images found!")
        return
    
    # Choose a random image
    image_id = random.choice(image_ids)
    
    # Visualize all buildings in this image
    visualize_image_buildings(image_id, data_path)

if __name__ == "__main__":
    main()
