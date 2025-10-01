#!/usr/bin/env python3
"""
Script to visualize a specific building across all aerial images that contain it.
Adapted for RoofGraph project structure.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import rasterio
from rasterio.windows import Window


def get_building_bounding_box(corners, padding=200):
    """Calculate bounding box around building corners with padding."""
    if not corners or len(corners) == 0:
        raise ValueError("No valid corners provided")
    
    corners_array = np.array(corners)
    x_min, y_min = np.min(corners_array, axis=0)
    x_max, y_max = np.max(corners_array, axis=0)
    
    # Add padding
    x_min = max(0, int(x_min - padding))
    y_min = max(0, int(y_min - padding))
    width = int(x_max - x_min + 2 * padding)
    height = int(y_max - y_min + 2 * padding)
    
    return x_min, y_min, width, height


def load_building_by_id(buildings_path, building_id):
    """Load a specific building by ID from the buildings_transformed directory."""
    building_file = buildings_path / f"{building_id}.json"
    
    if not building_file.exists():
        raise FileNotFoundError(f"Building file not found: {building_file}")
    
    with open(building_file, 'r') as f:
        building = json.load(f)
    
    return building


def visualize_building_in_image(building, image_id, cog_path, ax, show_context=True, padding=500):
    """Visualize a building in a specific aerial image."""
    
    if image_id not in building["corners"]:
        print(f"Building {building['id']} not found in image {image_id}")
        return False
    
    corners = building["corners"][image_id]
    
    # Validate corners data
    if not corners or len(corners) == 0:
        print(f"Warning: Building {building['id']} has no valid corners data in image {image_id}")
        return False
    
    image_path = cog_path / f"{image_id}.tif"
    
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with rasterio.open(image_path) as src:
            if show_context:
                # Show building with context (larger area around building)
                try:
                    x_min, y_min, width, height = get_building_bounding_box(corners, padding=padding)
                except ValueError as e:
                    print(f"Warning: Cannot calculate bounding box for building in image {image_id}: {e}")
                    return False
                
                # Ensure we don't go outside image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                if x_min >= src.width or y_min >= src.height:
                    print(f"Warning: Computed window outside image bounds for {image_id}")
                    return False
                
                width = max(1, min(width, src.width - x_min))
                height = max(1, min(height, src.height - y_min))
                
                if width <= 0 or height <= 0:
                    print(f"Warning: Non-positive window size for {image_id}")
                    return False
                
                window = Window(x_min, y_min, width, height)
                image_data = src.read(window=window, boundless=True, fill_value=0)
                
                # Adjust corners relative to the window
                adjusted_corners = np.array(corners) - [x_min, y_min]
            else:
                # Show full image
                image_data = src.read()
                adjusted_corners = np.array(corners)
                x_min, y_min = 0, 0
        
        # Handle different band configurations
        if image_data.shape[0] == 1:  # Grayscale
            image = image_data[0]
            # Convert to RGB for consistent display
            image = np.stack([image, image, image], axis=-1)
        elif image_data.shape[0] == 3:  # RGB
            image = np.transpose(image_data, (1, 2, 0))
        elif image_data.shape[0] == 4:  # RGBA
            image = np.transpose(image_data, (1, 2, 0))[:, :, :3]
        else:
            # Take first 3 bands
            image = np.transpose(image_data[:3], (1, 2, 0))
        
        # Normalize image to [0, 1] range for display
        if image.dtype != np.uint8:
            image = np.clip(image / np.max(image) if np.max(image) > 0 else image, 0, 1)
        else:
            image = image.astype(np.float32) / 255.0
        
        # Ensure corners are within image bounds
        H, W = image.shape[0], image.shape[1]
        if adjusted_corners.size == 0:
            print(f"Warning: No valid corners for {image_id}")
            return False
        
        # Display the image
        ax.imshow(image, origin='upper')
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect('equal')
        
        # Plot building outline and structure
        if len(adjusted_corners) > 0:
            # Plot building edges from the graph structure
            if "edges" in building and building["edges"]:
                for corner_idx, connected_corners in enumerate(building["edges"]):
                    if corner_idx < len(adjusted_corners):
                        corner1 = adjusted_corners[corner_idx]
                        for connected_idx in connected_corners:
                            if connected_idx < len(adjusted_corners):
                                corner2 = adjusted_corners[connected_idx]
                                ax.plot([corner1[0], corner2[0]], [corner1[1], corner2[1]],
                                       color='red', linewidth=2, alpha=0.8)
            else:
                # If no edges, just draw the polygon outline
                if len(adjusted_corners) > 2:
                    polygon = plt.Polygon(adjusted_corners, fill=False, edgecolor='red', 
                                        linewidth=2, alpha=0.8)
                    ax.add_patch(polygon)
            
            # Plot corner points
            ax.scatter(adjusted_corners[:, 0], adjusted_corners[:, 1], 
                      color='yellow', s=30, zorder=5, alpha=0.9, edgecolors='red', linewidth=1)
            
            # Add building ID as text
            centroid = np.mean(adjusted_corners, axis=0)
            ax.text(centroid[0], centroid[1], f"ID: {building['id']}", 
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8),
                   ha='center', va='center')
        
        ax.set_title(f"Image: {image_id}\n{len(corners)} corners", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        return True
        
    except Exception as e:
        print(f"Error processing image {image_id}: {str(e)}")
        return False


def visualize_building_across_images(data_path, area, building_id, show_context=True, padding=500, save_output=False):
    """Visualize a building across all images that contain it."""
    
    data_path = Path(data_path)
    area_path = data_path / area
    cog_path = area_path / "COG"
    buildings_path = area_path / "buildings_transformed"
    
    if not cog_path.exists():
        print(f"COG directory not found: {cog_path}")
        return
    
    if not buildings_path.exists():
        print(f"Buildings directory not found: {buildings_path}")
        return
    
    # Load the specific building
    try:
        building = load_building_by_id(buildings_path, building_id)
        print(f"Loaded building {building_id}")
    except Exception as e:
        print(f"Error loading building {building_id}: {str(e)}")
        return
    
    # Get all images that contain this building
    image_ids = list(building["corners"].keys())
    
    if not image_ids:
        print(f"Building {building_id} is not found in any images")
        return
    
    print(f"Building {building_id} appears in {len(image_ids)} images: {image_ids}")
    
    # Calculate grid dimensions for subplots
    n_images = len(image_ids)
    if n_images == 1:
        rows, cols = 1, 1
    elif n_images <= 4:
        rows, cols = 2, 2
    elif n_images <= 6:
        rows, cols = 2, 3
    elif n_images <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, int(np.ceil(n_images / 4))
    
    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    fig.suptitle(f"Building {building_id} in {area} ({n_images} images)", 
                 fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if n_images == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Visualize the building in each image
    success_count = 0
    for i, image_id in enumerate(image_ids):
        if i < len(axes):
            success = visualize_building_in_image(building, image_id, cog_path, axes[i], 
                                                show_context, padding)
            if success:
                success_count += 1
        else:
            print(f"Too many images to display, skipping {image_id}")
    
    # Hide unused subplots
    for i in range(len(image_ids), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_path = data_path / f"{area}_building_{building_id}_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.show()
    
    print(f"Successfully visualized building {building_id} in {success_count}/{n_images} images")


def list_available_buildings(data_path, area, limit=20):
    """List all available building IDs in the specified area."""
    data_path = Path(data_path)
    area_path = data_path / area
    buildings_path = area_path / "buildings_transformed"
    
    if not buildings_path.exists():
        print(f"Buildings directory not found: {buildings_path}")
        return
    
    try:
        building_files = list(buildings_path.glob("*.json"))
        building_ids = [f.stem for f in building_files]
        
        print(f"Found {len(building_ids)} buildings in {area}:")
        
        for i, building_id in enumerate(sorted(building_ids)[:limit]):
            try:
                building = load_building_by_id(buildings_path, building_id)
                image_count = len(building.get('corners', {}))
                corner_count = sum(len(corners) for corners in building.get('corners', {}).values())
                print(f"  - {building_id} (appears in {image_count} images, {corner_count} total corners)")
            except Exception as e:
                print(f"  - {building_id} (error loading: {e})")
        
        if len(building_ids) > limit:
            print(f"  ... and {len(building_ids) - limit} more buildings")
            
    except Exception as e:
        print(f"Error loading buildings: {str(e)}")


def list_available_areas(data_path):
    """List all available areas in the data directory."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return
    
    areas = []
    for item in data_path.iterdir():
        if item.is_dir() and (item / "buildings_transformed").exists() and (item / "COG").exists():
            areas.append(item.name)
    
    if areas:
        print(f"Available areas: {', '.join(sorted(areas))}")
    else:
        print("No valid areas found (areas must have both 'buildings_transformed' and 'COG' directories)")
    
    return areas


def main():
    parser = argparse.ArgumentParser(description='Visualize a building across all aerial images')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to the data directory (default: data)')
    parser.add_argument('--area', type=str,
                       help='Area name (e.g., bergen, tromso, stavanger)')
    parser.add_argument('--building_id', type=str,
                       help='ID of the building to visualize')
    parser.add_argument('--list_areas', action='store_true',
                       help='List all available areas')
    parser.add_argument('--list_buildings', action='store_true',
                       help='List all available building IDs in the specified area')
    parser.add_argument('--show_full_image', action='store_true',
                       help='Show full image instead of cropped context around building')
    parser.add_argument('--padding', type=int, default=500,
                       help='Padding around building in pixels (default: 500)')
    parser.add_argument('--save', action='store_true',
                       help='Save the visualization as PNG file')
    parser.add_argument('--limit', type=int, default=20,
                       help='Limit number of buildings to list (default: 20)')
    
    args = parser.parse_args()
    
    if args.list_areas:
        list_available_areas(args.data_path)
        return
    
    if not args.area:
        print("Please provide an area with --area or use --list_areas to see available areas")
        return
        
    if args.list_buildings:
        list_available_buildings(args.data_path, args.area, args.limit)
        return
    
    if not args.building_id:
        print("Please provide a building ID with --building_id or use --list_buildings to see available IDs")
        return
    
    show_context = not args.show_full_image
    
    visualize_building_across_images(
        args.data_path, 
        args.area,
        args.building_id, 
        show_context=show_context,
        padding=args.padding,
        save_output=args.save
    )


if __name__ == "__main__":
    main()
