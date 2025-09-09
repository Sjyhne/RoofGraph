#!/usr/bin/env python3
"""
Simple script to precompute which buildings are in each tile.
This creates a mapping: tile_id -> list of building_ids
"""

import json
import pathlib
import numpy as np
from rtree import index
from rasterio.windows import Window
import argparse


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

def create_spatial_index(buildings, image_id):
    """Create a spatial index for fast lookup of buildings in an image."""
    idx = index.Index()
    
    for i, building in enumerate(buildings):
        # Check if this building has corners for this image
        if image_id in building["corners"]:
            corners = building["corners"][image_id]
            if len(corners) == 0:
                continue  # Skip empty buildings
            
            # Get bounding box of the building
            corners_array = np.array(corners)
            min_x, min_y = np.min(corners_array, axis=0)
            max_x, max_y = np.max(corners_array, axis=0)
            
            # Add to spatial index
            idx.insert(i, (min_x, min_y, max_x, max_y))
    
    return idx

def line_intersects_rectangle(line_start, line_end, rect_x, rect_y, rect_width, rect_height):
    """Check if a line segment intersects with a rectangle using proper geometric intersection."""
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Check if line is completely outside rectangle (quick rejection)
    if max(x1, x2) < rect_x or min(x1, x2) > rect_x + rect_width:
        return False
    if max(y1, y2) < rect_y or min(y1, y2) > rect_y + rect_height:
        return False
    
    # Check if any endpoint is inside rectangle
    if (rect_x <= x1 < rect_x + rect_width and rect_y <= y1 < rect_y + rect_height):
        return True
    if (rect_x <= x2 < rect_x + rect_width and rect_y <= y2 < rect_y + rect_height):
        return True
    
    # Check if line crosses rectangle boundary using parametric intersection
    # Line equation: P = P1 + t * (P2 - P1), where 0 <= t <= 1
    dx = x2 - x1
    dy = y2 - y1
    
    # Avoid division by zero
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return False
    
    # Check intersection with rectangle edges
    # Left edge: x = rect_x
    if abs(dx) > 1e-10:
        t = (rect_x - x1) / dx
        if 0 <= t <= 1:
            y = y1 + t * dy
            if rect_y <= y < rect_y + rect_height:
                return True
    
    # Right edge: x = rect_x + rect_width
    if abs(dx) > 1e-10:
        t = (rect_x + rect_width - x1) / dx
        if 0 <= t <= 1:
            y = y1 + t * dy
            if rect_y <= y < rect_y + rect_height:
                return True
    
    # Top edge: y = rect_y
    if abs(dy) > 1e-10:
        t = (rect_y - y1) / dy
        if 0 <= t <= 1:
            x = x1 + t * dx
            if rect_x <= x < rect_x + rect_width:
                return True
    
    # Bottom edge: y = rect_y + rect_height
    if abs(dy) > 1e-10:
        t = (rect_y + rect_height - y1) / dy
        if 0 <= t <= 1:
            x = x1 + t * dx
            if rect_x <= x < rect_x + rect_width:
                return True
    
    return False

def building_intersects_tile(building, image_id, tile_x, tile_y, tile_size, min_corner_ratio=0.1):
    """Check if a building intersects with a specific tile: any corner inside OR any edge intersects."""
    if image_id not in building["corners"]:
        return False
    
    corners = np.array(building["corners"][image_id])
    if len(corners) == 0:
        return False
    
    # Check if any corner is inside the tile
    for corner in corners:
        if (tile_x <= corner[0] < tile_x + tile_size and 
            tile_y <= corner[1] < tile_y + tile_size):
            return True
    
    # Check if any edge intersects with the tile
    if "edges" in building:
        for corner_idx, connected_corners in enumerate(building["edges"]):
            if corner_idx < len(corners):
                corner1 = corners[corner_idx]
                for connected_idx in connected_corners:
                    if connected_idx < len(corners):
                        corner2 = corners[connected_idx]
                        if line_intersects_rectangle(corner1, corner2, tile_x, tile_y, tile_size, tile_size):
                            return True
    
    return False

def main(args):
    # Configuration
    data_path = pathlib.Path("data/" + args.data_path)
    tile_size = args.tile_size
    
    # Paths
    buildings_path = data_path / "buildings_transformed"
    cog_path = data_path / "COG"
    
    print(f"Loading buildings from: {buildings_path}")
    print(f"Loading images from: {cog_path}")
    
    # Step 1: Get all building IDs
    building_ids = get_all_building_ids(buildings_path)
    print(f"Found {len(building_ids)} buildings")
    
    # Step 2: Get all image IDs
    image_ids = get_all_image_ids(cog_path)
    print(f"Found {len(image_ids)} images")
    
    # Step 3: Create mapping from image_id to buildings that appear in that image
    image_to_buildings = {}
    for building_id in building_ids:
        building = load_building(buildings_path, building_id)
        
        # For each image this building appears in, add it to the mapping
        for image_id in building["corners"].keys():
            if image_id not in image_to_buildings:
                image_to_buildings[image_id] = []
            image_to_buildings[image_id].append(building_id)
    
    print(f"Created mapping for {len(image_to_buildings)} images")
    
    # Step 4: For each image, create tiles and find which buildings are in each tile
    tile_to_buildings = {}
    
    for image_id in image_ids:
        if image_id not in image_to_buildings:
            continue  # Skip images with no buildings
        
        print(f"Processing image: {image_id}")
        
        # Load all buildings for this image
        buildings = []
        for building_id in image_to_buildings[image_id]:
            building = load_building(buildings_path, building_id)
            buildings.append(building)
        
        # Create spatial index for fast lookup
        spatial_index = create_spatial_index(buildings, image_id)
        
        # Assume image size (you might want to read this from the actual image)
        # For now, let's use a reasonable default
        image_width = 4000  # You can read this from the actual image file
        image_height = 4000
        
        # Create tiles across the image
        for y in range(0, image_height, tile_size):
            for x in range(0, image_width, tile_size):
                # Create tile key
                tile_key = f"{image_id}_{x}_{y}_{tile_size}"
                
                # Find buildings that might intersect with this tile using spatial index
                bbox = (x, y, x + tile_size, y + tile_size)
                possible_buildings = list(spatial_index.intersection(bbox))
                
                # Check which buildings actually intersect
                intersecting_buildings = []
                for i in possible_buildings:
                    building = buildings[i]
                    if building_intersects_tile(building, image_id, x, y, tile_size, args.min_corner_ratio):
                        intersecting_buildings.append(building["id"])
                
                # Only keep tiles that have buildings
                if len(intersecting_buildings) > 0:
                    tile_to_buildings[tile_key] = intersecting_buildings
    
    # Step 5: Save the results
    output_file = data_path / "tile_to_buildings.json"
    with open(output_file, 'w') as f:
        json.dump(tile_to_buildings, f, indent=2)
    
    print(f"Saved {len(tile_to_buildings)} tiles to {output_file}")
    
    # Show some statistics
    total_buildings = sum(len(buildings) for buildings in tile_to_buildings.values())
    print(f"Total building-tile associations: {total_buildings}")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute tiles")
    parser.add_argument("--data_path", type=str, default="data/rana")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--min_corner_ratio", type=float, default=0.1, 
                       help="Minimum ratio of corners/edges that must intersect with tile")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
