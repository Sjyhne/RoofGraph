#!/usr/bin/env python3
"""
Dataset Summary Script for RoofGraph

This script generates comprehensive statistics about the RoofGraph dataset,
including metrics for each area and the complete dataset.
"""

import json
import pathlib
import argparse
import numpy as np
from collections import defaultdict
import sys

# Optional imports
try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. Image dimension analysis will be skipped.")


def get_image_dimensions(image_path):
    """Get dimensions of a GeoTIFF image."""
    if not RASTERIO_AVAILABLE:
        return None, None
    
    try:
        with rasterio.open(image_path) as src:
            return src.width, src.height
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return None, None


def load_building_data(buildings_path):
    """Load all building data from a directory."""
    buildings = {}
    building_files = list(buildings_path.glob("*.json"))
    
    for building_file in building_files:
        try:
            with open(building_file, 'r') as f:
                building_data = json.load(f)
                buildings[building_data['id']] = building_data
        except Exception as e:
            print(f"Warning: Could not load building {building_file}: {e}")
    
    return buildings


def analyze_building_complexity(building):
    """Analyze the complexity of a building."""
    metrics = {
        'num_corners': 0,
        'num_edges': 0,
        'num_images': 0,
        'has_utm_corners': 'utm_corners' in building,
        'is_pruned': building.get('pruned', False)
    }
    
    # Count corners in UTM coordinates
    if 'utm_corners' in building:
        metrics['num_corners'] = len(building['utm_corners'])
    
    # Count edges
    if 'edges' in building:
        metrics['num_edges'] = len(building['edges'])
    
    # Count images this building appears in
    if 'corners' in building:
        metrics['num_images'] = len(building['corners'])
    
    return metrics


def calculate_building_area(building):
    """Calculate approximate area of a building using UTM corners."""
    if 'utm_corners' not in building or len(building['utm_corners']) < 3:
        return None
    
    # Use shoelace formula for polygon area
    corners = np.array(building['utm_corners'])
    x = corners[:, 0]
    y = corners[:, 1]
    
    # Shoelace formula
    area = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))
    return area


def analyze_tile_distribution(tile_to_buildings):
    """Analyze the distribution of buildings across tiles."""
    building_counts = [len(buildings) for buildings in tile_to_buildings.values()]
    
    if not building_counts:
        return {
            'total_tiles': 0,
            'avg_buildings_per_tile': 0,
            'min_buildings_per_tile': 0,
            'max_buildings_per_tile': 0,
            'std_buildings_per_tile': 0,
            'tiles_with_1_building': 0,
            'tiles_with_multiple_buildings': 0
        }
    
    return {
        'total_tiles': len(tile_to_buildings),
        'avg_buildings_per_tile': np.mean(building_counts),
        'min_buildings_per_tile': np.min(building_counts),
        'max_buildings_per_tile': np.max(building_counts),
        'std_buildings_per_tile': np.std(building_counts),
        'tiles_with_1_building': sum(1 for count in building_counts if count == 1),
        'tiles_with_multiple_buildings': sum(1 for count in building_counts if count > 1)
    }


def analyze_area(data_path, area_name):
    """Analyze a single area and return comprehensive statistics."""
    print(f"Analyzing area: {area_name}")
    
    # Paths
    buildings_path = data_path / "buildings_transformed"
    cog_path = data_path / "COG"
    tile_mapping_path = data_path / "tile_to_buildings.json"
    
    # Check if paths exist
    if not buildings_path.exists():
        print(f"Warning: Buildings path does not exist: {buildings_path}")
        return None
    
    if not cog_path.exists():
        print(f"Warning: COG path does not exist: {cog_path}")
        return None
    
    # Load building data
    buildings = load_building_data(buildings_path)
    print(f"  Loaded {len(buildings)} buildings")
    
    # Load tile mapping if it exists
    tile_to_buildings = {}
    if tile_mapping_path.exists():
        try:
            with open(tile_mapping_path, 'r') as f:
                tile_to_buildings = json.load(f)
            print(f"  Loaded {len(tile_to_buildings)} tiles")
        except Exception as e:
            print(f"Warning: Could not load tile mapping: {e}")
    
    # Get image files
    image_files = list(cog_path.glob("*.tif"))
    print(f"  Found {len(image_files)} images")
    
    # Analyze buildings
    building_metrics = []
    building_areas = []
    image_coverage = defaultdict(int)
    
    for building_id, building in buildings.items():
        metrics = analyze_building_complexity(building)
        building_metrics.append(metrics)
        
        # Calculate area
        area = calculate_building_area(building)
        if area is not None:
            building_areas.append(area)
        
        # Count images this building appears in
        if 'corners' in building:
            for image_id in building['corners'].keys():
                image_coverage[image_id] += 1
    
    # Calculate building statistics
    if building_metrics:
        avg_corners = np.mean([m['num_corners'] for m in building_metrics])
        avg_edges = np.mean([m['num_edges'] for m in building_metrics])
        avg_images_per_building = np.mean([m['num_images'] for m in building_metrics])
        pruned_buildings = sum(1 for m in building_metrics if m['is_pruned'])
    else:
        avg_corners = avg_edges = avg_images_per_building = 0
        pruned_buildings = 0
    
    # Calculate area statistics
    if building_areas:
        avg_area = np.mean(building_areas)
        median_area = np.median(building_areas)
        min_area = np.min(building_areas)
        max_area = np.max(building_areas)
        total_area = np.sum(building_areas)
    else:
        avg_area = median_area = min_area = max_area = total_area = 0
    
    # Analyze image dimensions
    image_dimensions = []
    for image_file in image_files:
        width, height = get_image_dimensions(image_file)
        if width is not None and height is not None:
            image_dimensions.append((width, height))
    
    if image_dimensions:
        avg_width = np.mean([d[0] for d in image_dimensions])
        avg_height = np.mean([d[1] for d in image_dimensions])
        total_pixels = sum(d[0] * d[1] for d in image_dimensions)
    else:
        avg_width = avg_height = total_pixels = 0
    
    # Analyze tile distribution
    tile_stats = analyze_tile_distribution(tile_to_buildings)
    
    # Calculate coverage statistics
    images_with_buildings = len(image_coverage)
    coverage_percentage = (images_with_buildings / len(image_files) * 100) if image_files else 0
    
    # Calculate building-image density (buildings per image)
    total_building_image_associations = sum(image_coverage.values())
    building_image_density = (total_building_image_associations / len(image_files)) if image_files else 0
    
    return {
        'area_name': area_name,
        'buildings': {
            'total_count': len(buildings),
            'avg_corners': avg_corners,
            'avg_edges': avg_edges,
            'avg_images_per_building': avg_images_per_building,
            'pruned_count': pruned_buildings,
            'pruned_percentage': (pruned_buildings / len(buildings) * 100) if buildings else 0
        },
        'areas': {
            'avg_area': avg_area,
            'median_area': median_area,
            'min_area': min_area,
            'max_area': max_area,
            'total_area': total_area,
            'count_with_area': len(building_areas)
        },
        'images': {
            'total_count': len(image_files),
            'avg_width': avg_width,
            'avg_height': avg_height,
            'total_pixels': total_pixels,
            'images_with_buildings': images_with_buildings,
            'coverage_percentage': coverage_percentage,
            'building_image_density': building_image_density
        },
        'tiles': tile_stats
    }


def print_summary(summary, detailed=False):
    """Print a formatted summary of the statistics."""
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY: {summary['area_name'].upper()}")
    print(f"{'='*60}")
    
    # Buildings
    buildings = summary['buildings']
    print(f"\nBUILDINGS:")
    print(f"  Total count: {buildings['total_count']:,}")
    print(f"  Average corners per building: {buildings['avg_corners']:.2f}")
    print(f"  Average edges per building: {buildings['avg_edges']:.2f}")
    print(f"  Average images per building: {buildings['avg_images_per_building']:.2f}")
    print(f"  Pruned buildings: {buildings['pruned_count']:,} ({buildings['pruned_percentage']:.1f}%)")
    
    # Areas
    areas = summary['areas']
    print(f"\nBUILDING AREAS:")
    print(f"  Buildings with area data: {areas['count_with_area']:,}")
    if areas['count_with_area'] > 0:
        print(f"  Average area: {areas['avg_area']:.2f} m²")
        print(f"  Median area: {areas['median_area']:.2f} m²")
        print(f"  Min area: {areas['min_area']:.2f} m²")
        print(f"  Max area: {areas['max_area']:.2f} m²")
        print(f"  Total area: {areas['total_area']:.2f} m²")
    
    # Images
    images = summary['images']
    print(f"\nIMAGES:")
    print(f"  Total count: {images['total_count']:,}")
    print(f"  Average dimensions: {images['avg_width']:.0f} x {images['avg_height']:.0f} pixels")
    print(f"  Total pixels: {images['total_pixels']:,}")
    print(f"  Images with buildings: {images['images_with_buildings']:,} ({images['coverage_percentage']:.1f}%)")
    print(f"  Building-image density: {images['building_image_density']:.1f} buildings per image")
    
    # Tiles
    tiles = summary['tiles']
    print(f"\nTILES:")
    print(f"  Total tiles: {tiles['total_tiles']:,}")
    if tiles['total_tiles'] > 0:
        print(f"  Average buildings per tile: {tiles['avg_buildings_per_tile']:.2f}")
        print(f"  Min buildings per tile: {tiles['min_buildings_per_tile']}")
        print(f"  Max buildings per tile: {tiles['max_buildings_per_tile']}")
        print(f"  Std buildings per tile: {tiles['std_buildings_per_tile']:.2f}")
        print(f"  Tiles with 1 building: {tiles['tiles_with_1_building']:,}")
        print(f"  Tiles with multiple buildings: {tiles['tiles_with_multiple_buildings']:,}")


def calculate_overall_statistics(area_summaries):
    """Calculate comprehensive overall statistics across all areas."""
    valid_summaries = [s for s in area_summaries if s]
    if not valid_summaries:
        return None
    
    # Basic counts
    total_areas = len(valid_summaries)
    total_buildings = sum(s['buildings']['total_count'] for s in valid_summaries)
    total_images = sum(s['images']['total_count'] for s in valid_summaries)
    total_tiles = sum(s['tiles']['total_tiles'] for s in valid_summaries)
    total_area = sum(s['areas']['total_area'] for s in valid_summaries)
    
    # Building complexity statistics
    all_avg_corners = [s['buildings']['avg_corners'] for s in valid_summaries]
    all_avg_edges = [s['buildings']['avg_edges'] for s in valid_summaries]
    all_avg_images_per_building = [s['buildings']['avg_images_per_building'] for s in valid_summaries]
    all_pruned_percentages = [s['buildings']['pruned_percentage'] for s in valid_summaries]
    
    # Area statistics
    all_avg_areas = [s['areas']['avg_area'] for s in valid_summaries if s['areas']['count_with_area'] > 0]
    all_median_areas = [s['areas']['median_area'] for s in valid_summaries if s['areas']['count_with_area'] > 0]
    all_max_areas = [s['areas']['max_area'] for s in valid_summaries if s['areas']['count_with_area'] > 0]
    
    # Tile statistics
    all_avg_buildings_per_tile = [s['tiles']['avg_buildings_per_tile'] for s in valid_summaries if s['tiles']['total_tiles'] > 0]
    all_max_buildings_per_tile = [s['tiles']['max_buildings_per_tile'] for s in valid_summaries if s['tiles']['total_tiles'] > 0]
    all_std_buildings_per_tile = [s['tiles']['std_buildings_per_tile'] for s in valid_summaries if s['tiles']['total_tiles'] > 0]
    
    # Coverage statistics
    all_coverage_percentages = [s['images']['coverage_percentage'] for s in valid_summaries]
    all_building_image_densities = [s['images']['building_image_density'] for s in valid_summaries]
    
    # Calculate weighted averages
    weighted_avg_buildings_per_tile = 0
    if total_tiles > 0:
        weighted_avg_buildings_per_tile = sum(s['tiles']['avg_buildings_per_tile'] * s['tiles']['total_tiles'] 
                                            for s in valid_summaries) / total_tiles
    
    # Calculate distributions
    buildings_per_area = [s['buildings']['total_count'] for s in valid_summaries]
    images_per_area = [s['images']['total_count'] for s in valid_summaries]
    tiles_per_area = [s['tiles']['total_tiles'] for s in valid_summaries]
    
    return {
        'basic_counts': {
            'total_areas': total_areas,
            'total_buildings': total_buildings,
            'total_images': total_images,
            'total_tiles': total_tiles,
            'total_area': total_area
        },
        'building_complexity': {
            'avg_corners_min': min(all_avg_corners) if all_avg_corners else 0,
            'avg_corners_max': max(all_avg_corners) if all_avg_corners else 0,
            'avg_corners_mean': np.mean(all_avg_corners) if all_avg_corners else 0,
            'avg_edges_min': min(all_avg_edges) if all_avg_edges else 0,
            'avg_edges_max': max(all_avg_edges) if all_avg_edges else 0,
            'avg_edges_mean': np.mean(all_avg_edges) if all_avg_edges else 0,
            'avg_images_per_building_min': min(all_avg_images_per_building) if all_avg_images_per_building else 0,
            'avg_images_per_building_max': max(all_avg_images_per_building) if all_avg_images_per_building else 0,
            'avg_images_per_building_mean': np.mean(all_avg_images_per_building) if all_avg_images_per_building else 0,
            'pruned_percentage_min': min(all_pruned_percentages) if all_pruned_percentages else 0,
            'pruned_percentage_max': max(all_pruned_percentages) if all_pruned_percentages else 0,
            'pruned_percentage_mean': np.mean(all_pruned_percentages) if all_pruned_percentages else 0
        },
        'area_statistics': {
            'avg_area_min': min(all_avg_areas) if all_avg_areas else 0,
            'avg_area_max': max(all_avg_areas) if all_avg_areas else 0,
            'avg_area_mean': np.mean(all_avg_areas) if all_avg_areas else 0,
            'median_area_min': min(all_median_areas) if all_median_areas else 0,
            'median_area_max': max(all_median_areas) if all_median_areas else 0,
            'median_area_mean': np.mean(all_median_areas) if all_median_areas else 0,
            'max_area_min': min(all_max_areas) if all_max_areas else 0,
            'max_area_max': max(all_max_areas) if all_max_areas else 0,
            'max_area_mean': np.mean(all_max_areas) if all_max_areas else 0
        },
        'tile_statistics': {
            'weighted_avg_buildings_per_tile': weighted_avg_buildings_per_tile,
            'avg_buildings_per_tile_min': min(all_avg_buildings_per_tile) if all_avg_buildings_per_tile else 0,
            'avg_buildings_per_tile_max': max(all_avg_buildings_per_tile) if all_avg_buildings_per_tile else 0,
            'avg_buildings_per_tile_mean': np.mean(all_avg_buildings_per_tile) if all_avg_buildings_per_tile else 0,
            'max_buildings_per_tile_min': min(all_max_buildings_per_tile) if all_max_buildings_per_tile else 0,
            'max_buildings_per_tile_max': max(all_max_buildings_per_tile) if all_max_buildings_per_tile else 0,
            'max_buildings_per_tile_mean': np.mean(all_max_buildings_per_tile) if all_max_buildings_per_tile else 0,
            'std_buildings_per_tile_min': min(all_std_buildings_per_tile) if all_std_buildings_per_tile else 0,
            'std_buildings_per_tile_max': max(all_std_buildings_per_tile) if all_std_buildings_per_tile else 0,
            'std_buildings_per_tile_mean': np.mean(all_std_buildings_per_tile) if all_std_buildings_per_tile else 0
        },
        'coverage_statistics': {
            'coverage_min': min(all_coverage_percentages) if all_coverage_percentages else 0,
            'coverage_max': max(all_coverage_percentages) if all_coverage_percentages else 0,
            'coverage_mean': np.mean(all_coverage_percentages) if all_coverage_percentages else 0,
            'building_image_density_min': min(all_building_image_densities) if all_building_image_densities else 0,
            'building_image_density_max': max(all_building_image_densities) if all_building_image_densities else 0,
            'building_image_density_mean': np.mean(all_building_image_densities) if all_building_image_densities else 0
        },
        'distribution_statistics': {
            'buildings_per_area_min': min(buildings_per_area) if buildings_per_area else 0,
            'buildings_per_area_max': max(buildings_per_area) if buildings_per_area else 0,
            'buildings_per_area_mean': np.mean(buildings_per_area) if buildings_per_area else 0,
            'buildings_per_area_std': np.std(buildings_per_area) if buildings_per_area else 0,
            'images_per_area_min': min(images_per_area) if images_per_area else 0,
            'images_per_area_max': max(images_per_area) if images_per_area else 0,
            'images_per_area_mean': np.mean(images_per_area) if images_per_area else 0,
            'images_per_area_std': np.std(images_per_area) if images_per_area else 0,
            'tiles_per_area_min': min(tiles_per_area) if tiles_per_area else 0,
            'tiles_per_area_max': max(tiles_per_area) if tiles_per_area else 0,
            'tiles_per_area_mean': np.mean(tiles_per_area) if tiles_per_area else 0,
            'tiles_per_area_std': np.std(tiles_per_area) if tiles_per_area else 0
        }
    }


def print_overall_summary(area_summaries):
    """Print overall dataset summary."""
    print(f"\n{'='*80}")
    print(f"OVERALL DATASET SUMMARY")
    print(f"{'='*80}")
    
    # Calculate comprehensive statistics
    overall_stats = calculate_overall_statistics(area_summaries)
    if not overall_stats:
        print("No valid area summaries found.")
        return
    
    basic = overall_stats['basic_counts']
    building = overall_stats['building_complexity']
    areas = overall_stats['area_statistics']
    tiles = overall_stats['tile_statistics']
    coverage = overall_stats['coverage_statistics']
    distribution = overall_stats['distribution_statistics']
    
    # Basic counts
    print(f"\nBASIC COUNTS:")
    print(f"  Total areas: {basic['total_areas']}")
    print(f"  Total buildings: {basic['total_buildings']:,}")
    print(f"  Total images: {basic['total_images']:,}")
    print(f"  Total tiles: {basic['total_tiles']:,}")
    print(f"  Total building area: {basic['total_area']:.2f} m²")
    
    # Building complexity
    print(f"\nBUILDING COMPLEXITY (across areas):")
    print(f"  Average corners per building:")
    print(f"    Range: {building['avg_corners_min']:.1f} - {building['avg_corners_max']:.1f}")
    print(f"    Mean across areas: {building['avg_corners_mean']:.1f}")
    print(f"  Average edges per building:")
    print(f"    Range: {building['avg_edges_min']:.1f} - {building['avg_edges_max']:.1f}")
    print(f"    Mean across areas: {building['avg_edges_mean']:.1f}")
    print(f"  Average images per building:")
    print(f"    Range: {building['avg_images_per_building_min']:.1f} - {building['avg_images_per_building_max']:.1f}")
    print(f"    Mean across areas: {building['avg_images_per_building_mean']:.1f}")
    print(f"  Pruned buildings percentage:")
    print(f"    Range: {building['pruned_percentage_min']:.1f}% - {building['pruned_percentage_max']:.1f}%")
    print(f"    Mean across areas: {building['pruned_percentage_mean']:.1f}%")
    
    # Area statistics
    print(f"\nBUILDING AREA STATISTICS (across areas):")
    print(f"  Average building area:")
    print(f"    Range: {areas['avg_area_min']:.1f} - {areas['avg_area_max']:.1f} m²")
    print(f"    Mean across areas: {areas['avg_area_mean']:.1f} m²")
    print(f"  Median building area:")
    print(f"    Range: {areas['median_area_min']:.1f} - {areas['median_area_max']:.1f} m²")
    print(f"    Mean across areas: {areas['median_area_mean']:.1f} m²")
    print(f"  Largest building area:")
    print(f"    Range: {areas['max_area_min']:.1f} - {areas['max_area_max']:.1f} m²")
    print(f"    Mean across areas: {areas['max_area_mean']:.1f} m²")
    
    # Tile statistics
    print(f"\nTILE STATISTICS (across areas):")
    print(f"  Weighted average buildings per tile: {tiles['weighted_avg_buildings_per_tile']:.2f}")
    print(f"  Average buildings per tile (per area):")
    print(f"    Range: {tiles['avg_buildings_per_tile_min']:.2f} - {tiles['avg_buildings_per_tile_max']:.2f}")
    print(f"    Mean across areas: {tiles['avg_buildings_per_tile_mean']:.2f}")
    print(f"  Maximum buildings per tile (per area):")
    print(f"    Range: {tiles['max_buildings_per_tile_min']:.0f} - {tiles['max_buildings_per_tile_max']:.0f}")
    print(f"    Mean across areas: {tiles['max_buildings_per_tile_mean']:.1f}")
    print(f"  Standard deviation of buildings per tile (per area):")
    print(f"    Range: {tiles['std_buildings_per_tile_min']:.2f} - {tiles['std_buildings_per_tile_max']:.2f}")
    print(f"    Mean across areas: {tiles['std_buildings_per_tile_mean']:.2f}")
    
    # Coverage statistics
    print(f"\nIMAGE COVERAGE (across areas):")
    print(f"  Building coverage percentage:")
    print(f"    Range: {coverage['coverage_min']:.1f}% - {coverage['coverage_max']:.1f}%")
    print(f"    Mean across areas: {coverage['coverage_mean']:.1f}%")
    print(f"  Building-image density (buildings per image):")
    print(f"    Range: {coverage['building_image_density_min']:.1f} - {coverage['building_image_density_max']:.1f}")
    print(f"    Mean across areas: {coverage['building_image_density_mean']:.1f}")
    
    # Distribution statistics
    print(f"\nDISTRIBUTION ACROSS AREAS:")
    print(f"  Buildings per area:")
    print(f"    Range: {distribution['buildings_per_area_min']:,} - {distribution['buildings_per_area_max']:,}")
    print(f"    Mean: {distribution['buildings_per_area_mean']:.0f} ± {distribution['buildings_per_area_std']:.0f}")
    print(f"  Images per area:")
    print(f"    Range: {distribution['images_per_area_min']:,} - {distribution['images_per_area_max']:,}")
    print(f"    Mean: {distribution['images_per_area_mean']:.0f} ± {distribution['images_per_area_std']:.0f}")
    print(f"  Tiles per area:")
    print(f"    Range: {distribution['tiles_per_area_min']:,} - {distribution['tiles_per_area_max']:,}")
    print(f"    Mean: {distribution['tiles_per_area_mean']:.0f} ± {distribution['tiles_per_area_std']:.0f}")
    
    # Per-area breakdown table
    print(f"\nPER-AREA BREAKDOWN:")
    print(f"{'Area':<15} {'Buildings':<12} {'Images':<10} {'Tiles':<10} {'Coverage':<10}")
    print(f"{'-'*60}")
    
    for summary in area_summaries:
        if summary:
            coverage = summary['images']['coverage_percentage']
            print(f"{summary['area_name']:<15} {summary['buildings']['total_count']:<12,} "
                  f"{summary['images']['total_count']:<10,} {summary['tiles']['total_tiles']:<10,} "
                  f"{coverage:<9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset summary statistics")
    parser.add_argument("--data_root", type=str, default="data", 
                       help="Root directory containing area subdirectories")
    parser.add_argument("--areas", nargs="+", 
                       default=["bergen", "kristiansand", "rana", "sandvika", "stavanger", "tromso"],
                       help="List of areas to analyze")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed statistics for each area")
    parser.add_argument("--output", type=str, 
                       help="Output file to save summary (JSON format)")
    
    args = parser.parse_args()
    
    data_root = pathlib.Path(args.data_root)
    area_summaries = []
    
    print("RoofGraph Dataset Summary")
    print("=" * 40)
    
    for area in args.areas:
        area_path = data_root / area
        if not area_path.exists():
            print(f"Warning: Area directory does not exist: {area_path}")
            continue
        
        summary = analyze_area(area_path, area)
        if summary:
            area_summaries.append(summary)
            if args.detailed:
                print_summary(summary, detailed=True)
    
    # Print overall summary
    print_overall_summary(area_summaries)
    
    # Save to file if requested
    if args.output:
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Calculate comprehensive overall statistics
        overall_stats = calculate_overall_statistics(area_summaries)
        
        output_data = {
            'overall': overall_stats,
            'areas': area_summaries
        }
        
        # Convert all numpy types to Python native types
        output_data = convert_numpy_types(output_data)
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()
