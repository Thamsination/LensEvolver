"""
Uniformity analysis functions for the Lens Optimizer.

This module provides functions to analyze light distribution uniformity
at the absorber exit surface.
"""

import numpy as np


def analyze_uniformity_grid(exit_positions, exit_intensities, grid_size=10, num_rays=None):
    """Analyze uniformity using a grid approach.
    
    Args:
        exit_positions: Nx3 array of positions
        exit_intensities: N array of intensities
        grid_size: Number of cells per dimension
        num_rays: Total number of rays simulated (for normalizing power distribution)
        
    Returns:
        dict with grid analysis results including irradiance in mW/cm²
    """
    if len(exit_positions) == 0:
        return {
            'grid_counts': np.zeros((grid_size, grid_size)),
            'grid_intensities': np.zeros((grid_size, grid_size)),
            'uniformity_index': 0,
            'hot_zones': [],
            'cold_zones': [],
            'mean_intensity': 0,
            'max_intensity': 0,
            'min_intensity': 0
        }
    
    # If num_rays not provided, use exit count as fallback
    if num_rays is None:
        num_rays = len(exit_positions)
    
    # Use X and Y for grid (assuming Z is optical axis)
    x = exit_positions[:, 0]
    y = exit_positions[:, 1]
    
    # Create grid bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add small padding to avoid edge issues
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    
    # Calculate cell area for irradiance calculation (mW/cm²)
    cell_area_mm2 = (x_range / grid_size) * (y_range / grid_size)
    cell_area_cm2 = cell_area_mm2 / 100.0  # Convert mm² to cm²
    
    # Compute grid indices
    x_idx = np.clip(((x - x_min) / x_range * (grid_size - 1)).astype(int), 0, grid_size - 1)
    y_idx = np.clip(((y - y_min) / y_range * (grid_size - 1)).astype(int), 0, grid_size - 1)
    
    # Accumulate into grid using vectorized numpy operations
    grid_counts = np.zeros((grid_size, grid_size))
    grid_intensities = np.zeros((grid_size, grid_size))
    
    np.add.at(grid_counts, (x_idx, y_idx), 1)
    np.add.at(grid_intensities, (x_idx, y_idx), exit_intensities)
    
    # Calculate uniformity index (1 = perfect, 0 = worst)
    non_empty = grid_counts > 0
    if np.sum(non_empty) > 1:
        values = grid_intensities[non_empty]
        mean_val = np.mean(values)
        if mean_val > 0:
            cv = np.std(values) / mean_val
            uniformity_index = 1 / (1 + cv)  # Transform to 0-1 range
        else:
            uniformity_index = 0
    else:
        uniformity_index = 0
    
    # Calculate irradiance statistics in mW/cm²
    mean_intensity = (np.mean(grid_intensities[non_empty]) / cell_area_cm2 / num_rays) if np.any(non_empty) else 0
    max_intensity = (np.max(grid_intensities[non_empty]) / cell_area_cm2 / num_rays) if np.any(non_empty) else 0
    min_intensity = (np.min(grid_intensities[non_empty]) / cell_area_cm2 / num_rays) if np.any(non_empty) else 0
    
    # Identify hot zones (above average) and cold zones (below average)
    hot_zones = []
    cold_zones = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_counts[i, j] > 0:
                cell_center_x = x_min + (i + 0.5) * x_range / grid_size
                cell_center_y = y_min + (j + 0.5) * y_range / grid_size
                
                if grid_intensities[i, j] > mean_intensity * 1.2:  # 20% above average
                    hot_zones.append({
                        'grid_pos': (i, j),
                        'world_pos': (cell_center_x, cell_center_y),
                        'intensity': grid_intensities[i, j],
                        'excess': grid_intensities[i, j] / mean_intensity - 1
                    })
                elif grid_intensities[i, j] < mean_intensity * 0.8:  # 20% below average
                    cold_zones.append({
                        'grid_pos': (i, j),
                        'world_pos': (cell_center_x, cell_center_y),
                        'intensity': grid_intensities[i, j],
                        'deficit': 1 - grid_intensities[i, j] / mean_intensity
                    })
    
    return {
        'grid_counts': grid_counts,
        'grid_intensities': grid_intensities,
        'uniformity_index': uniformity_index,
        'hot_zones': hot_zones,
        'cold_zones': cold_zones,
        'mean_intensity': mean_intensity,
        'max_intensity': max_intensity,
        'min_intensity': min_intensity,
        'bounds': (x_min, x_max, y_min, y_max)
    }


def calculate_exit_distribution(exit_positions, num_bins=10):
    """Calculate distribution of exit positions.
    
    Args:
        exit_positions: Nx3 array of exit positions
        num_bins: Number of histogram bins per dimension
        
    Returns:
        dict with distribution statistics
    """
    if len(exit_positions) == 0:
        return {
            'x_hist': np.zeros(num_bins),
            'y_hist': np.zeros(num_bins),
            'coverage': 0.0
        }
    
    x = exit_positions[:, 0]
    y = exit_positions[:, 1]
    
    x_hist, _ = np.histogram(x, bins=num_bins)
    y_hist, _ = np.histogram(y, bins=num_bins)
    
    # Coverage: fraction of bins with at least one exit
    coverage = np.sum(x_hist > 0) / num_bins
    
    return {
        'x_hist': x_hist,
        'y_hist': y_hist,
        'coverage': coverage
    }


def calculate_fitness(uniformity, efficiency, lens_entry_rate=1.0,
                      absorber_capture_rate=1.0,
                      exit_distribution_score=None,
                      efficiency_threshold=0.95, entry_threshold=0.98,
                      capture_threshold=0.95,
                      distribution_weight=0.0):
    """Calculate combined fitness score with priority ordering.
    
    Priority order:
    1. Lens entry rate must be >= entry_threshold (capture all rays from LED)
    2. Absorber capture rate must be >= capture_threshold (rays reach absorber, don't escape)
    3. Efficiency must be >= efficiency_threshold (rays exit absorber)
    4. Uniformity and exit distribution (even distribution at absorber exit)
    
    Fitness ranges:
    - 0.00 - 0.20: Lens not capturing enough rays from LED
    - 0.20 - 0.40: Rays escaping lens without hitting absorber
    - 0.40 - 0.60: Rays hit absorber but efficiency too low
    - 0.60 - 1.00: All thresholds met, optimizing uniformity and distribution
    
    Args:
        uniformity: Uniformity index (0.0 to 1.0, higher = more uniform)
        efficiency: Efficiency (0.0 to 1.0, higher = more rays reach absorber exit)
        lens_entry_rate: Fraction of rays hitting the lens (0.0 to 1.0)
        absorber_capture_rate: Fraction of lens rays reaching absorber (0.0 to 1.0)
        exit_distribution_score: Score for side vs top exit distribution (0.0 to 1.0)
        efficiency_threshold: Minimum required efficiency (default 0.95)
        entry_threshold: Minimum required lens entry rate (default 0.98)
        capture_threshold: Minimum required absorber capture rate (default 0.95)
        distribution_weight: Weight for exit distribution in final score (0.0 to 1.0)
                            0.0 = only uniformity, 1.0 = only distribution
        
    Returns:
        Fitness score (0.0 to 1.0)
    """
    # First priority: lens must capture rays from LED
    if lens_entry_rate < entry_threshold:
        # Heavily penalized - range 0.0 to 0.20
        return lens_entry_rate * 0.20
    
    # Second priority: rays must reach absorber (not escape lens)
    if absorber_capture_rate < capture_threshold:
        # Penalized - range 0.20 to 0.40
        return 0.20 + absorber_capture_rate * 0.20
    
    # Third priority: rays must exit absorber
    if efficiency < efficiency_threshold:
        # Penalized but better than escaping - range 0.40 to 0.60
        return 0.40 + efficiency * 0.20
    
    # All thresholds met: optimize uniformity and distribution - range 0.60 to 1.0
    # Blend uniformity and exit distribution based on weight
    if exit_distribution_score is not None and distribution_weight > 0:
        combined_quality = (
            (1.0 - distribution_weight) * uniformity + 
            distribution_weight * exit_distribution_score
        )
    else:
        combined_quality = uniformity
    
    return 0.60 + combined_quality * 0.40
