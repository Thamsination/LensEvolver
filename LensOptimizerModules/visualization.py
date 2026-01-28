"""
Visualization functions for the Lens Optimizer.

This module provides functions to create FreeCAD visualization objects
for optimization results, ray paths, and analysis data.
"""

from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import FreeCAD
import Part
import Mesh

from .data_classes import CenterlinePoint


def visualize_centerline(doc, parent_group, centerline_points: List[CenterlinePoint], 
                         name_suffix=""):
    """Create visualization of centerline in FreeCAD.
    
    Args:
        doc: FreeCAD document
        parent_group: Parent group for visualization objects
        centerline_points: List of CenterlinePoint objects
        name_suffix: Optional suffix for object names
        
    Returns:
        Created visualization object or None
    """
    if len(centerline_points) < 2:
        return None
    
    try:
        # Create spline through centerline points
        points = [cp.position for cp in centerline_points]
        
        try:
            curve = Part.BSplineCurve()
            curve.interpolate(points)
            edge = curve.toShape()
        except:
            # Fallback: make line segments
            edges = []
            for i in range(len(points) - 1):
                edges.append(Part.makeLine(points[i], points[i+1]))
            edge = Part.Wire(edges)
        
        obj = doc.addObject("Part::Feature", f"Centerline{name_suffix}")
        obj.Shape = edge
        obj.ViewObject.LineColor = (1.0, 1.0, 0.0)  # Yellow
        obj.ViewObject.LineWidth = 3.0
        parent_group.addObject(obj)
        
        return obj
        
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Centerline visualization failed: {e}\n")
        return None


def visualize_profiles(doc, parent_group, profiles, centerline_points, name_suffix=""):
    """Create visualization of profile shapes along centerline.
    
    Args:
        doc: FreeCAD document
        parent_group: Parent group for visualization objects
        profiles: List of ProfileParams
        centerline_points: List of CenterlinePoint objects
        name_suffix: Optional suffix for object names
        
    Returns:
        Created visualization group or None
    """
    try:
        from .profile_creation import interpolate_centerline_at_t, create_polygon_profile
        
        profile_group = doc.addObject("App::DocumentObjectGroup", f"Profiles{name_suffix}")
        parent_group.addObject(profile_group)
        
        for i, profile in enumerate(profiles):
            center, tangent = interpolate_centerline_at_t(centerline_points, profile.z_position)
            wire = create_polygon_profile(profile, center, tangent)
            
            obj = doc.addObject("Part::Feature", f"Profile_{i+1}")
            obj.Shape = wire
            obj.ViewObject.LineColor = (0.0, 1.0, 0.0)  # Green
            obj.ViewObject.LineWidth = 2.0
            profile_group.addObject(obj)
        
        return profile_group
        
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Profile visualization failed: {e}\n")
        return None


def create_heatmap_visualization(doc, parent_group, exit_positions, exit_intensities,
                                  grid_analysis, iteration=0, num_color_bins=8, num_rays=None):
    """Create a heatmap visualization of exit intensity distribution with irradiance-based coloring.
    
    Points are colored by local irradiance (mW/cm²) and labeled with their irradiance range.
    
    Args:
        doc: FreeCAD document
        parent_group: Parent group for visualization
        exit_positions: Nx3 array of exit positions
        exit_intensities: N array of intensities (raw ray intensities)
        grid_analysis: Results from analyze_uniformity_grid
        iteration: Iteration number for naming
        num_color_bins: Number of color bins for intensity gradient
        num_rays: Total number of rays simulated (for irradiance normalization)
        
    Returns:
        Created heatmap group or None
    """
    if len(exit_positions) == 0:
        return None
    
    try:
        from Part import Compound, Vertex
        
        # Sample points for performance (max 5000 points)
        max_points = 5000
        n_points = len(exit_positions)
        if n_points > max_points:
            indices = np.random.choice(n_points, max_points, replace=False)
            positions = np.array(exit_positions)[indices]
            intensities = np.array(exit_intensities)[indices]
            sampling_ratio = n_points / max_points
        else:
            positions = np.array(exit_positions)
            intensities = np.array(exit_intensities)
            sampling_ratio = 1.0
        
        # Use num_rays for normalization (fallback to exit count if not provided)
        if num_rays is None:
            num_rays = n_points
        
        # Calculate local irradiance using grid-based approach (matches original)
        x = positions[:, 0]
        y = positions[:, 1]
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = max(x_max - x_min, 0.1)
        y_range = max(y_max - y_min, 0.1)
        
        # Grid for irradiance calculation
        grid_size = 20
        cell_width_mm = x_range / grid_size
        cell_height_mm = y_range / grid_size
        cell_area_cm2 = (cell_width_mm * cell_height_mm) / 100.0  # mm² to cm²
        
        # Compute grid indices for each point
        x_idx = np.clip(((x - x_min) / x_range * (grid_size - 1)).astype(int), 0, grid_size - 1)
        y_idx = np.clip(((y - y_min) / y_range * (grid_size - 1)).astype(int), 0, grid_size - 1)
        
        # Accumulate intensities into grid
        grid_intensity = np.zeros((grid_size, grid_size))
        np.add.at(grid_intensity, (x_idx, y_idx), intensities)
        
        # Convert to irradiance (mW/cm²) - compensate for sampling
        grid_irradiance = grid_intensity * sampling_ratio / max(1, num_rays) / max(0.01, cell_area_cm2)
        
        # Look up irradiance for each point
        local_irradiance = grid_irradiance[x_idx, y_idx]
        
        irr_min = np.min(local_irradiance) if len(local_irradiance) > 0 else 0
        irr_max = np.max(local_irradiance) if len(local_irradiance) > 0 else 1
        irr_range = irr_max - irr_min
        
        if irr_range == 0:
            irr_range = 1  # Avoid division by zero
        
        # Create heatmap group with mean irradiance in name
        mean_irr = grid_analysis.get('mean_intensity', 0)
        heatmap_group = doc.addObject("App::DocumentObjectGroup", f"Heatmap_{mean_irr:.1f}mWcm2")
        parent_group.addObject(heatmap_group)
        
        def get_heatmap_color(normalized_value):
            """Get color based on normalized irradiance (0-1).
            Blue (cold) -> Cyan -> Green -> Yellow -> Red (hot)
            """
            t = max(0.0, min(1.0, normalized_value))
            
            if t < 0.25:
                # Blue to Cyan
                r, g, b = 0.0, t * 4, 1.0
            elif t < 0.5:
                # Cyan to Green
                r, g, b = 0.0, 1.0, 1.0 - (t - 0.25) * 4
            elif t < 0.75:
                # Green to Yellow
                r, g, b = (t - 0.5) * 4, 1.0, 0.0
            else:
                # Yellow to Red
                r, g, b = 1.0, 1.0 - (t - 0.75) * 4, 0.0
            
            return (r, g, b)
        
        # Bin points by irradiance and create colored compounds
        for bin_idx in range(num_color_bins):
            bin_irr_min = irr_min + (bin_idx / num_color_bins) * irr_range
            bin_irr_max = irr_min + ((bin_idx + 1) / num_color_bins) * irr_range
            
            # Find points in this bin
            if bin_idx == num_color_bins - 1:
                # Include max value in last bin
                mask = (local_irradiance >= bin_irr_min) & (local_irradiance <= bin_irr_max)
            else:
                mask = (local_irradiance >= bin_irr_min) & (local_irradiance < bin_irr_max)
            
            bin_positions = positions[mask]
            bin_irradiances = local_irradiance[mask]
            
            if len(bin_positions) == 0:
                continue
            
            # Create vertices for this bin
            vertices = [Vertex(FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))) 
                       for p in bin_positions]
            
            if vertices:
                compound = Compound(vertices)
                
                # Get color for this bin
                normalized_irr = (bin_idx + 0.5) / num_color_bins
                color = get_heatmap_color(normalized_irr)
                
                # Label with average irradiance in mW/cm² (like original)
                avg_irradiance = np.mean(bin_irradiances)
                obj_name = f"Irradiance_{avg_irradiance:.1f}mWcm2"
                bin_obj = doc.addObject("Part::Feature", obj_name)
                bin_obj.Shape = compound
                bin_obj.ViewObject.PointSize = 4.0
                bin_obj.ViewObject.PointColor = color
                heatmap_group.addObject(bin_obj)
        
        return heatmap_group
    
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Heatmap visualization failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    return None


def create_ray_visualization(doc, parent_group, ray_paths, name="Rays", 
                             color=(1.0, 1.0, 0.0), max_rays=1000):
    """Create visualization of ray paths.
    
    Args:
        doc: FreeCAD document
        parent_group: Parent group for visualization
        ray_paths: List of ray path arrays
        name: Name for the ray object
        color: RGB tuple for ray color
        max_rays: Maximum number of rays to visualize
        
    Returns:
        Created ray object or None
    """
    if not ray_paths or len(ray_paths) == 0:
        return None
    
    try:
        # Sample rays if too many
        if len(ray_paths) > max_rays:
            indices = np.random.choice(len(ray_paths), max_rays, replace=False)
            paths = [ray_paths[i] for i in indices]
        else:
            paths = ray_paths
        
        # Create line segments for each ray
        edges = []
        for path in paths:
            if len(path) >= 2:
                for i in range(len(path) - 1):
                    try:
                        p1 = FreeCAD.Vector(path[i][0], path[i][1], path[i][2])
                        p2 = FreeCAD.Vector(path[i+1][0], path[i+1][1], path[i+1][2])
                        if (p2 - p1).Length > 0.001:
                            edges.append(Part.makeLine(p1, p2))
                    except:
                        continue
        
        if edges:
            compound = Part.Compound(edges)
            ray_obj = doc.addObject("Part::Feature", name)
            ray_obj.Shape = compound
            ray_obj.ViewObject.LineColor = color
            ray_obj.ViewObject.LineWidth = 1.0
            parent_group.addObject(ray_obj)
            return ray_obj
    
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Ray visualization failed: {e}\n")
    
    return None


def add_iteration_analytics(doc, parent_group, iteration, uniformity, efficiency,
                            lens_entry_rate=None, absorber_capture_rate=None):
    """Add spreadsheet with iteration analytics.
    
    Args:
        doc: FreeCAD document
        parent_group: Parent group for analytics
        iteration: Iteration number
        uniformity: Uniformity index
        efficiency: Ray efficiency
        lens_entry_rate: Fraction of rays hitting lens
        absorber_capture_rate: Fraction of lens rays reaching absorber
        
    Returns:
        Created spreadsheet object or None
    """
    try:
        spreadsheet = doc.addObject("Spreadsheet::Sheet", f"Analytics_{iteration:03d}")
        
        spreadsheet.set("A1", "Metric")
        spreadsheet.set("B1", "Value")
        
        spreadsheet.set("A2", "Iteration")
        spreadsheet.set("B2", str(iteration))
        
        spreadsheet.set("A3", "Uniformity")
        spreadsheet.set("B3", f"{uniformity:.4f}")
        
        spreadsheet.set("A4", "Efficiency")
        spreadsheet.set("B4", f"{efficiency*100:.1f}%")
        
        if lens_entry_rate is not None:
            spreadsheet.set("A5", "Lens Entry Rate")
            spreadsheet.set("B5", f"{lens_entry_rate*100:.1f}%")
        
        if absorber_capture_rate is not None:
            spreadsheet.set("A6", "Absorber Capture")
            spreadsheet.set("B6", f"{absorber_capture_rate*100:.1f}%")
        
        spreadsheet.set("A7", "Timestamp")
        spreadsheet.set("B7", datetime.now().strftime("%H:%M:%S"))
        
        parent_group.addObject(spreadsheet)
        return spreadsheet
        
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Analytics creation failed: {e}\n")
        return None


def get_wavelength_base_color(wavelength):
    """Get base RGB color (0-255) for a given wavelength.
    
    Args:
        wavelength: LED wavelength in nm
        
    Returns:
        Tuple of (r, g, b) in 0-255 range
    """
    if wavelength is None:
        return (128, 128, 128)  # Gray for unknown
    if wavelength <= 375:
        return (148, 0, 211)      # Violet/purple for 375nm
    elif wavelength <= 385:
        return (100, 0, 230)      # Blue-violet for 385nm
    elif wavelength <= 395:
        return (50, 50, 235)      # Deep blue-violet for 395nm
    else:
        return (0, 100, 255)      # Blue for 405nm


def create_wavelength_heatmap(doc, parent_group, led_exits, led_name, num_rays):
    """Create heatmap for a single LED source with wavelength-based coloring.
    
    Args:
        doc: FreeCAD document
        parent_group: Parent group for visualization
        led_exits: Dict with 'positions', 'intensities', 'wavelength' keys
        led_name: Name identifier for this LED (e.g., "LED1", "LED2")
        num_rays: Total number of rays simulated (for irradiance normalization)
        
    Returns:
        Created heatmap group or None
    """
    from Part import Compound, Vertex
    
    positions = led_exits.get('positions', np.array([]))
    intensities = led_exits.get('intensities', np.array([]))
    wavelength = led_exits.get('wavelength', 405)
    
    if len(positions) == 0:
        return None
    
    try:
        n_points = len(positions)
        MAX_VIZ_POINTS = 5000  # Limit for performance
        
        # Sample if too many points
        if n_points > MAX_VIZ_POINTS:
            sample_indices = np.random.choice(n_points, MAX_VIZ_POINTS, replace=False)
            viz_positions = positions[sample_indices]
            viz_intensities = intensities[sample_indices]
            sampling_ratio = n_points / MAX_VIZ_POINTS
        else:
            viz_positions = positions
            viz_intensities = intensities
            sampling_ratio = 1.0
        
        # Grid-based irradiance calculation
        grid_size = 20
        x = viz_positions[:, 0]
        y = viz_positions[:, 1]
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = max(x_max - x_min, 0.1)
        y_range = max(y_max - y_min, 0.1)
        
        # Calculate neighborhood area for irradiance
        avg_spread = np.mean(np.std(positions, axis=0)) if len(positions) > 1 else 1.0
        energy_radius = max(avg_spread / 3.0, 1.0)
        neighborhood_area_mm2 = np.pi * energy_radius ** 2
        neighborhood_area_cm2 = neighborhood_area_mm2 / 100.0
        
        # Compute grid indices
        x_idx = np.clip(((x - x_min) / x_range * (grid_size - 1)).astype(int), 0, grid_size - 1)
        y_idx = np.clip(((y - y_min) / y_range * (grid_size - 1)).astype(int), 0, grid_size - 1)
        
        # Accumulate intensities into grid
        grid_intensity = np.zeros((grid_size, grid_size))
        np.add.at(grid_intensity, (x_idx, y_idx), viz_intensities)
        
        # Convert to irradiance (mW/cm²)
        grid_irradiance = grid_intensity * sampling_ratio / max(1, num_rays) / max(0.01, neighborhood_area_cm2)
        
        # Look up irradiance for each point
        local_energies = grid_irradiance[x_idx, y_idx]
        
        energy_min = np.min(local_energies) if len(local_energies) > 0 else 0
        energy_max = np.max(local_energies) if len(local_energies) > 0 else 1
        
        # Get base color for this wavelength
        base_r, base_g, base_b = get_wavelength_base_color(wavelength)
        
        def get_heatmap_color(value, min_val, max_val):
            """Get color based on intensity - brighter = more intense."""
            if max_val <= min_val:
                return (0.55, 0.55, 0.55)  # Gray (140/255)
            t = (value - min_val) / (max_val - min_val)
            t = max(0, min(1, t))
            brightness = 0.4 + (t * 0.6)
            return (base_r / 255.0 * brightness, base_g / 255.0 * brightness, base_b / 255.0 * brightness)
        
        # Create heatmap group
        heatmap_group = doc.addObject("App::DocumentObjectGroup", f"Heatmap_{led_name}_{wavelength}nm")
        parent_group.addObject(heatmap_group)
        
        # Group points by intensity level
        color_bins = 8
        if energy_max > energy_min:
            bin_indices = np.clip(((local_energies - energy_min) / (energy_max - energy_min) * color_bins).astype(int), 0, color_bins - 1)
        else:
            bin_indices = np.full(len(local_energies), color_bins // 2, dtype=int)
        
        # Create colored point compounds (one per intensity bin)
        for bin_idx in range(color_bins):
            mask = bin_indices == bin_idx
            if not np.any(mask):
                continue
            
            bin_positions = viz_positions[mask]
            bin_energies = local_energies[mask]
            
            t = bin_idx / (color_bins - 1) if color_bins > 1 else 0.5
            fake_energy = energy_min + t * (energy_max - energy_min)
            color = get_heatmap_color(fake_energy, energy_min, energy_max)
            
            # Create vertices
            vertices = [Vertex(FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))) 
                       for p in bin_positions]
            
            if vertices:
                compound = Compound(vertices)
                avg_irradiance = np.mean(bin_energies)
                obj = doc.addObject("Part::Feature", f"Heatmap_{wavelength}nm_{avg_irradiance:.1f}mWcm2")
                obj.Shape = compound
                obj.ViewObject.PointColor = color
                obj.ViewObject.PointSize = 6.0
                heatmap_group.addObject(obj)
        
        return heatmap_group
    
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Wavelength heatmap creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None
