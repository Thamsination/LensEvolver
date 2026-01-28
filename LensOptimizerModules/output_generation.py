"""
Output generation functions for the Lens Optimizer.

This module provides functions to create FreeCAD objects and reports
from optimization results.
"""

from datetime import datetime
from typing import Dict, Optional
import FreeCAD
import Part

from .materials import LENS_REFRACTIVE_INDEX, ABSORBER_REFRACTIVE_INDEX
from .config import LED_MODELS


def create_optimized_lens_object(doc, optimized_solid, name="OptimizedLens"):
    """Add optimized lens to FreeCAD document.
    
    Args:
        doc: FreeCAD document
        optimized_solid: Part.Shape of the optimized lens
        name: Name for the lens object
        
    Returns:
        Created FreeCAD object
    """
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = optimized_solid
    
    # Set visual properties
    obj.ViewObject.ShapeColor = (0.6, 0.8, 1.0)  # Light blue
    obj.ViewObject.Transparency = 50
    
    return obj


def create_optimization_report(doc, results: Dict, name="OptimizationReport"):
    """Create a report text document with optimization results.
    
    Args:
        doc: FreeCAD document
        results: Dict with optimization results
        name: Name for the text document
        
    Returns:
        Text document object
    """
    text_doc = doc.addObject("App::TextDocument", name)
    
    # Extract data
    best_iteration = results.get('best_iteration', 0)
    total_iterations = results.get('iterations', 0)
    best_fitness = results.get('best_fitness', 0)
    best_uniformity = results.get('best_uniformity', 0)
    best_efficiency = results.get('best_efficiency', 0)
    efficiency_threshold = results.get('efficiency_threshold', 0.95)
    starting_scale = results.get('starting_scale', 0.8)
    
    # Format history strings
    uniformity_hist = results.get('uniformity_history', [])
    efficiency_hist = results.get('efficiency_history', [])
    fitness_hist = results.get('fitness_history', [])
    
    uniformity_str = ", ".join([f"{u:.4f}" for u in uniformity_hist[-20:]])
    efficiency_str = ", ".join([f"{e*100:.1f}%" for e in efficiency_hist[-20:]])
    fitness_str = ", ".join([f"{f:.4f}" for f in fitness_hist[-20:]])
    
    efficiency_status = "MET" if best_efficiency >= efficiency_threshold else "NOT MET"
    
    report_text = f"""Lens Geometry Optimization Report
{'='*50}

OPTIMIZATION RESULTS
--------------------
  Best Iteration:        {best_iteration} (of {total_iterations} total)
  Best Fitness Score:    {best_fitness:.4f}
  Best Uniformity Index: {best_uniformity:.4f}
  Best Efficiency:       {best_efficiency*100:.1f}% {efficiency_status}
  
  Efficiency Threshold:  {efficiency_threshold*100:.0f}% (rays exiting absorber)
  Starting Scale:        {starting_scale*100:.0f}% of envelope

FITNESS HISTORY (last 20 iterations)
------------------------------------
  {fitness_str}

UNIFORMITY HISTORY (last 20 iterations)
---------------------------------------
  {uniformity_str}

EFFICIENCY HISTORY (last 20 iterations)
---------------------------------------
  {efficiency_str}

METRICS INTERPRETATION
----------------------
  FITNESS SCORE (0.0 - 1.0):
    Combines uniformity and efficiency. Higher is better.
    
  UNIFORMITY INDEX (0.0 - 1.0):
    1.0 = Perfect uniformity (all areas receive equal energy)
    0.0 = No uniformity (all energy concentrated in one spot)
    > 0.8 = Excellent | > 0.7 = Good | > 0.5 = Moderate
    
  EFFICIENCY (%):
    Percentage of rays that exit the absorber.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    text_doc.Text = report_text
    return text_doc


def create_evolution_report(doc, results: Dict, name="EvolutionReport"):
    """Create a report text document for evolutionary optimization.
    
    Args:
        doc: FreeCAD document
        results: Dict with evolution results
        name: Name for the text document
        
    Returns:
        Text document object
    """
    text_doc = doc.addObject("App::TextDocument", name)
    
    # Extract basic data
    best_fitness = results.get('best_fitness', 0)
    best_uniformity = results.get('best_uniformity', 0)
    best_efficiency = results.get('best_efficiency', 0)
    best_lens_entry = results.get('best_lens_entry', 1.0)
    best_absorber_capture = results.get('best_absorber_capture', 1.0)
    generations = results.get('generations', 0)
    population_size = results.get('population_size', 0)
    best_generation = results.get('best_generation', 0)
    
    # Timing
    total_time_seconds = results.get('total_time_seconds', 0)
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    # LED 1 configuration
    led1_model = results.get('led1_model', 'U405')
    led1_current_mA = results.get('led1_current_mA', 700)
    led1_wavelength = results.get('led1_wavelength', 405)
    led1_power_mW = results.get('led1_power_mW', 1420)
    
    # Get full LED model name
    led1_full_name = LED_MODELS.get(led1_model, {}).get('name', f'NVSU119CT-{led1_model}')
    
    # LED 2 configuration
    led2_config = results.get('led2_config')
    led2_section = ""
    if led2_config:
        led2_model = led2_config.get('model', 'U405')
        led2_current_mA = led2_config.get('current_mA', 700)
        led2_wavelength = led2_config.get('wavelength', 405)
        led2_power_mW = led2_config.get('power_mW', 1420)
        led2_full_name = LED_MODELS.get(led2_model, {}).get('name', f'NVSU119CT-{led2_model}')
        led2_section = f"""  LED 2: {led2_full_name} ({led2_wavelength}nm)
         Current: {led2_current_mA:.0f} mA -> {led2_power_mW:.0f} mW"""
    else:
        led2_section = "  LED 2: Disabled"
    
    # Raytracing settings
    num_rays = results.get('num_rays', 0)
    budget_name = results.get('budget_name', 'Unknown')
    
    # Material info
    lens_mat = results.get('lens_material')
    absorber_mat = results.get('absorber_material')
    
    lens_name = lens_mat.get('name', 'Custom') if lens_mat else 'Default'
    lens_n = lens_mat.get('refractive_index', LENS_REFRACTIVE_INDEX) if lens_mat else LENS_REFRACTIVE_INDEX
    lens_abs = lens_mat.get('absorption_coeff', 0.001) if lens_mat else 0.001
    
    absorber_name = absorber_mat.get('name', 'Custom') if absorber_mat else 'Default'
    absorber_n = absorber_mat.get('refractive_index', ABSORBER_REFRACTIVE_INDEX) if absorber_mat else ABSORBER_REFRACTIVE_INDEX
    absorber_abs = absorber_mat.get('absorption_coeff', 0.05) if absorber_mat else 0.05
    
    # Profile information from best individual
    best_individual = results.get('best_individual')
    profile_section = ""
    if best_individual and hasattr(best_individual, 'profiles'):
        profiles = sorted(best_individual.profiles, key=lambda p: p.z_position)
        profile_lines = []
        for i, p in enumerate(profiles):
            profile_lines.append(f"  Profile {i+1}: t={p.z_position:.2f}, radius={p.radius:.2f}mm")
        profile_section = "\n".join(profile_lines)
    else:
        profile_section = "  (no profile data available)"
    
    # Profile shape type
    use_circle = results.get('use_circle', False)
    profile_shape = "Circular" if use_circle else "Polygon"
    
    # Fitness history
    fitness_hist = results.get('fitness_history', [])
    fitness_str = ", ".join([f"{f:.4f}" for f in fitness_hist[-20:]])
    
    report_text = f"""Evolutionary Lens Optimization Report
{'='*50}

LED CONFIGURATION
-----------------
  LED 1: {led1_full_name} ({led1_wavelength}nm)
         Current: {led1_current_mA:.0f} mA -> {led1_power_mW:.0f} mW
{led2_section}

RAYTRACING SETTINGS
-------------------
  Rays per LED:    {num_rays:,}
  Quality Budget:  {budget_name}

MATERIAL PROPERTIES
-------------------
  Lens:     {lens_name} (n={lens_n:.3f}, a={lens_abs:.4f}/mm)
  Absorber: {absorber_name} (n={absorber_n:.3f}, a={absorber_abs:.4f}/mm)

EVOLUTION RESULTS
-----------------
  Best Generation:       {best_generation} (of {generations} total)
  Best Fitness Score:    {best_fitness:.4f}
  Best Uniformity Index: {best_uniformity:.4f}
  Best Efficiency:       {best_efficiency*100:.1f}%
  Lens Entry Rate:       {best_lens_entry*100:.1f}%
  Absorber Capture:      {best_absorber_capture*100:.1f}%
  
  Generations:           {generations}
  Population Size:       {population_size}

BEST LENS PROFILE ({profile_shape})
-----------------------
{profile_section}

FITNESS HISTORY (last 20 generations)
-------------------------------------
  {fitness_str}

TIMING
------
  Total Time: {time_str}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    text_doc.Text = report_text
    return text_doc


def create_analysis_report(doc, results: Dict, lens_name: str = "Unknown", name="AnalysisReport"):
    """Create a report text document for raytracing analysis of existing geometry.
    
    Args:
        doc: FreeCAD document
        results: Dict with analysis results
        lens_name: Name of the lens being analyzed
        name: Name for the text document
        
    Returns:
        Text document object
    """
    text_doc = doc.addObject("App::TextDocument", name)
    
    # Extract analysis results
    fitness = results.get('fitness', 0)
    uniformity = results.get('uniformity', 0)
    efficiency = results.get('efficiency', 0)
    lens_entry_rate = results.get('lens_entry_rate', 1.0)
    absorber_capture_rate = results.get('absorber_capture_rate', 1.0)
    
    # Timing
    total_time_seconds = results.get('total_time_seconds', 0)
    minutes = int(total_time_seconds // 60)
    seconds = int(total_time_seconds % 60)
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    # LED 1 configuration
    led1_model = results.get('led1_model', 'U405')
    led1_current_mA = results.get('led1_current_mA', 700)
    led1_wavelength = results.get('led1_wavelength', 405)
    led1_power_mW = results.get('led1_power_mW', 1420)
    
    # Get full LED model name
    led1_full_name = LED_MODELS.get(led1_model, {}).get('name', f'NVSU119CT-{led1_model}')
    
    # LED 2 configuration
    led2_config = results.get('led2_config')
    led2_section = ""
    if led2_config:
        led2_model = led2_config.get('model', 'U405')
        led2_current_mA = led2_config.get('current_mA', 700)
        led2_wavelength = led2_config.get('wavelength', 405)
        led2_power_mW = led2_config.get('power_mW', 1420)
        led2_full_name = LED_MODELS.get(led2_model, {}).get('name', f'NVSU119CT-{led2_model}')
        led2_section = f"""  LED 2: {led2_full_name} ({led2_wavelength}nm)
         Current: {led2_current_mA:.0f} mA -> {led2_power_mW:.0f} mW"""
    else:
        led2_section = "  LED 2: Disabled"
    
    # Raytracing settings
    num_rays = results.get('num_rays', 0)
    total_rays = results.get('total_rays', num_rays)
    budget_name = results.get('budget_name', 'Unknown')
    
    # Material info
    lens_mat = results.get('lens_material')
    absorber_mat = results.get('absorber_material')
    
    lens_mat_name = lens_mat.get('name', 'Custom') if lens_mat else 'Default'
    lens_n = lens_mat.get('refractive_index', LENS_REFRACTIVE_INDEX) if lens_mat else LENS_REFRACTIVE_INDEX
    lens_abs = lens_mat.get('absorption_coeff', 0.001) if lens_mat else 0.001
    
    absorber_mat_name = absorber_mat.get('name', 'Custom') if absorber_mat else 'Default'
    absorber_n = absorber_mat.get('refractive_index', ABSORBER_REFRACTIVE_INDEX) if absorber_mat else ABSORBER_REFRACTIVE_INDEX
    absorber_abs = absorber_mat.get('absorption_coeff', 0.05) if absorber_mat else 0.05
    
    # Irradiance distribution from grid analysis
    grid_analysis = results.get('grid_analysis', {})
    mean_intensity = grid_analysis.get('mean_intensity', 0)
    max_intensity = grid_analysis.get('max_intensity', 0)
    min_intensity = grid_analysis.get('min_intensity', 0)
    hot_zones = len(grid_analysis.get('hot_zones', []))
    cold_zones = len(grid_analysis.get('cold_zones', []))
    
    report_text = f"""Raytracing Analysis Report
{'='*50}

LENS ANALYZED
-------------
  {lens_name}

LED CONFIGURATION
-----------------
  LED 1: {led1_full_name} ({led1_wavelength}nm)
         Current: {led1_current_mA:.0f} mA -> {led1_power_mW:.0f} mW
{led2_section}

RAYTRACING SETTINGS
-------------------
  Rays per LED:    {num_rays:,}
  Total Rays:      {total_rays:,}
  Quality Budget:  {budget_name}

MATERIAL PROPERTIES
-------------------
  Lens:     {lens_mat_name} (n={lens_n:.3f}, a={lens_abs:.4f}/mm)
  Absorber: {absorber_mat_name} (n={absorber_n:.3f}, a={absorber_abs:.4f}/mm)

ANALYSIS RESULTS
----------------
  Fitness Score:         {fitness:.4f}
  Uniformity Index:      {uniformity:.4f}
  Efficiency:            {efficiency*100:.1f}%
  Lens Entry Rate:       {lens_entry_rate*100:.1f}%
  Absorber Capture Rate: {absorber_capture_rate*100:.1f}%

IRRADIANCE DISTRIBUTION
-----------------------
  Mean Intensity:  {mean_intensity:.4f} mW/cm²
  Max Intensity:   {max_intensity:.4f} mW/cm²
  Min Intensity:   {min_intensity:.4f} mW/cm²
  Hot Zones:       {hot_zones}
  Cold Zones:      {cold_zones}

METRICS INTERPRETATION
----------------------
  FITNESS SCORE (0.0 - 1.0):
    Combines uniformity, efficiency, and capture rates. Higher is better.
    
  UNIFORMITY INDEX (0.0 - 1.0):
    1.0 = Perfect uniformity (all areas receive equal energy)
    0.0 = No uniformity (all energy concentrated in one spot)
    > 0.8 = Excellent | > 0.7 = Good | > 0.5 = Moderate
    
  EFFICIENCY (%):
    Percentage of rays that exit the absorber.
    
  HOT/COLD ZONES:
    Areas with intensity significantly above/below the mean.

TIMING
------
  Analysis Time: {time_str}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    text_doc.Text = report_text
    return text_doc


def create_parametric_report(doc, results: Dict, name="ParametricReport"):
    """Create a report for parametric loft optimization.
    
    Args:
        doc: FreeCAD document
        results: Dict with optimization results (includes profile data)
        name: Name for the text document
        
    Returns:
        Text document object
    """
    text_doc = doc.addObject("App::TextDocument", name)
    
    best_iteration = results.get('best_iteration', 0)
    total_iterations = results.get('iterations', 0)
    best_fitness = results.get('best_fitness', 0)
    best_uniformity = results.get('best_uniformity', 0)
    best_efficiency = results.get('best_efficiency', 0)
    num_profiles = results.get('num_profiles', 5)
    best_profiles = results.get('best_profiles', [])
    
    # Format profile parameters
    profile_lines = []
    for i, p in enumerate(best_profiles):
        profile_lines.append(
            f"    Profile {i+1}: radius={p.radius:.2f}mm, sides={p.sides}, "
            f"angle={p.angle:.1f}, z_pos={p.z_position:.2f}"
        )
    profile_str = "\n".join(profile_lines) if profile_lines else "    (no profile data)"
    
    report_text = f"""Parametric Lens Optimization Report
{'='*50}

OPTIMIZATION METHOD: Lofted Polygon Profiles
---------------------------------------------
  Number of Profiles: {num_profiles}

OPTIMIZATION RESULTS
--------------------
  Best Iteration:        {best_iteration} (of {total_iterations} total)
  Best Fitness Score:    {best_fitness:.4f}
  Best Uniformity Index: {best_uniformity:.4f}
  Best Efficiency:       {best_efficiency*100:.1f}%

BEST PROFILE PARAMETERS
-----------------------
{profile_str}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    text_doc.Text = report_text
    return text_doc


def create_best_lens_geometry(doc, result_group, centerline_points, best_individual, 
                               use_circle: bool = False):
    """Create toggleable geometry objects for the best lens centerline and profiles.
    
    Creates FreeCAD wire objects representing:
    - The centerline curve (BSpline through centerline points)
    - Profile shapes at each profile position (circles or polygons)
    
    All objects are created with Visibility=False so the user can toggle them on.
    
    Args:
        doc: FreeCAD document
        result_group: Parent group to add geometry to
        centerline_points: List of CenterlinePoint objects
        best_individual: Individual object containing profile data
        use_circle: If True, create circular profiles; if False, create polygon profiles
        
    Returns:
        Created group object containing the geometry, or None on failure
    """
    from .profile_creation import create_profile, interpolate_centerline_at_t
    
    if not centerline_points or not best_individual:
        FreeCAD.Console.PrintWarning("Cannot create lens geometry: missing centerline or profile data\n")
        return None
    
    try:
        # Create a subgroup for the geometry
        geo_group = doc.addObject("App::DocumentObjectGroup", "BestLensGeometry")
        result_group.addObject(geo_group)
        
        # --- Create Centerline BSpline ---
        try:
            spine_points = [cp.position for cp in centerline_points]
            
            # Create BSpline curve through the centerline points
            bspline = Part.BSplineCurve()
            bspline.interpolate(spine_points)
            centerline_wire = Part.Wire([bspline.toShape()])
            
            # Create FreeCAD object
            centerline_obj = doc.addObject("Part::Feature", "Centerline")
            centerline_obj.Shape = centerline_wire
            centerline_obj.ViewObject.LineColor = (1.0, 0.5, 0.0)  # Orange
            centerline_obj.ViewObject.LineWidth = 2.0
            centerline_obj.ViewObject.Visibility = False
            geo_group.addObject(centerline_obj)
            
            FreeCAD.Console.PrintMessage(f"Created centerline with {len(spine_points)} points\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not create centerline: {e}\n")
        
        # --- Create Profile Wires ---
        if hasattr(best_individual, 'profiles'):
            sorted_profiles = sorted(best_individual.profiles, key=lambda p: p.z_position)
            
            for i, profile in enumerate(sorted_profiles):
                try:
                    # Get position and tangent at this profile's z_position
                    center, tangent = interpolate_centerline_at_t(centerline_points, profile.z_position)
                    
                    # Create profile wire (circle or polygon)
                    profile_wire = create_profile(profile, center, tangent, use_circle)
                    
                    # Create FreeCAD object
                    profile_obj = doc.addObject("Part::Feature", f"Profile_{i+1:02d}")
                    profile_obj.Shape = profile_wire
                    profile_obj.ViewObject.LineColor = (0.0, 0.8, 0.2)  # Green
                    profile_obj.ViewObject.LineWidth = 1.5
                    profile_obj.ViewObject.Visibility = False
                    geo_group.addObject(profile_obj)
                    
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Could not create profile {i+1}: {e}\n")
            
            FreeCAD.Console.PrintMessage(f"Created {len(sorted_profiles)} profile geometries\n")
        
        # Set the group itself as invisible
        geo_group.ViewObject.Visibility = False
        
        return geo_group
        
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Could not create lens geometry: {e}\n")
        import traceback
        traceback.print_exc()
        return None
