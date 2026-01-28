"""
Raytracing analysis for existing lens geometry.

This module provides the analyze_lens function that runs raytracing
on an existing lens solid and returns analysis results.
"""

import os
import tempfile
import time
from datetime import datetime
from typing import Optional, Dict, Tuple
import FreeCAD
import FreeCADGui

from .config import (
    COMPUTE_BUDGETS,
    DEFAULT_MAX_BOUNCES,
    USE_RAYTRACER_SERVER
)
from .sleep_prevention import prevent_sleep, allow_sleep, keep_ui_responsive
from .server_management import start_raytracer_server, stop_raytracer_server
from .raytracer_integration import (
    find_python_executable,
    run_raytracer_evaluation,
    calculate_efficiency,
    calculate_lens_entry_rate,
    calculate_absorber_capture_rate
)
from .mesh_operations import solid_to_mesh, export_mesh_to_stl
from .uniformity_analysis import analyze_uniformity_grid, calculate_fitness
from .user_interface import get_led_position, show_progress_dialog


def analyze_lens(led_obj, lens_obj, absorber_obj,
                 budget_name: str = "Quick",
                 led_power_mW: float = 1420.0,
                 led_current_mA: float = 700.0,
                 led_model: str = "U405",
                 led_wavelength: int = 405,
                 led1_direction: Tuple = None,
                 led2_config: Dict = None,
                 lens_material: Dict = None,
                 absorber_material: Dict = None) -> Optional[Tuple[Dict, Dict]]:
    """Analyze existing lens geometry with raytracing.
    
    Args:
        led_obj: FreeCAD object for LED position
        lens_obj: FreeCAD solid to analyze (the lens geometry)
        absorber_obj: FreeCAD object defining absorber
        budget_name: "Quick", "Medium", or "Thorough"
        led_power_mW: LED radiant power in milliwatts
        led_current_mA: LED forward current in milliamps (for reporting)
        led_model: LED model identifier
        led_wavelength: LED wavelength in nm
        led1_direction: Optional LED direction tuple (x, y, z)
        led2_config: Optional configuration for second LED
        lens_material: Optional lens material properties
        absorber_material: Optional absorber material properties
        
    Returns:
        Tuple of (analysis_result, stats_dict) or None on failure
    """
    # Check for Python with PyCUDA FIRST
    python_exe = find_python_executable()
    if python_exe is None:
        FreeCAD.Console.PrintError("\n" + "="*70 + "\n")
        FreeCAD.Console.PrintError("ANALYSIS CANCELLED: PyCUDA not available\n")
        FreeCAD.Console.PrintError("="*70 + "\n")
        return None
    
    # Prevent system sleep
    prevent_sleep()
    
    # Start raytracer server
    if USE_RAYTRACER_SERVER:
        server_started = start_raytracer_server(python_exe)
        if not server_started:
            FreeCAD.Console.PrintError("\n" + "="*70 + "\n")
            FreeCAD.Console.PrintError("ANALYSIS CANCELLED: Raytracer server failed to start\n")
            FreeCAD.Console.PrintError("="*70 + "\n")
            allow_sleep()
            return None
    else:
        FreeCAD.Console.PrintError("USE_RAYTRACER_SERVER is disabled - cannot run analysis\n")
        allow_sleep()
        return None
    
    budget = COMPUTE_BUDGETS[budget_name]
    start_time = time.time()
    
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("RAYTRACING ANALYSIS\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage(f"Lens: {lens_obj.Label}\n")
    FreeCAD.Console.PrintMessage(f"Budget: {budget_name} ({budget['rays']} rays)\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n\n")
    
    # Get LED position and direction
    led_pos = get_led_position(led_obj)
    
    if led1_direction:
        led_dir = FreeCAD.Vector(led1_direction[0], led1_direction[1], led1_direction[2])
        led_dir.normalize()
    else:
        led_dir = FreeCAD.Vector(0, 1, 0)  # Default Y direction
    
    FreeCAD.Console.PrintMessage(f"LED Position: ({led_pos.x:.2f}, {led_pos.y:.2f}, {led_pos.z:.2f})\n")
    FreeCAD.Console.PrintMessage(f"LED Direction: ({led_dir.x:.2f}, {led_dir.y:.2f}, {led_dir.z:.2f})\n\n")
    
    # Export lens and absorber to STL
    FreeCAD.Console.PrintMessage("Exporting geometry to STL...\n")
    
    lens_stl = tempfile.mktemp(suffix='_lens.stl')
    absorber_stl = tempfile.mktemp(suffix='_absorber.stl')
    
    try:
        lens_mesh = solid_to_mesh(lens_obj.Shape, budget['mesh_resolution'])
        export_mesh_to_stl(lens_mesh, lens_stl)
        
        absorber_mesh = solid_to_mesh(absorber_obj.Shape, budget.get('absorber_resolution', 1.0))
        export_mesh_to_stl(absorber_mesh, absorber_stl)
        
        keep_ui_responsive()
    except Exception as e:
        FreeCAD.Console.PrintError(f"Failed to export geometry: {e}\n")
        stop_raytracer_server()
        allow_sleep()
        return None
    
    # Progress dialog
    progress = None
    try:
        progress = show_progress_dialog("Analyzing Lens Geometry")
        progress.setValue(10)
    except:
        pass
    
    try:
        import numpy as np
        
        # Run raytracing for LED 1
        FreeCAD.Console.PrintMessage(f"Running raytracing for LED 1 ({led_wavelength}nm)...\n")
        if progress:
            progress.setValue(20)
        
        led1_result = run_raytracer_evaluation(
            lens_stl, absorber_stl,
            led_pos, led_dir,
            num_rays=budget['rays'],
            max_bounces=DEFAULT_MAX_BOUNCES,
            led_power=led_power_mW,
            led_model=led_model,
            led_wavelength=led_wavelength,
            lens_material=lens_material,
            absorber_material=absorber_material
        )
        
        keep_ui_responsive()
        
        if led1_result is None:
            FreeCAD.Console.PrintError("Raytracing failed for LED 1!\n")
            return None
        
        # Structure LED 1 results
        led1_exits = {
            'positions': np.array(led1_result.get('exit_positions', [])),
            'intensities': np.array(led1_result.get('exit_intensities', [])),
            'wavelength': led_wavelength
        }
        
        FreeCAD.Console.PrintMessage(f"  LED 1: {len(led1_exits['positions'])} absorber exits\n")
        
        if progress:
            progress.setValue(50)
        
        # Run raytracing for LED 2 if configured
        led2_exits = {'positions': np.array([]), 'intensities': np.array([]), 'wavelength': None}
        led2_result = None
        
        if led2_config is not None:
            led2_obj_name = led2_config.get('position_object')
            if led2_obj_name:
                doc = FreeCAD.ActiveDocument
                led2_obj = doc.getObject(led2_obj_name) if doc else None
                if led2_obj:
                    led2_pos = get_led_position(led2_obj)
                    led2_dir_tuple = led2_config.get('direction', (0, 1, 0))
                    led2_dir = FreeCAD.Vector(led2_dir_tuple[0], led2_dir_tuple[1], led2_dir_tuple[2])
                    if led2_dir.Length > 0.001:
                        led2_dir.normalize()
                    led2_power = led2_config.get('power_mW', led_power_mW)
                    led2_model = led2_config.get('model', led_model)
                    led2_wavelength = led2_config.get('wavelength', led_wavelength)
                    
                    FreeCAD.Console.PrintMessage(f"Running raytracing for LED 2 ({led2_wavelength}nm)...\n")
                    
                    led2_result = run_raytracer_evaluation(
                        lens_stl, absorber_stl,
                        led2_pos, led2_dir,
                        num_rays=budget['rays'],
                        max_bounces=DEFAULT_MAX_BOUNCES,
                        led_power=led2_power,
                        led_model=led2_model,
                        led_wavelength=led2_wavelength,
                        lens_material=lens_material,
                        absorber_material=absorber_material
                    )
                    
                    keep_ui_responsive()
                    
                    if led2_result:
                        led2_exits = {
                            'positions': np.array(led2_result.get('exit_positions', [])),
                            'intensities': np.array(led2_result.get('exit_intensities', [])),
                            'wavelength': led2_wavelength
                        }
                        FreeCAD.Console.PrintMessage(f"  LED 2: {len(led2_exits['positions'])} absorber exits\n")
        
        if progress:
            progress.setValue(80)
        
        # Combine results for analysis
        all_positions = []
        all_intensities = []
        if len(led1_exits['positions']) > 0:
            all_positions.append(led1_exits['positions'])
            all_intensities.append(led1_exits['intensities'])
        if len(led2_exits['positions']) > 0:
            all_positions.append(led2_exits['positions'])
            all_intensities.append(led2_exits['intensities'])
        
        if all_positions:
            combined_positions = np.vstack(all_positions)
            combined_intensities = np.concatenate(all_intensities)
        else:
            combined_positions = np.array([]).reshape(0, 3)
            combined_intensities = np.array([])
        
        # Calculate metrics
        total_rays = budget['rays'] * (2 if led2_result else 1)
        
        grid_analysis = analyze_uniformity_grid(
            combined_positions,
            combined_intensities,
            num_rays=total_rays
        )
        
        uniformity = grid_analysis['uniformity_index']
        efficiency = calculate_efficiency(led1_result, budget['rays'])
        lens_entry_rate = calculate_lens_entry_rate(led1_result, budget['rays'])
        absorber_capture_rate = calculate_absorber_capture_rate(led1_result, budget['rays'])
        fitness = calculate_fitness(uniformity, efficiency, lens_entry_rate, absorber_capture_rate)
        
        if progress:
            progress.setValue(100)
        
    finally:
        if progress:
            progress.close()
        
        # Cleanup temp files
        try:
            os.remove(lens_stl)
        except:
            pass
        try:
            os.remove(absorber_stl)
        except:
            pass
        
        stop_raytracer_server()
        allow_sleep()
    
    total_time = time.time() - start_time
    
    # Build results
    analysis_result = {
        'exit_positions': combined_positions,
        'exit_intensities': combined_intensities,
        'led1_exits': led1_exits,
        'led2_exits': led2_exits,
        'led1_result': led1_result,
        'led2_result': led2_result,
    }
    
    stats = {
        'fitness': fitness,
        'uniformity': uniformity,
        'efficiency': efficiency,
        'lens_entry_rate': lens_entry_rate,
        'absorber_capture_rate': absorber_capture_rate,
        'grid_analysis': grid_analysis,
        'num_rays': budget['rays'],
        'total_rays': total_rays,
        'budget_name': budget_name,
        'total_time_seconds': total_time,
        'led1_model': led_model,
        'led1_current_mA': led_current_mA,
        'led1_wavelength': led_wavelength,
        'led1_power_mW': led_power_mW,
        'led2_config': led2_config,
        'lens_material': lens_material,
        'absorber_material': absorber_material,
    }
    
    # Print summary
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("ANALYSIS COMPLETE\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage(f"Analysis Time: {total_time:.1f}s\n")
    FreeCAD.Console.PrintMessage(f"Fitness Score: {fitness:.4f}\n")
    FreeCAD.Console.PrintMessage(f"Uniformity Index: {uniformity:.4f}\n")
    FreeCAD.Console.PrintMessage(f"Efficiency: {efficiency*100:.1f}%\n")
    FreeCAD.Console.PrintMessage(f"Lens Entry Rate: {lens_entry_rate*100:.1f}%\n")
    FreeCAD.Console.PrintMessage(f"Absorber Capture Rate: {absorber_capture_rate*100:.1f}%\n")
    
    if grid_analysis:
        FreeCAD.Console.PrintMessage("\nIrradiance Distribution:\n")
        FreeCAD.Console.PrintMessage(f"  Mean: {grid_analysis.get('mean_intensity', 0):.4f} mW/cm²\n")
        FreeCAD.Console.PrintMessage(f"  Max:  {grid_analysis.get('max_intensity', 0):.4f} mW/cm²\n")
        FreeCAD.Console.PrintMessage(f"  Min:  {grid_analysis.get('min_intensity', 0):.4f} mW/cm²\n")
        FreeCAD.Console.PrintMessage(f"  Hot Zones: {len(grid_analysis.get('hot_zones', []))}\n")
        FreeCAD.Console.PrintMessage(f"  Cold Zones: {len(grid_analysis.get('cold_zones', []))}\n")
    
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    
    return analysis_result, stats
