"""
Evolutionary optimization engine for the Lens Optimizer.

This module provides the main evolve_lens function that uses evolutionary
algorithms to optimize lens geometry.
"""

import os
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import FreeCAD
import FreeCADGui
import Part

from .config import (
    COMPUTE_BUDGETS,
    DEFAULT_MAX_BOUNCES,
    DEFAULT_NUM_PROFILES,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_ELITE_COUNT,
    USE_RAYTRACER_SERVER
)
from .data_classes import ProfileParams, Individual, CenterlinePoint
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
from .user_interface import get_led_position, get_led_direction_from_user, show_progress_dialog
from .centerline_extraction import extract_curved_centerline_auto, extract_centerline_from_sketch
from .geometry_validation import validate_lens_geometry
from .lofting import create_swept_lens
from .entry_spheres import create_lens_with_entry_spheres, cut_lens_entry_spheres
from .evolutionary_core import (
    create_random_individual,
    mutate_individual,
    evolve_generation
)


def build_led_lists(led_pos, led_dir, led2_config=None):
    """Build lists of LED positions and directions including LED 2 if configured.
    
    Args:
        led_pos: LED 1 position vector
        led_dir: LED 1 direction vector
        led2_config: Optional LED 2 configuration dict with 'position_object' or 'position' and 'direction'
        
    Returns:
        Tuple of (led_positions list, led_directions list)
    """
    led_positions = [led_pos]
    led_directions = [led_dir]
    
    # Add LED 2 if configured
    if led2_config is not None:
        led2_obj_name = led2_config.get('position_object')
        if led2_obj_name:
            active_doc = FreeCAD.ActiveDocument
            led2_obj = active_doc.getObject(led2_obj_name) if active_doc else None
            if led2_obj:
                led2_pos = get_led_position(led2_obj)
                led_positions.append(led2_pos)
                led2_dir_tuple = led2_config.get('direction', (0, 1, 0))
                led2_dir = FreeCAD.Vector(led2_dir_tuple[0], led2_dir_tuple[1], led2_dir_tuple[2])
                if led2_dir.Length > 0.001:
                    led2_dir.normalize()
                led_directions.append(led2_dir)
                FreeCAD.Console.PrintMessage(f"LED 2 will be included in spherical entry cuts\n")
    
    return led_positions, led_directions


def evaluate_individual(individual: Individual,
                        centerline_points: List[CenterlinePoint],
                        led_pos, led_dir,
                        absorber_stl_path: str,
                        budget: Dict,
                        led_power_mW: float = 1420.0,
                        led_model: str = "U405",
                        led_wavelength: int = 405,
                        enable_entry_spheres: bool = True,
                        entry_sphere_depth: float = 0.3,
                        use_circle: bool = False,
                        led_positions: List = None,
                        led_directions: List = None,
                        lens_material: Dict = None,
                        absorber_material: Dict = None) -> Individual:
    """Evaluate an individual by raytracing its lens geometry.
    
    Args:
        individual: The individual to evaluate
        centerline_points: Centerline for lens generation
        led_pos: LED 1 position (for raytracing)
        led_dir: LED 1 direction (for raytracing)
        absorber_stl_path: Path to absorber STL file
        budget: Compute budget settings
        led_power_mW: LED power in mW
        led_model: LED model identifier
        led_wavelength: LED wavelength in nm
        enable_entry_spheres: Whether to cut spherical entry surfaces
        entry_sphere_depth: Depth factor for entry spheres
        use_circle: Use circular profiles instead of polygons
        led_positions: Pre-computed list of all LED positions (for entry spheres)
        led_directions: Pre-computed list of all LED directions (for entry spheres)
        lens_material: Optional lens material properties dict
        absorber_material: Optional absorber material properties dict
        
    Returns:
        Individual with updated fitness scores
    """
    # Generate lens geometry (no debug spheres during evaluation for performance)
    if enable_entry_spheres:
        # Use pre-computed LED positions/directions (avoids FreeCAD document access)
        if led_positions is None:
            led_positions = [led_pos]
        if led_directions is None:
            led_directions = [led_dir]
        
        # Create base lens then cut entry spheres (no debug geometry)
        base_lens = create_swept_lens(individual.profiles, centerline_points, use_circle)
        if base_lens is not None:
            sorted_profiles = sorted(individual.profiles, key=lambda p: p.z_position)
            first_profile_radius = sorted_profiles[0].radius
            centerline_start = centerline_points[0].position
            lens_solid = cut_lens_entry_spheres(
                base_lens, led_positions, led_directions,
                first_profile_radius, centerline_start,
                entry_sphere_depth, debug_save_spheres=False
            )
        else:
            lens_solid = None
    else:
        lens_solid = create_swept_lens(
            individual.profiles,
            centerline_points,
            use_circle
        )
    
    if lens_solid is None:
        individual.is_valid = False
        individual.validation_error = "Lens generation failed"
        return individual
    
    # Export to STL for raytracing
    try:
        lens_mesh = solid_to_mesh(lens_solid, budget['mesh_resolution'])
        lens_stl = tempfile.mktemp(suffix='_lens.stl')
        export_mesh_to_stl(lens_mesh, lens_stl)
    except Exception as e:
        individual.is_valid = False
        individual.validation_error = f"Mesh export failed: {e}"
        return individual
    
    # Run raytracing (pass material properties for correct simulation)
    try:
        result = run_raytracer_evaluation(
            lens_stl, absorber_stl_path,
            led_pos, led_dir,
            num_rays=budget['rays'],
            max_bounces=DEFAULT_MAX_BOUNCES,
            led_power=led_power_mW,
            led_model=led_model,
            led_wavelength=led_wavelength,
            lens_material=lens_material,
            absorber_material=absorber_material
        )
    finally:
        try:
            os.remove(lens_stl)
        except:
            pass
    
    if result is None:
        individual.is_valid = False
        individual.validation_error = "Raytracing failed"
        FreeCAD.Console.PrintWarning(f"  Raytracing returned None for individual\n")
        return individual
    
    # Analyze results
    grid_analysis = analyze_uniformity_grid(
        result.get('exit_positions', []),
        result.get('exit_intensities', []),
        num_rays=budget['rays']
    )
    
    individual.uniformity = grid_analysis['uniformity_index']
    individual.efficiency = calculate_efficiency(result, budget['rays'])
    individual.lens_entry_rate = calculate_lens_entry_rate(result, budget['rays'])
    individual.absorber_capture_rate = calculate_absorber_capture_rate(result, budget['rays'])
    
    individual.fitness = calculate_fitness(
        individual.uniformity,
        individual.efficiency,
        individual.lens_entry_rate,
        individual.absorber_capture_rate
    )
    
    individual.is_valid = True
    return individual


def evolve_lens(led_obj, envelope_obj, absorber_obj, centerline_obj,
                budget_name: str = "Quick",
                population_size: int = DEFAULT_POPULATION_SIZE,
                generations: int = DEFAULT_GENERATIONS,
                num_profiles: int = DEFAULT_NUM_PROFILES,
                vary_profile_count: bool = False,
                led_power_mW: float = 1420.0,
                led_current_mA: float = 700.0,
                led_model: str = "U405",
                led_wavelength: int = 405,
                led1_direction: Tuple = None,
                led2_config: Dict = None,
                lens_material: Dict = None,
                absorber_material: Dict = None,
                enable_entry_spheres: bool = True,
                entry_sphere_depth: float = 0.3,
                use_circle: bool = False,
                initial_population: List = None,
                distribution_weight: float = 0.0,
                envelope_reduction: float = 0.0) -> Optional[Tuple[Part.Shape, Dict]]:
    """Main evolutionary lens optimization function.
    
    Args:
        led_obj: FreeCAD object for LED position
        envelope_obj: FreeCAD solid defining max lens volume
        absorber_obj: FreeCAD object defining absorber
        centerline_obj: Optional FreeCAD sketch with centerline curve
        budget_name: "Quick", "Medium", or "Thorough"
        population_size: Number of individuals per generation
        generations: Number of generations to evolve
        num_profiles: Number of cross-section profiles
        vary_profile_count: Allow profile count to change during evolution
        led_power_mW: LED radiant power in milliwatts
        led_current_mA: LED forward current in milliamps (for reporting)
        led_model: LED model identifier
        led_wavelength: LED wavelength in nm
        led1_direction: Optional LED direction tuple (x, y, z)
        led2_config: Optional configuration for second LED
        lens_material: Optional lens material properties
        absorber_material: Optional absorber material properties
        enable_entry_spheres: Whether to cut spherical LED entry surfaces
        entry_sphere_depth: Depth factor for entry spheres (0.0-0.5)
        use_circle: Use circular profiles instead of polygons
        initial_population: Optional list of Individual objects to seed population (for multi-phase)
        distribution_weight: Weight for spatial distribution in fitness
        envelope_reduction: Percentage to reduce envelope constraint
        
    Returns:
        Tuple of (optimized_solid, stats_dict) or None on failure
    """
    # Check for Python with PyCUDA FIRST - cancel if not found
    python_exe = find_python_executable()
    if python_exe is None:
        FreeCAD.Console.PrintError("\n" + "="*70 + "\n")
        FreeCAD.Console.PrintError("OPERATION CANCELLED: PyCUDA not available\n")
        FreeCAD.Console.PrintError("="*70 + "\n")
        FreeCAD.Console.PrintError("The raytracer requires a Python installation with PyCUDA.\n")
        FreeCAD.Console.PrintError("Please install the required packages and try again.\n")
        FreeCAD.Console.PrintError("="*70 + "\n")
        return None
    
    # Prevent system sleep
    prevent_sleep()
    
    # Start raytracer server - MUST succeed for optimization to work
    if USE_RAYTRACER_SERVER:
        server_started = start_raytracer_server(python_exe)
        if not server_started:
            FreeCAD.Console.PrintError("\n" + "="*70 + "\n")
            FreeCAD.Console.PrintError("OPERATION CANCELLED: Raytracer server failed to start\n")
            FreeCAD.Console.PrintError("="*70 + "\n")
            FreeCAD.Console.PrintError("Check the error messages above for details.\n")
            FreeCAD.Console.PrintError("="*70 + "\n")
            allow_sleep()
            return None
    else:
        FreeCAD.Console.PrintError("USE_RAYTRACER_SERVER is disabled - cannot run optimization\n")
        allow_sleep()
        return None
    
    budget = COMPUTE_BUDGETS[budget_name]
    
    # Track start time for total operation timing
    start_time = time.time()
    
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("EVOLUTIONARY LENS OPTIMIZER\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage(f"Budget: {budget_name}\n")
    FreeCAD.Console.PrintMessage(f"Population: {population_size}, Generations: {generations}\n")
    FreeCAD.Console.PrintMessage(f"Profiles: {num_profiles}\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n\n")
    
    # Get LED position and direction
    led_pos = get_led_position(led_obj)
    
    if led1_direction:
        led_dir = FreeCAD.Vector(led1_direction[0], led1_direction[1], led1_direction[2])
        led_dir.normalize()
    else:
        led_dir = get_led_direction_from_user()
        if led_dir is None:
            stop_raytracer_server()
            allow_sleep()
            return None
    
    FreeCAD.Console.PrintMessage(f"LED Position: ({led_pos.x:.2f}, {led_pos.y:.2f}, {led_pos.z:.2f})\n")
    FreeCAD.Console.PrintMessage(f"LED Direction: ({led_dir.x:.2f}, {led_dir.y:.2f}, {led_dir.z:.2f})\n\n")
    
    # Extract centerline
    FreeCAD.Console.PrintMessage("Extracting centerline...\n")
    if centerline_obj:
        centerline_points = extract_centerline_from_sketch(
            centerline_obj, envelope_obj.Shape, led_pos
        )
    else:
        centerline_points = extract_curved_centerline_auto(
            envelope_obj.Shape, led_pos, led_dir
        )
    
    if not centerline_points or len(centerline_points) < 2:
        FreeCAD.Console.PrintError("Failed to extract centerline!\n")
        stop_raytracer_server()
        allow_sleep()
        return None
    
    # Get max radii from centerline
    max_radii = [cp.max_radius for cp in centerline_points]
    
    # OPTIMIZATION: Pre-compute LED positions/directions ONCE (avoid repeated doc access)
    led_positions, led_directions = build_led_lists(led_pos, led_dir, led2_config)
    FreeCAD.Console.PrintMessage(f"LED configuration: {len(led_positions)} LED(s)\n")
    
    # Export absorber
    absorber_stl = tempfile.mktemp(suffix='_absorber.stl')
    absorber_mesh = solid_to_mesh(absorber_obj.Shape, budget.get('absorber_resolution', 1.0))
    export_mesh_to_stl(absorber_mesh, absorber_stl)
    keep_ui_responsive()  # Keep UI responsive after mesh export
    
    # Create initial population (seed with initial_population if provided, fill rest with random)
    FreeCAD.Console.PrintMessage(f"Creating initial population of {population_size}...\n")
    population = []
    
    # Seed with initial_population if provided (for multi-phase optimization)
    if initial_population:
        FreeCAD.Console.PrintMessage(f"  Seeding with {len(initial_population)} individuals from previous phase\n")
        for seed_individual in initial_population[:population_size]:
            # Copy and reset fitness (needs re-evaluation with new budget)
            individual = seed_individual.copy()
            individual.fitness = 0.0
            individual.is_valid = False
            population.append(individual)
    
    # Fill remainder with random individuals
    while len(population) < population_size:
        population.append(create_random_individual(num_profiles, max_radii))
    keep_ui_responsive()  # Keep UI responsive after population creation
    
    # Create result group
    doc = FreeCAD.ActiveDocument
    timestamp = datetime.now().strftime("%H%M%S")
    result_group = doc.addObject("App::DocumentObjectGroup", f"LensEvolution_{timestamp}")
    
    # Evolution tracking
    best_individual = None
    best_fitness = 0.0
    best_generation = 0  # Track which generation produced the best individual
    fitness_history = []
    consecutive_failures = 0  # Track consecutive generations with no valid individuals
    MAX_CONSECUTIVE_FAILURES = 3  # Abort if server seems dead
    
    # Progress dialog
    progress = None
    try:
        progress = show_progress_dialog("Evolving Lens Design")
    except:
        pass
    
    try:
        # Evolution loop
        for gen in range(generations):
            if progress:
                progress.setValue(int(gen / generations * 100))
                if progress.wasCanceled():
                    FreeCAD.Console.PrintMessage("\nEvolution cancelled by user\n")
                    break
            
            FreeCAD.Console.PrintMessage(f"\nGeneration {gen + 1}/{generations}:\n")
            
            # Evaluate population
            for i, individual in enumerate(population):
                if not individual.is_valid or individual.fitness == 0:
                    population[i] = evaluate_individual(
                        individual,
                        centerline_points,
                        led_pos, led_dir,
                        absorber_stl,
                        budget,
                        led_power_mW,
                        led_model,
                        led_wavelength,
                        enable_entry_spheres,
                        entry_sphere_depth,
                        use_circle,
                        led_positions=led_positions,  # Pre-computed (avoids doc access)
                        led_directions=led_directions,
                        lens_material=lens_material,
                        absorber_material=absorber_material
                    )
                    individual = population[i]
                    individual.generation = gen + 1
                    keep_ui_responsive()  # Keep UI responsive after each evaluation
            
            # Find best in generation
            valid_pop = [ind for ind in population if ind.is_valid]
            if valid_pop:
                consecutive_failures = 0  # Reset failure counter on success
                gen_best = max(valid_pop, key=lambda x: x.fitness)
                fitness_history.append(gen_best.fitness)
                
                FreeCAD.Console.PrintMessage(
                    f"  Best: Fitness={gen_best.fitness:.4f}, "
                    f"Uniformity={gen_best.uniformity:.4f}, "
                    f"Efficiency={gen_best.efficiency*100:.1f}%\n"
                )
                
                if gen_best.fitness > best_fitness:
                    best_fitness = gen_best.fitness
                    best_individual = gen_best.copy()
                    best_generation = gen + 1  # Track which generation produced best
                
                # Save best lens from this generation
                try:
                    gen_group = doc.addObject("App::DocumentObjectGroup", f"Gen_{gen+1:02d}")
                    result_group.addObject(gen_group)
                    
                    # Use pre-computed LED positions/directions (avoid redundant doc access)
                    # Create lens solid (include debug spheres in gen_group)
                    if enable_entry_spheres:
                        gen_lens = create_lens_with_entry_spheres(
                            gen_best.profiles, centerline_points,
                            led_positions, led_directions, entry_sphere_depth,
                            use_circle, result_group=gen_group
                        )
                    else:
                        gen_lens = create_swept_lens(gen_best.profiles, centerline_points, use_circle)
                    
                    if gen_lens:
                        lens_obj = doc.addObject("Part::Feature", f"BestLens_Gen{gen+1:02d}")
                        lens_obj.Shape = gen_lens
                        lens_obj.ViewObject.ShapeColor = (0.6, 0.8, 1.0)
                        lens_obj.ViewObject.Transparency = 70
                        lens_obj.ViewObject.Visibility = (gen == generations - 1)  # Only show last gen
                        gen_group.addObject(lens_obj)
                    
                    # OPTIMIZATION: Only call keep_ui_responsive, skip heavy recompute/updateGui
                    # (These will be called once at the end of evolution)
                    keep_ui_responsive()
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Could not save generation lens: {e}\n")
            else:
                consecutive_failures += 1
                FreeCAD.Console.PrintWarning(f"  No valid individuals! (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})\n")
                
                # Check if server has likely crashed
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    FreeCAD.Console.PrintError("\n" + "="*70 + "\n")
                    FreeCAD.Console.PrintError("OPTIMIZATION ABORTED: Raytracer server appears to have crashed\n")
                    FreeCAD.Console.PrintError("="*70 + "\n")
                    FreeCAD.Console.PrintError(f"No valid results for {consecutive_failures} consecutive generations.\n")
                    FreeCAD.Console.PrintError("The server may have run out of memory or encountered a CUDA error.\n")
                    FreeCAD.Console.PrintError("Please restart FreeCAD and try again with fewer rays or iterations.\n")
                    break
            
            # Evolve to next generation
            if gen < generations - 1:
                population = evolve_generation(
                    population, max_radii,
                    elite_count=DEFAULT_ELITE_COUNT
                )
    
    finally:
        if progress:
            progress.close()
        # Note: Don't delete absorber_stl yet - we need it for final visualization raytracing
        # Note: Don't stop server yet - we need it for final visualization raytracing
    
    # Generate final lens from best individual
    if best_individual is None:
        FreeCAD.Console.PrintError("No valid lens design found!\n")
        return None
    
    # OPTIMIZATION: Single recompute after all generations (instead of per-generation)
    doc.recompute()
    FreeCADGui.updateGui()
    
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("EVOLUTION COMPLETE\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage(f"Best Fitness: {best_individual.fitness:.4f}\n")
    FreeCAD.Console.PrintMessage(f"Best Uniformity: {best_individual.uniformity:.4f}\n")
    FreeCAD.Console.PrintMessage(f"Best Efficiency: {best_individual.efficiency*100:.1f}%\n")
    
    # Create final lens solid (use pre-computed LED lists)
    if enable_entry_spheres:
        final_solid = create_lens_with_entry_spheres(
            best_individual.profiles,
            centerline_points,
            led_positions,
            led_directions,
            entry_sphere_depth,
            use_circle,
            result_group=result_group
        )
    else:
        final_solid = create_swept_lens(
            best_individual.profiles,
            centerline_points,
            use_circle
        )
    
    # Get top individuals for potential multi-phase seeding
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    top_individuals = [ind.copy() for ind in sorted_population[:min(10, len(sorted_population))]]
    
    # Re-run raytracing on best lens to get full result for visualization
    # (This provides exit_positions and grid_analysis for heatmap/irradiance display)
    best_result = None
    best_grid_analysis = None
    
    if final_solid is not None:
        FreeCAD.Console.PrintMessage("Running final raytracing for visualization data...\n")
        try:
            # Export final lens to STL
            final_lens_mesh = solid_to_mesh(final_solid, budget['mesh_resolution'])
            final_lens_stl = tempfile.mktemp(suffix='_final_lens.stl')
            export_mesh_to_stl(final_lens_mesh, final_lens_stl)
            
            # Run raytracing for LED 1 (absorber_stl still exists from evolution)
            led1_result = run_raytracer_evaluation(
                final_lens_stl, absorber_stl,
                led_pos, led_dir,
                num_rays=budget['rays'],
                max_bounces=DEFAULT_MAX_BOUNCES,
                led_power=led_power_mW,
                led_model=led_model,
                led_wavelength=led_wavelength,
                lens_material=lens_material,
                absorber_material=absorber_material
            )
            
            keep_ui_responsive()  # Keep UI responsive after LED 1 raytracing
            
            # Structure LED 1 exits
            import numpy as np
            led1_exits = {
                'positions': np.array(led1_result.get('exit_positions', [])) if led1_result else np.array([]),
                'intensities': np.array(led1_result.get('exit_intensities', [])) if led1_result else np.array([]),
                'wavelength': led_wavelength
            }
            
            # Run raytracing for LED 2 if configured
            led2_exits = {'positions': np.array([]), 'intensities': np.array([]), 'wavelength': None}
            if led2_config is not None:
                led2_obj_name = led2_config.get('position_object')
                if led2_obj_name:
                    active_doc = FreeCAD.ActiveDocument
                    led2_obj = active_doc.getObject(led2_obj_name) if active_doc else None
                    if led2_obj:
                        led2_pos = get_led_position(led2_obj)
                        led2_dir_tuple = led2_config.get('direction', (0, 1, 0))
                        led2_dir = FreeCAD.Vector(led2_dir_tuple[0], led2_dir_tuple[1], led2_dir_tuple[2])
                        if led2_dir.Length > 0.001:
                            led2_dir.normalize()
                        led2_power = led2_config.get('power_mW', led_power_mW)
                        led2_model = led2_config.get('model', led_model)
                        led2_wavelength = led2_config.get('wavelength', led_wavelength)
                        
                        FreeCAD.Console.PrintMessage(f"  Running raytracing for LED 2 ({led2_wavelength}nm)...\n")
                        led2_result = run_raytracer_evaluation(
                            final_lens_stl, absorber_stl,
                            led2_pos, led2_dir,
                            num_rays=budget['rays'],
                            max_bounces=DEFAULT_MAX_BOUNCES,
                            led_power=led2_power,
                            led_model=led2_model,
                            led_wavelength=led2_wavelength,
                            lens_material=lens_material,
                            absorber_material=absorber_material
                        )
                        
                        keep_ui_responsive()  # Keep UI responsive after LED 2 raytracing
                        
                        if led2_result:
                            led2_exits = {
                                'positions': np.array(led2_result.get('exit_positions', [])),
                                'intensities': np.array(led2_result.get('exit_intensities', [])),
                                'wavelength': led2_wavelength
                            }
                            FreeCAD.Console.PrintMessage(f"    LED 2: {len(led2_exits['positions'])} absorber exits\n")
            
            # Cleanup lens temp file
            try:
                os.remove(final_lens_stl)
            except:
                pass
            
            # Combine results for best_result (for backward compatibility)
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
                combined_positions = np.array([])
                combined_intensities = np.array([])
            
            best_result = {
                'exit_positions': combined_positions,
                'exit_intensities': combined_intensities,
                'led1_exits': led1_exits,
                'led2_exits': led2_exits
            }
            
            # Analyze combined results for irradiance stats
            if len(combined_positions) > 0:
                best_grid_analysis = analyze_uniformity_grid(
                    combined_positions,
                    combined_intensities,
                    num_rays=budget['rays'] * (2 if len(led2_exits['positions']) > 0 else 1)
                )
                FreeCAD.Console.PrintMessage("  Visualization data ready\n")
                FreeCAD.Console.PrintMessage(f"    LED 1 ({led_wavelength}nm): {len(led1_exits['positions'])} absorber exits\n")
                if led2_exits['wavelength'] is not None:
                    FreeCAD.Console.PrintMessage(f"    LED 2 ({led2_exits['wavelength']}nm): {len(led2_exits['positions'])} absorber exits\n")
            else:
                FreeCAD.Console.PrintWarning("  Could not get visualization data\n")
                
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not generate visualization data: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Now cleanup absorber STL and stop server
    try:
        os.remove(absorber_stl)
    except:
        pass
    stop_raytracer_server()
    allow_sleep()
    
    # Calculate total operation time
    total_time_seconds = time.time() - start_time
    
    # Return results
    stats = {
        'best_fitness': best_individual.fitness,
        'best_uniformity': best_individual.uniformity,
        'best_efficiency': best_individual.efficiency,
        'best_lens_entry': best_individual.lens_entry_rate,
        'best_absorber_capture': best_individual.absorber_capture_rate,
        'generations': generations,
        'population_size': population_size,
        'fitness_history': fitness_history,
        'result_group': result_group,
        'lens_material': lens_material,
        'absorber_material': absorber_material,
        'best_individuals': top_individuals,  # For multi-phase optimization
        'best_result': best_result,  # For heatmap visualization
        'best_grid_analysis': best_grid_analysis,  # For irradiance display
        'num_rays': budget['rays'],  # For irradiance normalization in heatmap
        # New fields for enhanced report
        'best_generation': best_generation,  # Which generation produced best lens
        'total_time_seconds': total_time_seconds,  # Total operation time
        'budget_name': budget_name,  # Raytracing quality level
        'led1_model': led_model,  # LED 1 model identifier
        'led1_current_mA': led_current_mA,  # LED 1 current in mA
        'led1_wavelength': led_wavelength,  # LED 1 wavelength in nm
        'led1_power_mW': led_power_mW,  # LED 1 power in mW
        'led2_config': led2_config,  # LED 2 configuration (or None)
        'centerline_points': centerline_points,  # For geometry visualization
        'best_individual': best_individual,  # For profile data
        'use_circle': use_circle,  # Profile shape type
    }
    
    return final_solid, stats


def run_multi_phase_optimization(led_obj, envelope_obj, absorber_obj, centerline_obj,
                                  population_size: int,
                                  generations: int,
                                  num_profiles: int,
                                  led_power_mW: float,
                                  led_current_mA: float,
                                  led_model: str,
                                  led_wavelength: int,
                                  led1_direction: tuple,
                                  led2_config: dict,
                                  lens_material: dict,
                                  absorber_material: dict,
                                  enable_entry_spheres: bool,
                                  entry_sphere_depth: float,
                                  use_circle: bool,
                                  distribution_weight: float = 0.0,
                                  envelope_reduction: float = 0.0):
    """Run multi-phase optimization: Quick → Medium → Thorough.
    
    Each phase builds on the previous, carrying forward the best individuals
    as seeds for the next phase with higher raytracing accuracy.
    
    Phase 1 (Quick): Broad exploration with profile count variation
    Phase 2 (Medium): Refinement of promising designs  
    Phase 3 (Thorough): Final polish with fixed optimal profile count
    
    Args:
        All args same as evolve_lens()
        
    Returns:
        Tuple of (best_solid, combined_stats_dict) or None on failure
    """
    # Check for Python with PyCUDA FIRST - cancel if not found
    # (evolve_lens will also check, but we check early for cleaner error message)
    python_exe = find_python_executable()
    if python_exe is None:
        FreeCAD.Console.PrintError("\n" + "="*70 + "\n")
        FreeCAD.Console.PrintError("OPERATION CANCELLED: PyCUDA not available\n")
        FreeCAD.Console.PrintError("="*70 + "\n")
        FreeCAD.Console.PrintError("The raytracer requires a Python installation with PyCUDA.\n")
        FreeCAD.Console.PrintError("Please install the required packages and try again.\n")
        FreeCAD.Console.PrintError("="*70 + "\n")
        return None
    
    # Also verify server can start before showing multi-phase banner
    from .server_management import start_raytracer_server, stop_raytracer_server
    from .config import USE_RAYTRACER_SERVER
    
    if USE_RAYTRACER_SERVER:
        server_started = start_raytracer_server(python_exe)
        if not server_started:
            FreeCAD.Console.PrintError("\n" + "="*70 + "\n")
            FreeCAD.Console.PrintError("OPERATION CANCELLED: Raytracer server failed to start\n")
            FreeCAD.Console.PrintError("="*70 + "\n")
            FreeCAD.Console.PrintError("Check the error messages above for details.\n")
            FreeCAD.Console.PrintError("="*70 + "\n")
            return None
        # Stop it - evolve_lens will start its own
        stop_raytracer_server()
    
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("MULTI-PHASE OPTIMIZATION\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage("This will run 3 phases automatically:\n")
    FreeCAD.Console.PrintMessage("  Phase 1 (Quick): Explore design space, vary profile count\n")
    FreeCAD.Console.PrintMessage("  Phase 2 (Medium): Refine promising designs\n")
    FreeCAD.Console.PrintMessage("  Phase 3 (Thorough): Final polish with high accuracy\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n\n")
    
    all_phase_stats = []
    best_individuals_phase1 = []
    best_individuals_phase2 = []
    best_profile_count = num_profiles  # Will be updated based on Phase 1 results
    
    # =========================================================================
    # PHASE 1: Quick Exploration
    # =========================================================================
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("PHASE 1/3: Quick Exploration\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage("Goal: Explore design space with profile count variation\n\n")
    
    phase1_result = evolve_lens(
        led_obj, envelope_obj, absorber_obj, centerline_obj,
        budget_name="Quick",
        population_size=population_size,
        generations=generations,
        num_profiles=num_profiles,
        vary_profile_count=True,  # Enable profile variation for exploration
        led_power_mW=led_power_mW,
        led_current_mA=led_current_mA,
        led_model=led_model,
        led_wavelength=led_wavelength,
        led1_direction=led1_direction,
        led2_config=led2_config,
        lens_material=lens_material,
        absorber_material=absorber_material,
        enable_entry_spheres=enable_entry_spheres,
        entry_sphere_depth=entry_sphere_depth,
        use_circle=use_circle,
        initial_population=None,  # Start fresh
        distribution_weight=distribution_weight,
        envelope_reduction=envelope_reduction
    )
    
    if phase1_result is None:
        FreeCAD.Console.PrintError("Phase 1 failed! Aborting multi-phase optimization.\n")
        return None
    
    phase1_solid, phase1_stats = phase1_result
    all_phase_stats.append(('Phase 1 (Quick)', phase1_stats))
    
    # Extract best individuals from Phase 1
    if 'best_individuals' in phase1_stats:
        best_individuals_phase1 = phase1_stats['best_individuals'][:5]  # Top 5
        # Determine best profile count from Phase 1 winners
        if best_individuals_phase1:
            profile_counts = [len(ind.profiles) for ind in best_individuals_phase1]
            best_profile_count = max(set(profile_counts), key=profile_counts.count)  # Mode
    
    FreeCAD.Console.PrintMessage(f"\nPhase 1 Complete: Best fitness = {phase1_stats.get('best_fitness', 0):.4f}\n")
    FreeCAD.Console.PrintMessage(f"Best profile count discovered: {best_profile_count}\n")
    FreeCAD.Console.PrintMessage(f"Transferring top {len(best_individuals_phase1)} individuals to Phase 2...\n\n")
    
    # =========================================================================
    # PHASE 2: Medium Refinement
    # =========================================================================
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("PHASE 2/3: Medium Refinement\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage("Goal: Refine promising designs with better ray accuracy\n\n")
    
    phase2_generations = int(generations * 1.5)  # 50% more generations
    
    phase2_result = evolve_lens(
        led_obj, envelope_obj, absorber_obj, centerline_obj,
        budget_name="Medium",
        population_size=population_size,
        generations=phase2_generations,
        num_profiles=best_profile_count,  # Use discovered optimal count
        vary_profile_count=True,  # Still allow some variation
        led_power_mW=led_power_mW,
        led_current_mA=led_current_mA,
        led_model=led_model,
        led_wavelength=led_wavelength,
        led1_direction=led1_direction,
        led2_config=led2_config,
        lens_material=lens_material,
        absorber_material=absorber_material,
        enable_entry_spheres=enable_entry_spheres,
        entry_sphere_depth=entry_sphere_depth,
        use_circle=use_circle,
        initial_population=best_individuals_phase1,  # Seed with Phase 1 winners
        distribution_weight=distribution_weight,
        envelope_reduction=envelope_reduction
    )
    
    if phase2_result is None:
        FreeCAD.Console.PrintError("Phase 2 failed! Returning Phase 1 result.\n")
        return phase1_result
    
    phase2_solid, phase2_stats = phase2_result
    all_phase_stats.append(('Phase 2 (Medium)', phase2_stats))
    
    # Extract best individuals from Phase 2
    if 'best_individuals' in phase2_stats:
        best_individuals_phase2 = phase2_stats['best_individuals'][:3]  # Top 3
        # Update best profile count if improved
        if best_individuals_phase2:
            profile_counts = [len(ind.profiles) for ind in best_individuals_phase2]
            best_profile_count = max(set(profile_counts), key=profile_counts.count)
    
    FreeCAD.Console.PrintMessage(f"\nPhase 2 Complete: Best fitness = {phase2_stats.get('best_fitness', 0):.4f}\n")
    FreeCAD.Console.PrintMessage(f"Final profile count: {best_profile_count}\n")
    FreeCAD.Console.PrintMessage(f"Transferring top {len(best_individuals_phase2)} individuals to Phase 3...\n\n")
    
    # =========================================================================
    # PHASE 3: Thorough Polish
    # =========================================================================
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("PHASE 3/3: Thorough Polish\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage("Goal: Final optimization with maximum ray accuracy\n\n")
    
    phase3_generations = int(generations * 2)  # Double generations for thorough polish
    
    phase3_result = evolve_lens(
        led_obj, envelope_obj, absorber_obj, centerline_obj,
        budget_name="Thorough",
        population_size=population_size,
        generations=phase3_generations,
        num_profiles=best_profile_count,  # Fixed to optimal count
        vary_profile_count=False,  # No more variation - focus on radius refinement
        led_power_mW=led_power_mW,
        led_current_mA=led_current_mA,
        led_model=led_model,
        led_wavelength=led_wavelength,
        led1_direction=led1_direction,
        led2_config=led2_config,
        lens_material=lens_material,
        absorber_material=absorber_material,
        enable_entry_spheres=enable_entry_spheres,
        entry_sphere_depth=entry_sphere_depth,
        use_circle=use_circle,
        initial_population=best_individuals_phase2,  # Seed with Phase 2 winners
        distribution_weight=distribution_weight,
        envelope_reduction=envelope_reduction
    )
    
    if phase3_result is None:
        FreeCAD.Console.PrintError("Phase 3 failed! Returning Phase 2 result.\n")
        return phase2_result
    
    phase3_solid, phase3_stats = phase3_result
    all_phase_stats.append(('Phase 3 (Thorough)', phase3_stats))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("MULTI-PHASE OPTIMIZATION COMPLETE\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    for phase_name, stats in all_phase_stats:
        FreeCAD.Console.PrintMessage(f"  {phase_name}: Best fitness = {stats.get('best_fitness', 0):.4f}, "
                                     f"Efficiency = {stats.get('best_efficiency', 0)*100:.1f}%\n")
    FreeCAD.Console.PrintMessage(f"\nFinal Result: {best_profile_count} profiles, "
                                 f"Fitness = {phase3_stats.get('best_fitness', 0):.4f}\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    
    # Combine stats from all phases
    combined_stats = phase3_stats.copy()
    combined_stats['multi_phase'] = True
    combined_stats['phase_history'] = all_phase_stats
    combined_stats['final_profile_count'] = best_profile_count
    
    return (phase3_solid, combined_stats)
