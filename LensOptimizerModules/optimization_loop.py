"""
Main optimization loop for the Lens Optimizer (legacy mesh-based).

This module provides the legacy optimize_lens function that uses
mesh deformation for optimization.
"""

import os
import tempfile
from datetime import datetime
import numpy as np
import FreeCAD
import FreeCADGui

from .config import (
    COMPUTE_BUDGETS,
    DEFAULT_MAX_BOUNCES,
    DEFAULT_STARTING_SCALE,
    DEFAULT_SUBDIVISION_ITERATIONS,
    USE_RAYTRACER_SERVER
)
from .sleep_prevention import prevent_sleep, allow_sleep
from .server_management import start_raytracer_server, stop_raytracer_server
from .raytracer_integration import find_python_executable, run_raytracer_evaluation
from .mesh_operations import (
    solid_to_mesh, numpy_to_mesh, export_mesh_to_stl, scale_mesh_from_center
)
from .mesh_symmetry import create_symmetric_mesh
from .uniformity_analysis import analyze_uniformity_grid, calculate_fitness
from .user_interface import get_led_position, get_led_direction_from_user, show_progress_dialog
from .visualization import add_iteration_analytics


try:
    import trimesh
except ImportError:
    trimesh = None


def optimize_lens(led_obj, envelope_obj, absorber_obj, budget_name="Quick"):
    """Main lens optimization function using mesh deformation.
    
    This is the legacy optimization approach that directly deforms a mesh
    based on raytracing feedback.
    
    Args:
        led_obj: FreeCAD object for LED position
        envelope_obj: FreeCAD solid defining max lens volume
        absorber_obj: FreeCAD object defining absorber
        budget_name: "Quick", "Medium", or "Thorough"
        
    Returns:
        Optimized FreeCAD solid, or None on failure
    """
    # Prevent system sleep during optimization
    prevent_sleep()
    
    # Start raytracer server for faster evaluation
    python_exe = find_python_executable()
    if python_exe and USE_RAYTRACER_SERVER:
        start_raytracer_server(python_exe)
    
    budget = COMPUTE_BUDGETS[budget_name]
    
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("LENS GEOMETRY OPTIMIZER\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage(f"Budget: {budget_name} - {budget['description']}\n")
    FreeCAD.Console.PrintMessage(f"  Rays per iteration: {budget['rays']}\n")
    FreeCAD.Console.PrintMessage(f"  Max iterations: {budget['iterations']}\n")
    FreeCAD.Console.PrintMessage(f"  Lens mesh resolution: {budget['mesh_resolution']} mm\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n\n")
    
    # Get LED position
    led_pos = get_led_position(led_obj)
    FreeCAD.Console.PrintMessage(f"LED Position: ({led_pos.x:.2f}, {led_pos.y:.2f}, {led_pos.z:.2f})\n")
    
    # Get LED direction from user
    led_dir = get_led_direction_from_user()
    if led_dir is None:
        FreeCAD.Console.PrintMessage("Operation cancelled by user.\n")
        stop_raytracer_server()
        allow_sleep()
        return None
    FreeCAD.Console.PrintMessage(f"LED Direction: ({led_dir.x:.2f}, {led_dir.y:.2f}, {led_dir.z:.2f})\n\n")
    
    # Create symmetric mesh by construction
    FreeCAD.Console.PrintMessage("Creating symmetric mesh by construction...\n")
    vertices, faces, mirror_map = create_symmetric_mesh(
        envelope_obj.Shape, budget['mesh_resolution']
    )
    
    # Subdivide mesh for better control
    if DEFAULT_SUBDIVISION_ITERATIONS > 0 and trimesh is not None:
        temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        for _ in range(DEFAULT_SUBDIVISION_ITERATIONS):
            temp_mesh = temp_mesh.subdivide()
        vertices = np.array(temp_mesh.vertices)
        faces = np.array(temp_mesh.faces)
        
        # Rebuild mirror map for subdivided mesh
        centerline_tol = budget['mesh_resolution'] * 0.25
        new_mirror_map = {}
        
        for i, v in enumerate(vertices):
            if abs(v[1]) <= centerline_tol:
                continue
            elif v[1] > 0:
                target = np.array([v[0], -v[1], v[2]])
                distances = np.linalg.norm(vertices - target, axis=1)
                j = np.argmin(distances)
                if distances[j] < centerline_tol * 2:
                    new_mirror_map[i] = j
        
        mirror_map = new_mirror_map
        FreeCAD.Console.PrintMessage(f"    Subdivided mesh: {len(vertices)} vertices, {len(faces)} faces\n")
    
    FreeCAD.Console.PrintMessage(f"  Final mesh: {len(vertices)} vertices, {len(faces)} faces\n")
    
    # Save original envelope for constraints
    envelope_vertices = vertices.copy()
    envelope_faces = faces.copy()
    envelope_center = np.mean(envelope_vertices, axis=0)
    
    # Scale down starting geometry
    vertices = scale_mesh_from_center(vertices, DEFAULT_STARTING_SCALE)
    
    # Apply initial symmetry
    for y_plus_idx, y_minus_idx in mirror_map.items():
        vertices[y_minus_idx, 0] = vertices[y_plus_idx, 0]
        vertices[y_minus_idx, 1] = -vertices[y_plus_idx, 1]
        vertices[y_minus_idx, 2] = vertices[y_plus_idx, 2]
    
    FreeCAD.Console.PrintMessage(f"  Starting from {DEFAULT_STARTING_SCALE*100:.0f}% scaled envelope\n\n")
    
    # Export absorber to STL
    absorber_stl = tempfile.mktemp(suffix='_absorber.stl')
    absorber_resolution = budget.get('absorber_resolution', 1.0)
    absorber_mesh = solid_to_mesh(absorber_obj.Shape, absorber_resolution)
    export_mesh_to_stl(absorber_mesh, absorber_stl)
    
    # Initialize tracking
    best_fitness = 0
    best_uniformity = 0
    best_efficiency = 0
    best_vertices = vertices.copy()
    best_iteration = 0
    uniformity_history = []
    efficiency_history = []
    fitness_history = []
    
    # Create result group
    doc = FreeCAD.ActiveDocument
    timestamp = datetime.now().strftime("%H%M%S")
    result_group_name = f"LensOpt_{budget_name}_{timestamp}"
    result_group = doc.addObject("App::DocumentObjectGroup", result_group_name)
    
    # Create progress dialog
    progress = None
    try:
        progress = show_progress_dialog("Optimizing Lens Geometry")
    except:
        pass
    
    try:
        # Optimization loop
        for iteration in range(budget['iterations']):
            # Update progress
            if progress:
                progress.setValue(int(iteration / budget['iterations'] * 100))
                if progress.wasCanceled():
                    FreeCAD.Console.PrintMessage("\nOptimization cancelled by user\n")
                    break
            
            # Export current lens mesh to STL
            current_mesh = numpy_to_mesh(vertices, faces)
            lens_stl = tempfile.mktemp(suffix='_lens.stl')
            export_mesh_to_stl(current_mesh, lens_stl)
            
            # Run raytracer evaluation
            FreeCAD.Console.PrintMessage(f"Iteration {iteration + 1}/{budget['iterations']}: ")
            
            result = run_raytracer_evaluation(
                lens_stl, absorber_stl,
                led_pos, led_dir,
                num_rays=budget['rays'],
                max_bounces=DEFAULT_MAX_BOUNCES
            )
            
            # Clean up temp file
            try:
                os.remove(lens_stl)
            except:
                pass
            
            if result is None:
                FreeCAD.Console.PrintMessage("Raytracer failed\n")
                continue
            
            # Analyze uniformity
            grid_analysis = analyze_uniformity_grid(
                result['exit_positions'],
                result['exit_intensities'],
                num_rays=budget['rays']
            )
            
            uniformity = grid_analysis['uniformity_index']
            efficiency = len(result['exit_positions']) / budget['rays']
            fitness = calculate_fitness(uniformity, efficiency)
            
            uniformity_history.append(uniformity)
            efficiency_history.append(efficiency)
            fitness_history.append(fitness)
            
            FreeCAD.Console.PrintMessage(
                f"Fitness={fitness:.4f}, Uniformity={uniformity:.4f}, "
                f"Efficiency={efficiency*100:.1f}%\n"
            )
            
            # Track best result
            if fitness > best_fitness:
                best_fitness = fitness
                best_uniformity = uniformity
                best_efficiency = efficiency
                best_vertices = vertices.copy()
                best_iteration = iteration + 1
            
            # Check convergence
            if iteration > 5:
                recent_improvement = max(fitness_history[-5:]) - min(fitness_history[-5:])
                if recent_improvement < budget['convergence_threshold']:
                    FreeCAD.Console.PrintMessage(
                        f"\nConverged (improvement {recent_improvement:.4f} < "
                        f"threshold {budget['convergence_threshold']})\n"
                    )
                    break
    
    finally:
        if progress:
            progress.close()
        
        # Clean up
        try:
            os.remove(absorber_stl)
        except:
            pass
        
        stop_raytracer_server()
        allow_sleep()
    
    # Report results
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("OPTIMIZATION COMPLETE\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n")
    FreeCAD.Console.PrintMessage(f"Best Iteration: {best_iteration}\n")
    FreeCAD.Console.PrintMessage(f"Best Fitness: {best_fitness:.4f}\n")
    FreeCAD.Console.PrintMessage(f"Best Uniformity: {best_uniformity:.4f}\n")
    FreeCAD.Console.PrintMessage(f"Best Efficiency: {best_efficiency*100:.1f}%\n")
    
    # Create final mesh from best vertices
    from .mesh_operations import mesh_to_solid
    final_mesh = numpy_to_mesh(best_vertices, faces)
    final_solid = mesh_to_solid(final_mesh)
    
    return final_solid
