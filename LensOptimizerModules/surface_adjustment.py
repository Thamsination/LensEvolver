"""
Surface adjustment functions for the Lens Optimizer.

This module provides functions to calculate mesh deformations based on
uniformity analysis results.
"""

import numpy as np
import FreeCAD

from .mesh_operations import compute_vertex_normals


def calculate_deformation(vertices, faces, grid_analysis, learning_rate=0.5, led_pos=None):
    """Calculate vertex deformation to improve uniformity.
    
    Based on grid analysis results, calculates how much to move each vertex
    to redirect light from hot zones to cold zones.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        grid_analysis: Results from analyze_uniformity_grid
        learning_rate: How much to adjust (0-1)
        led_pos: Optional LED position for directional adjustment
        
    Returns:
        Nx3 array of deformation vectors
    """
    n_verts = len(vertices)
    deformation = np.zeros_like(vertices)
    
    # Get grid analysis results
    hot_zones = grid_analysis.get('hot_zones', [])
    cold_zones = grid_analysis.get('cold_zones', [])
    mean_intensity = grid_analysis.get('mean_intensity', 1.0)
    
    if len(hot_zones) == 0 and len(cold_zones) == 0:
        return deformation
    
    # Compute vertex normals for deformation direction
    normals = compute_vertex_normals(vertices, faces)
    
    # Get mesh center for radial reference
    mesh_center = np.mean(vertices, axis=0)
    
    # Process hot zones - push surface inward to spread light
    for hot in hot_zones:
        world_pos = hot['world_pos']
        excess = hot.get('excess', 0.1)
        
        # Find vertices near this hot zone
        for i, v in enumerate(vertices):
            dist = np.sqrt((v[0] - world_pos[0])**2 + (v[1] - world_pos[1])**2)
            
            if dist < 5.0:  # Within influence radius
                # Calculate deformation magnitude based on distance and excess
                influence = 1.0 - (dist / 5.0)
                magnitude = learning_rate * excess * influence * 0.5
                
                # Push inward along normal
                deformation[i] -= normals[i] * magnitude
    
    # Process cold zones - pull surface outward to direct more light
    for cold in cold_zones:
        world_pos = cold['world_pos']
        deficit = cold.get('deficit', 0.1)
        
        # Find vertices near this cold zone
        for i, v in enumerate(vertices):
            dist = np.sqrt((v[0] - world_pos[0])**2 + (v[1] - world_pos[1])**2)
            
            if dist < 5.0:  # Within influence radius
                # Calculate deformation magnitude
                influence = 1.0 - (dist / 5.0)
                magnitude = learning_rate * deficit * influence * 0.5
                
                # Pull outward along normal
                deformation[i] += normals[i] * magnitude
    
    return deformation


def apply_deformation_safely(vertices, faces, deformation, envelope_vertices, 
                             envelope_faces, envelope_center, 
                             max_displacement=0.3, mirror_map=None):
    """Apply deformation while maintaining mesh validity.
    
    Clamps deformation to prevent self-intersection and ensures
    result stays within envelope bounds.
    
    Args:
        vertices: Nx3 array of current vertex positions
        faces: Mx3 array of face indices
        deformation: Nx3 array of deformation vectors
        envelope_vertices: Nx3 array of envelope vertex positions
        envelope_faces: Mx3 array of envelope face indices
        envelope_center: Center point of envelope
        max_displacement: Maximum allowed displacement per vertex
        mirror_map: Optional dict for maintaining symmetry
        
    Returns:
        Nx3 array of new vertex positions
    """
    # Clamp deformation magnitude
    magnitudes = np.linalg.norm(deformation, axis=1, keepdims=True)
    scale = np.where(magnitudes > max_displacement, max_displacement / magnitudes, 1.0)
    clamped_deformation = deformation * scale
    
    # Apply deformation
    new_vertices = vertices + clamped_deformation
    
    # Apply symmetry if mirror map provided
    if mirror_map is not None:
        for y_plus_idx, y_minus_idx in mirror_map.items():
            # Average the deformation for paired vertices
            avg_def = (clamped_deformation[y_plus_idx] + clamped_deformation[y_minus_idx]) / 2
            
            # Apply with mirrored Y component
            new_vertices[y_plus_idx] = vertices[y_plus_idx] + avg_def
            new_vertices[y_minus_idx] = vertices[y_minus_idx] + avg_def
            new_vertices[y_minus_idx, 1] = -new_vertices[y_plus_idx, 1]  # Mirror Y
    
    # Clamp to envelope bounds
    from .geometry_constraints import clamp_to_envelope
    new_vertices = clamp_to_envelope(new_vertices, envelope_vertices, envelope_center)
    
    return new_vertices


def build_ray_influence_map(exit_positions, ray_paths, vertices, faces):
    """Build mapping from vertices to rays that passed near them.
    
    This allows adjusting specific surface regions based on
    which rays contributed to hot/cold zones.
    
    Args:
        exit_positions: Nx3 array of ray exit positions
        ray_paths: List of ray path arrays
        vertices: Mx3 array of mesh vertices
        faces: Kx3 array of face indices
        
    Returns:
        Dict mapping vertex indices to contributing ray indices
    """
    vertex_rays = {i: [] for i in range(len(vertices))}
    
    # For each ray, find vertices it passed near
    for ray_idx, path in enumerate(ray_paths):
        if len(path) < 2:
            continue
        
        # Sample points along ray path
        for point_idx in range(len(path)):
            point = np.array(path[point_idx])
            
            # Find nearest vertices
            distances = np.linalg.norm(vertices - point, axis=1)
            nearby = np.where(distances < 2.0)[0]  # 2mm influence radius
            
            for v_idx in nearby:
                if ray_idx not in vertex_rays[v_idx]:
                    vertex_rays[v_idx].append(ray_idx)
    
    return vertex_rays
