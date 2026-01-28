"""
Mesh symmetry operations for the Lens Optimizer.

This module provides functions to create and maintain Y-axis symmetric meshes
for lens optimization.
"""

import numpy as np
import FreeCAD
import Part

from .mesh_operations import solid_to_mesh, mesh_to_numpy


def make_mesh_symmetric(vertices, faces, tolerance=0.01):
    """Rebuild mesh to be perfectly symmetric across Y=0 (X-Z plane).
    
    Strategy (Half-Mesh Mirroring):
    1. Identify centerline vertices (Y ≈ 0) and Y+ vertices
    2. Keep ONLY faces that are entirely in Y+ region (including centerline)
    3. Create mirrored Y- vertices from Y+ vertices
    4. Create mirrored Y- faces from Y+ faces (with reversed winding)
    5. Combine to form a closed, symmetric mesh
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        tolerance: Y values within this range are considered centerline
        
    Returns:
        new_vertices: Symmetric vertex array
        new_faces: Symmetric face array (Y+ faces + mirrored Y- faces)
        mirror_map: Dict mapping Y+ vertex indices to their Y- mirror indices
    """
    # Step 1: Classify original vertices
    y_coords = vertices[:, 1]
    
    centerline_mask = np.abs(y_coords) <= tolerance
    y_positive_mask = y_coords > tolerance
    y_negative_mask = y_coords < -tolerance
    
    centerline_indices = np.where(centerline_mask)[0]
    y_positive_indices = np.where(y_positive_mask)[0]
    y_negative_indices = np.where(y_negative_mask)[0]
    
    # Create set for fast lookup
    centerline_set = set(centerline_indices)
    y_positive_set = set(y_positive_indices)
    y_negative_set = set(y_negative_indices)
    
    # Step 2: Build new vertex array
    # Centerline vertices (force Y=0)
    centerline_verts = vertices[centerline_indices].copy()
    if len(centerline_verts) > 0:
        centerline_verts[:, 1] = 0.0
    
    # Y+ vertices
    y_positive_verts = vertices[y_positive_indices].copy()
    
    # Mirrored Y- vertices (copy Y+ and flip Y sign)
    y_negative_verts = y_positive_verts.copy()
    if len(y_negative_verts) > 0:
        y_negative_verts[:, 1] = -y_negative_verts[:, 1]
    
    # Combine into new vertex array
    if len(centerline_verts) > 0 and len(y_positive_verts) > 0:
        new_vertices = np.vstack([centerline_verts, y_positive_verts, y_negative_verts])
    elif len(centerline_verts) > 0:
        new_vertices = centerline_verts.copy()
    elif len(y_positive_verts) > 0:
        new_vertices = np.vstack([y_positive_verts, y_negative_verts])
    else:
        FreeCAD.Console.PrintWarning("    No Y+ or centerline vertices found!\n")
        return vertices.copy(), faces.copy(), {}
    
    # Step 3: Create index mapping from OLD indices to NEW indices
    old_to_new = {}
    
    # Centerline: old index -> new index (0 to len(centerline)-1)
    for i, old_idx in enumerate(centerline_indices):
        old_to_new[old_idx] = i
    
    # Y+ vertices: old index -> new index
    offset_ypos = len(centerline_indices)
    for i, old_idx in enumerate(y_positive_indices):
        old_to_new[old_idx] = offset_ypos + i
    
    # Step 4: Create mirror_map (new Y+ index -> new Y- index)
    mirror_map = {}
    offset_yneg = len(centerline_indices) + len(y_positive_indices)
    
    # Centerline vertices map to themselves
    for i in range(len(centerline_indices)):
        mirror_map[i] = i
    
    # Y+ vertices map to their Y- mirrors
    for i in range(len(y_positive_indices)):
        ypos_new_idx = offset_ypos + i
        yneg_new_idx = offset_yneg + i
        mirror_map[ypos_new_idx] = yneg_new_idx
    
    # Step 5: Filter faces - keep ONLY faces with ALL vertices in Y+ region
    ypos_faces = []
    
    for face in faces:
        all_in_ypos_region = True
        for v_idx in face:
            if v_idx in y_negative_set:
                all_in_ypos_region = False
                break
        
        if all_in_ypos_region:
            ypos_faces.append(face)
    
    # Step 6: Remap Y+ faces to new vertex indices
    new_ypos_faces = []
    for face in ypos_faces:
        try:
            new_face = [old_to_new[v_idx] for v_idx in face]
            new_ypos_faces.append(new_face)
        except KeyError as e:
            FreeCAD.Console.PrintWarning(f"    Missing vertex mapping for Y+ face: {e}\n")
            continue
    
    # Step 7: Create mirrored Y- faces from Y+ faces
    new_yneg_faces = []
    
    for face in new_ypos_faces:
        mirror_face = []
        for v_idx in face:
            mirror_idx = mirror_map.get(v_idx, v_idx)
            mirror_face.append(mirror_idx)
        
        # Reverse winding order for correct outward-facing normals
        mirror_face = [mirror_face[0], mirror_face[2], mirror_face[1]]
        
        # Only add if this face is NOT entirely on centerline
        if not all(v_idx < len(centerline_indices) for v_idx in face):
            new_yneg_faces.append(mirror_face)
    
    # Step 8: Combine Y+ faces and mirrored Y- faces
    all_faces = new_ypos_faces + new_yneg_faces
    new_faces = np.array(all_faces)
    
    FreeCAD.Console.PrintMessage(
        f"    Made mesh symmetric: {len(vertices)} -> {len(new_vertices)} vertices, "
        f"{len(faces)} -> {len(new_faces)} faces\n"
    )
    
    return new_vertices, new_faces, mirror_map


def create_symmetric_mesh(envelope_shape, mesh_resolution):
    """Create a perfectly symmetric mesh by construction.
    
    This function guarantees perfect Y-axis symmetry by:
    1. Cutting the envelope at Y=0 plane (keeping Y+ half)
    2. Meshing the Y+ half
    3. Mirroring vertices to create Y- half
    4. Creating mirrored faces with reversed winding
    5. Combining into single mesh with perfect 1:1 vertex pairing
    
    Args:
        envelope_shape: FreeCAD Part.Shape of the envelope
        mesh_resolution: Mesh resolution in mm
        
    Returns:
        Tuple of (vertices, faces, mirror_map)
    """
    # Get bounding box to determine cutting box size
    bbox = envelope_shape.BoundBox
    size = max(bbox.XLength, bbox.YLength, bbox.ZLength) * 2
    
    # Create a large box covering Y < 0 region
    cutting_box = Part.makeBox(size, size, size)
    cutting_box.translate(FreeCAD.Vector(
        bbox.XMin - size/4,
        -size,
        bbox.ZMin - size/4
    ))
    
    # Cut envelope to get Y+ half only
    try:
        y_plus_half = envelope_shape.cut(cutting_box)
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Cut operation failed: {e}\n")
        FreeCAD.Console.PrintWarning("Falling back to full envelope mesh\n")
        full_mesh = solid_to_mesh(envelope_shape, mesh_resolution)
        vertices, faces = mesh_to_numpy(full_mesh)
        mirror_map = {i: i for i in range(len(vertices))}
        return vertices, faces, mirror_map
    
    # Mesh the Y+ half
    half_mesh = solid_to_mesh(y_plus_half, mesh_resolution)
    half_vertices, half_faces = mesh_to_numpy(half_mesh)
    
    n_half_vertices = len(half_vertices)
    n_half_faces = len(half_faces)
    
    FreeCAD.Console.PrintMessage(f"    Y+ half mesh: {n_half_vertices} vertices, {n_half_faces} faces\n")
    
    # Identify vertices on the centerline (Y ≈ 0)
    centerline_tolerance = mesh_resolution * 0.5
    is_centerline = np.abs(half_vertices[:, 1]) < centerline_tolerance
    centerline_indices = np.where(is_centerline)[0]
    non_centerline_indices = np.where(~is_centerline)[0]
    
    n_centerline = len(centerline_indices)
    n_non_centerline = len(non_centerline_indices)
    
    FreeCAD.Console.PrintMessage(f"    Centerline vertices: {n_centerline}, Y+ only: {n_non_centerline}\n")
    
    # Build mapping from old indices to new indices
    old_to_new = {}
    new_vertices_list = []
    
    # First, add centerline vertices (forced to Y=0)
    for new_idx, old_idx in enumerate(centerline_indices):
        old_to_new[old_idx] = new_idx
        v = half_vertices[old_idx].copy()
        v[1] = 0.0
        new_vertices_list.append(v)
    
    n_shared = len(new_vertices_list)
    
    # Next, add non-centerline Y+ vertices
    for old_idx in non_centerline_indices:
        new_idx = len(new_vertices_list)
        old_to_new[old_idx] = new_idx
        new_vertices_list.append(half_vertices[old_idx].copy())
    
    # Finally, add mirrored Y- vertices
    mirror_map = {}
    
    for i, old_idx in enumerate(non_centerline_indices):
        y_plus_new_idx = old_to_new[old_idx]
        y_minus_new_idx = len(new_vertices_list)
        
        mirrored = half_vertices[old_idx].copy()
        mirrored[1] = -mirrored[1]
        new_vertices_list.append(mirrored)
        
        mirror_map[y_plus_new_idx] = y_minus_new_idx
    
    new_vertices = np.array(new_vertices_list)
    
    # Build set of centerline vertex indices
    centerline_vertex_set = set(range(n_shared))
    
    # Remap faces to use new vertex indices
    new_faces_list = []
    n_centerline_faces_removed = 0
    
    # Add original Y+ faces with remapped indices
    for face in half_faces:
        new_face = [old_to_new[idx] for idx in face]
        
        if all(v_idx in centerline_vertex_set for v_idx in new_face):
            n_centerline_faces_removed += 1
            continue
            
        new_faces_list.append(new_face)
    
    # Add mirrored Y- faces
    for face in half_faces:
        new_face = [old_to_new[idx] for idx in face]
        
        if all(v_idx in centerline_vertex_set for v_idx in new_face):
            continue
        
        mirrored_face = []
        for old_idx in face:
            new_idx = old_to_new[old_idx]
            if new_idx in mirror_map:
                mirrored_face.append(mirror_map[new_idx])
            else:
                mirrored_face.append(new_idx)
            
        mirrored_face = mirrored_face[::-1]
        new_faces_list.append(mirrored_face)
    
    if n_centerline_faces_removed > 0:
        FreeCAD.Console.PrintMessage(f"    Removed {n_centerline_faces_removed} internal centerline faces\n")
    
    new_faces = np.array(new_faces_list)
    
    FreeCAD.Console.PrintMessage(
        f"    Symmetric mesh: {len(new_vertices)} vertices, {len(new_faces)} faces\n"
    )
    FreeCAD.Console.PrintMessage(
        f"    Mirror pairs: {len(mirror_map)} (perfect 1:1 mapping)\n"
    )
    
    return new_vertices, new_faces, mirror_map


def symmetrize_deformation(deformation, mirror_pairs):
    """Make deformation array symmetric using mirror pairs.
    
    Args:
        deformation: Nx3 array of deformation vectors
        mirror_pairs: Dict with 'pairs' list of (pos_idx, neg_idx) tuples
        
    Returns:
        Symmetrized deformation array
    """
    result = deformation.copy()
    
    for pos_idx, neg_idx in mirror_pairs.get('pairs', []):
        # Average the deformation magnitudes
        avg_def = (deformation[pos_idx] + deformation[neg_idx]) / 2.0
        
        # Y component should be mirrored (opposite sign)
        result[pos_idx] = avg_def.copy()
        result[neg_idx] = avg_def.copy()
        result[neg_idx, 1] = -avg_def[1]
    
    return result


def apply_symmetry(vertices, mirror_map):
    """Apply symmetry to vertices using mirror map.
    
    Args:
        vertices: Nx3 array of vertex positions
        mirror_map: Dict mapping Y+ vertex indices to Y- vertex indices
        
    Returns:
        Symmetrized vertices array
    """
    result = vertices.copy()
    
    for y_plus_idx, y_minus_idx in mirror_map.items():
        # Copy Y+ position to Y- with flipped Y
        result[y_minus_idx, 0] = result[y_plus_idx, 0]
        result[y_minus_idx, 1] = -result[y_plus_idx, 1]
        result[y_minus_idx, 2] = result[y_plus_idx, 2]
    
    return result
