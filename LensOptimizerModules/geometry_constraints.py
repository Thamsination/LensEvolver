"""
Geometry constraint functions for the Lens Optimizer.

This module provides functions to constrain mesh vertices within envelope
bounds and maintain geometric validity.
"""

import numpy as np
import FreeCAD

from .mesh_operations import get_bounding_box


def identify_mirror_pairs(vertices, tolerance=0.01):
    """Find vertex pairs that should be mirrored across Y=0 (X-Z plane).
    
    Args:
        vertices: Nx3 array of vertex positions
        tolerance: Distance from Y=0 to consider as centerline
        
    Returns:
        dict with:
            'positive': indices of Y+ vertices
            'negative': indices of Y- vertices
            'centerline': indices of centerline vertices
            'pairs': list of (pos_idx, neg_idx) tuples for matching pairs
    """
    y_coords = vertices[:, 1]
    
    positive = np.where(y_coords > tolerance)[0]
    negative = np.where(y_coords < -tolerance)[0]
    centerline = np.where(np.abs(y_coords) <= tolerance)[0]
    
    # Find matching pairs using X-Z distance
    pairs = []
    used_neg = set()
    
    mesh_extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    match_tolerance = np.mean(mesh_extent) * 0.1
    
    for pos_idx in positive:
        pos_v = vertices[pos_idx]
        pos_xz = np.array([pos_v[0], pos_v[2]])
        
        best_neg_idx = None
        best_dist = float('inf')
        
        for neg_idx in negative:
            if neg_idx in used_neg:
                continue
            neg_v = vertices[neg_idx]
            neg_xz = np.array([neg_v[0], neg_v[2]])
            
            dist = np.linalg.norm(pos_xz - neg_xz)
            if dist < best_dist:
                best_dist = dist
                best_neg_idx = neg_idx
        
        if best_neg_idx is not None and best_dist < match_tolerance:
            pairs.append((pos_idx, best_neg_idx))
            used_neg.add(best_neg_idx)
    
    return {
        'positive': positive,
        'negative': negative,
        'centerline': centerline,
        'pairs': pairs
    }


def clamp_to_envelope(vertices, envelope_vertices, envelope_center=None):
    """Clamp vertices to lie within envelope bounds.
    
    Uses the envelope's bounding box as a simple constraint.
    
    Args:
        vertices: Nx3 array of vertex positions to clamp
        envelope_vertices: Nx3 array of envelope vertex positions
        envelope_center: Optional center point (computed if not provided)
        
    Returns:
        Clamped vertices array
    """
    env_bbox = get_bounding_box(envelope_vertices)
    
    if envelope_center is None:
        envelope_center = np.mean(envelope_vertices, axis=0)
    
    # Clamp to bounding box with small margin
    margin = 0.01
    clamped = np.clip(
        vertices,
        env_bbox['min'] + margin,
        env_bbox['max'] - margin
    )
    
    return clamped


def is_inside_envelope(vertices, envelope_vertices, tolerance=0.001):
    """Check if all vertices are inside the envelope bounds.
    
    Uses bounding box check for efficiency.
    
    Args:
        vertices: Nx3 array of vertex positions
        envelope_vertices: Nx3 array of envelope vertex positions
        tolerance: Small margin for numerical precision
        
    Returns:
        True if all vertices are inside envelope bounds
    """
    env_bbox = get_bounding_box(envelope_vertices)
    
    min_check = np.all(vertices >= env_bbox['min'] - tolerance)
    max_check = np.all(vertices <= env_bbox['max'] + tolerance)
    
    return min_check and max_check


def count_vertices_outside_envelope(vertices, envelope_vertices, tolerance=0.001):
    """Count how many vertices are outside envelope bounds.
    
    Args:
        vertices: Nx3 array of vertex positions
        envelope_vertices: Nx3 array of envelope vertex positions
        tolerance: Small margin for numerical precision
        
    Returns:
        Number of vertices outside bounds
    """
    env_bbox = get_bounding_box(envelope_vertices)
    count = 0
    
    for v in vertices:
        if (v[0] < env_bbox['min'][0] - tolerance or v[0] > env_bbox['max'][0] + tolerance or
            v[1] < env_bbox['min'][1] - tolerance or v[1] > env_bbox['max'][1] + tolerance or
            v[2] < env_bbox['min'][2] - tolerance or v[2] > env_bbox['max'][2] + tolerance):
            count += 1
    
    return count


def clamp_to_envelope_surface(vertices, envelope_vertices, envelope_faces, envelope_center):
    """Clamp vertices that are outside envelope to its surface.
    
    Projects outside vertices back to the envelope surface along
    the radial direction from envelope center.
    
    Args:
        vertices: Nx3 array of vertex positions
        envelope_vertices: Nx3 array of envelope vertex positions
        envelope_faces: Mx3 array of envelope face indices
        envelope_center: Center point of envelope
        
    Returns:
        Clamped vertices array
    """
    result = vertices.copy()
    env_bbox = get_bounding_box(envelope_vertices)
    
    for i, v in enumerate(vertices):
        # Check if outside bounding box
        if (v[0] < env_bbox['min'][0] or v[0] > env_bbox['max'][0] or
            v[1] < env_bbox['min'][1] or v[1] > env_bbox['max'][1] or
            v[2] < env_bbox['min'][2] or v[2] > env_bbox['max'][2]):
            
            # Project back toward center
            direction = v - envelope_center
            if np.linalg.norm(direction) > 0.001:
                direction = direction / np.linalg.norm(direction)
                
                # Find surface point by binary search
                inside_point = envelope_center
                outside_point = v
                
                for _ in range(10):  # 10 iterations for precision
                    mid = (inside_point + outside_point) / 2
                    
                    is_inside = (
                        mid[0] >= env_bbox['min'][0] and mid[0] <= env_bbox['max'][0] and
                        mid[1] >= env_bbox['min'][1] and mid[1] <= env_bbox['max'][1] and
                        mid[2] >= env_bbox['min'][2] and mid[2] <= env_bbox['max'][2]
                    )
                    
                    if is_inside:
                        inside_point = mid
                    else:
                        outside_point = mid
                
                result[i] = inside_point
    
    return result


def check_mesh_valid(vertices, faces):
    """Check if mesh geometry is valid.
    
    Performs basic validity checks:
    - No degenerate faces (zero area)
    - No inverted faces
    - Reasonable vertex distribution
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for degenerate faces
    degenerate_count = 0
    inverted_count = 0
    
    center = np.mean(vertices, axis=0)
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Check for degenerate (zero area)
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        area = np.linalg.norm(normal) / 2
        
        if area < 1e-10:
            degenerate_count += 1
            continue
        
        # Check for inverted normal (pointing inward)
        face_center = (v0 + v1 + v2) / 3
        outward = face_center - center
        normal = normal / np.linalg.norm(normal)
        
        if np.dot(normal, outward) < 0:
            inverted_count += 1
    
    if degenerate_count > 0:
        return False, f"{degenerate_count} degenerate faces"
    
    if inverted_count > len(faces) / 2:
        return False, f"{inverted_count} inverted faces (>50%)"
    
    return True, "Valid"


def check_min_face_angles(vertices, faces, min_angle_deg=10.0):
    """Check if all faces have minimum angle constraint.
    
    Very thin triangles (small angles) can cause numerical issues.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        min_angle_deg: Minimum allowed angle in degrees
        
    Returns:
        Tuple of (passes_check, num_failed, min_angle_found)
    """
    min_angle_rad = np.radians(min_angle_deg)
    num_failed = 0
    min_angle_found = 180.0
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Compute all three angles
        e01 = v1 - v0
        e02 = v2 - v0
        e10 = v0 - v1
        e12 = v2 - v1
        e20 = v0 - v2
        e21 = v1 - v2
        
        # Normalize for angle computation
        def safe_angle(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-10 or nb < 1e-10:
                return 0.0
            cos_angle = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
            return np.arccos(cos_angle)
        
        angles = [
            safe_angle(e01, e02),
            safe_angle(e10, e12),
            safe_angle(e20, e21)
        ]
        
        face_min = min(angles)
        min_angle_found = min(min_angle_found, np.degrees(face_min))
        
        if face_min < min_angle_rad:
            num_failed += 1
    
    return num_failed == 0, num_failed, min_angle_found
