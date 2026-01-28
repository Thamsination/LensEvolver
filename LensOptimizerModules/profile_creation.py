"""
Profile creation functions for the Lens Optimizer.

This module provides functions to create cross-section profiles
(polygons and circles) for lens lofting operations.
"""

import math
from typing import List, Tuple, Optional
import FreeCAD
import Part

from .data_classes import ProfileParams, CenterlinePoint


def create_polygon_profile(params: ProfileParams, center: FreeCAD.Vector, 
                           tangent: FreeCAD.Vector,
                           target_vertices: Optional[int] = None) -> Part.Wire:
    """Create a closed polygon wire at the specified center point, perpendicular to tangent.
    
    The profile plane is perpendicular to the tangent direction, allowing profiles
    to follow a curved centerline properly.
    
    WINDING DIRECTION: All profiles are created with consistent counter-clockwise
    winding when viewed from the positive tangent direction.
    
    Args:
        params: ProfileParams defining the polygon shape
        center: FreeCAD.Vector position for the profile center
        tangent: FreeCAD.Vector tangent direction (profile is perpendicular to this)
        target_vertices: Optional number of vertices to resample to
        
    Returns:
        Part.Wire representing the closed polygon perpendicular to the tangent
    """
    # Normalize tangent
    if tangent.Length < 0.001:
        tangent = FreeCAD.Vector(0, 1, 0)  # Default fallback
    tangent_norm = FreeCAD.Vector(tangent)
    tangent_norm.normalize()
    
    # Compute perpendicular basis vectors for the profile plane
    if abs(tangent_norm.z) < 0.9:
        up = FreeCAD.Vector(0, 0, 1)  # Prefer world Z as up
    elif abs(tangent_norm.x) < 0.9:
        up = FreeCAD.Vector(1, 0, 0)  # Fall back to world X
    else:
        up = FreeCAD.Vector(0, 1, 0)  # Fall back to world Y
    
    # u_axis and v_axis span the plane perpendicular to tangent
    u_axis = tangent_norm.cross(up)
    if u_axis.Length < 0.001:
        up = FreeCAD.Vector(1, 0, 0) if abs(tangent_norm.x) < 0.9 else FreeCAD.Vector(0, 1, 0)
        u_axis = tangent_norm.cross(up)
    u_axis.normalize()
    
    # v_axis completes the right-handed coordinate system
    v_axis = tangent_norm.cross(u_axis)
    v_axis.normalize()
    
    points = []
    angle_rad = math.radians(params.angle)
    
    # Create vertices in counter-clockwise order
    for i in range(params.sides):
        vertex_angle = angle_rad + (2 * math.pi * i / params.sides)
        
        # Create vertex in the perpendicular plane
        offset = u_axis * (params.radius * math.cos(vertex_angle)) + \
                 v_axis * (params.radius * math.sin(vertex_angle))
        
        points.append(center + offset)
    
    # Resample to target vertex count if specified
    if target_vertices is not None and target_vertices > params.sides:
        points = resample_polygon_to_vertex_count(points, target_vertices)
    
    # Close the polygon
    points.append(points[0])
    
    return Part.makePolygon(points)


def create_circle_profile(params: ProfileParams, center: FreeCAD.Vector, 
                          tangent: FreeCAD.Vector) -> Part.Wire:
    """Create a closed circle wire at the specified center point, perpendicular to tangent.
    
    Creates a true circle (not a polygon approximation) for smooth lens surfaces.
    
    Args:
        params: ProfileParams defining the circle (uses radius, ignores sides/angle)
        center: FreeCAD.Vector position for the circle center
        tangent: FreeCAD.Vector tangent direction (circle is perpendicular to this)
        
    Returns:
        Part.Wire representing the closed circle perpendicular to the tangent
    """
    # Normalize tangent
    if tangent.Length < 0.001:
        tangent = FreeCAD.Vector(0, 1, 0)  # Default fallback
    tangent_norm = FreeCAD.Vector(tangent)
    tangent_norm.normalize()
    
    # Create a circle in the plane perpendicular to the tangent
    circle = Part.Circle(center, tangent_norm, params.radius)
    
    # Convert to edge and then to wire
    edge = circle.toShape()
    wire = Part.Wire([edge])
    
    return wire


def create_profile(params: ProfileParams, center: FreeCAD.Vector,
                   tangent: FreeCAD.Vector, use_circle: bool = False,
                   target_vertices: Optional[int] = None) -> Part.Wire:
    """Create a profile wire (polygon or circle) at the specified center point.
    
    This is a wrapper function that selects between polygon and circle profiles.
    
    Args:
        params: ProfileParams defining the profile shape
        center: FreeCAD.Vector position for the profile center
        tangent: FreeCAD.Vector tangent direction (profile is perpendicular to this)
        use_circle: If True, create circular profile; if False, create polygon
        target_vertices: For polygons, optional number of vertices to resample to
        
    Returns:
        Part.Wire representing the closed profile perpendicular to the tangent
    """
    if use_circle:
        return create_circle_profile(params, center, tangent)
    else:
        return create_polygon_profile(params, center, tangent, target_vertices)


def resample_polygon_to_vertex_count(points: List[FreeCAD.Vector], 
                                      target_count: int) -> List[FreeCAD.Vector]:
    """Resample a polygon to have exactly target_count vertices.
    
    Distributes vertices evenly along the polygon perimeter while
    preserving the original shape.
    
    Args:
        points: List of polygon vertices (not closed - no duplicate end point)
        target_count: Desired number of vertices in output
        
    Returns:
        List of resampled vertices (not closed - no duplicate end point)
    """
    if len(points) == 0:
        return points
    
    if len(points) >= target_count:
        return points[:target_count]
    
    n = len(points)
    
    # Calculate edge lengths and total perimeter
    edge_lengths = []
    for i in range(n):
        next_i = (i + 1) % n
        edge_len = (points[next_i] - points[i]).Length
        edge_lengths.append(edge_len)
    
    total_perimeter = sum(edge_lengths)
    
    if total_perimeter < 0.0001:
        # Degenerate polygon, just duplicate first point
        return [FreeCAD.Vector(points[0]) for _ in range(target_count)]
    
    # Calculate cumulative distances at each original vertex
    cumulative_distances = [0.0]
    for i in range(n):
        cumulative_distances.append(cumulative_distances[-1] + edge_lengths[i])
    
    # Target spacing between new vertices
    target_spacing = total_perimeter / target_count
    
    # Generate new vertices at regular arc-length intervals
    new_points = []
    for i in range(target_count):
        target_dist = i * target_spacing
        
        # Find which edge this distance falls on
        edge_idx = 0
        for j in range(n):
            if cumulative_distances[j + 1] >= target_dist - 0.0001:
                edge_idx = j
                break
            edge_idx = j
        
        # Calculate position along this edge
        edge_start_dist = cumulative_distances[edge_idx]
        edge_len = edge_lengths[edge_idx]
        
        if edge_len > 0.0001:
            t = (target_dist - edge_start_dist) / edge_len
            t = max(0.0, min(1.0, t))
        else:
            t = 0.0
        
        # Interpolate between edge endpoints
        p0 = points[edge_idx]
        p1 = points[(edge_idx + 1) % n]
        new_point = p0 + (p1 - p0) * t
        new_points.append(new_point)
    
    return new_points


def interpolate_centerline_at_t(centerline_points: List[CenterlinePoint], 
                                 t: float) -> Tuple[FreeCAD.Vector, FreeCAD.Vector]:
    """Interpolate position and tangent at a given t value along the centerline.
    
    Args:
        centerline_points: List of CenterlinePoint objects defining the curved centerline
        t: Normalized position along centerline (0 to 1)
        
    Returns:
        Tuple of (position, tangent) vectors at the interpolated point
    """
    # Clamp t to valid range
    t = max(0.0, min(1.0, t))
    
    # Find the two centerline points to interpolate between
    idx = 0
    for i, cp in enumerate(centerline_points):
        if cp.t >= t:
            idx = i
            break
        idx = i
    
    if idx == 0:
        # At or before start
        return centerline_points[0].position, centerline_points[0].tangent
    elif idx >= len(centerline_points) - 1:
        # At or after end
        return centerline_points[-1].position, centerline_points[-1].tangent
    else:
        # Interpolate between centerline_points[idx-1] and centerline_points[idx]
        cp0 = centerline_points[idx - 1]
        cp1 = centerline_points[idx]
        
        # Linear interpolation factor
        if cp1.t - cp0.t > 0.001:
            frac = (t - cp0.t) / (cp1.t - cp0.t)
        else:
            frac = 0.5
        
        # Interpolate position
        center = cp0.position + (cp1.position - cp0.position) * frac
        
        # Interpolate tangent (and renormalize)
        tangent = cp0.tangent + (cp1.tangent - cp0.tangent) * frac
        if tangent.Length > 0.001:
            tangent.normalize()
        else:
            tangent = FreeCAD.Vector(cp0.tangent)
        
        return center, tangent
