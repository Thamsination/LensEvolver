"""
Centerline extraction functions for the Lens Optimizer.

This module provides functions to extract centerlines from envelope shapes
and user-provided sketches for lens lofting operations.
"""

import math
from typing import List, Tuple, Optional
import FreeCAD
import Part

from .data_classes import CenterlinePoint


def compute_perpendicular_basis(axis: FreeCAD.Vector) -> Tuple[FreeCAD.Vector, FreeCAD.Vector]:
    """Compute two perpendicular vectors to form a local coordinate system.
    
    Given an axis direction, returns two unit vectors (u, v) that are:
    - Perpendicular to the axis
    - Perpendicular to each other
    - Together with axis form a right-handed coordinate system
    
    Args:
        axis: The principal axis direction
        
    Returns:
        Tuple of (u_axis, v_axis) - both are unit vectors perpendicular to axis
    """
    # Normalize the input axis
    axis_norm = FreeCAD.Vector(axis)
    axis_norm.normalize()
    
    # Pick a reference vector that's not parallel to the axis
    # Use whichever standard basis vector is least aligned with axis
    if abs(axis_norm.z) < 0.9:
        ref = FreeCAD.Vector(0, 0, 1)
    else:
        ref = FreeCAD.Vector(1, 0, 0)
    
    # Compute perpendicular vectors using cross products
    u = axis_norm.cross(ref)
    u.normalize()
    v = axis_norm.cross(u)
    v.normalize()
    
    return u, v


def compute_slice_centroid(envelope_shape: Part.Shape, 
                           slice_position: float, 
                           axis_vector: FreeCAD.Vector) -> Optional[FreeCAD.Vector]:
    """Compute the centroid of an envelope cross-section at a given position.
    
    Args:
        envelope_shape: The FreeCAD Part.Shape of the envelope solid
        slice_position: Position along the axis to slice at (coordinate value)
        axis_vector: Unit vector of the slicing axis (e.g., (0,1,0) for Y-axis)
        
    Returns:
        FreeCAD.Vector centroid of the cross-section, or None if slice failed
    """
    try:
        # Slice the envelope perpendicular to the axis
        section = envelope_shape.slice(axis_vector, slice_position)
        
        if not section:
            return None
        
        # Collect all vertices from the slice
        vertices = []
        for edge in section:
            for vertex in edge.Vertexes:
                vertices.append(vertex.Point)
        
        if not vertices:
            return None
        
        # Compute centroid as average of all vertices
        centroid_x = sum(v.x for v in vertices) / len(vertices)
        centroid_y = sum(v.y for v in vertices) / len(vertices)
        centroid_z = sum(v.z for v in vertices) / len(vertices)
        
        return FreeCAD.Vector(centroid_x, centroid_y, centroid_z)
        
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Slice centroid failed at {slice_position}: {e}\n")
        return None


def compute_slice_max_radius(envelope_shape: Part.Shape,
                             slice_position: float,
                             axis_vector: FreeCAD.Vector,
                             centroid: FreeCAD.Vector) -> float:
    """Compute the maximum inscribed radius at a slice position.
    
    Args:
        envelope_shape: The FreeCAD Part.Shape of the envelope solid
        slice_position: Position along the axis to slice at
        axis_vector: Unit vector of the slicing axis
        centroid: Centroid point of the slice (where radius is measured from)
        
    Returns:
        Minimum distance from centroid to envelope edge (inscribed radius)
    """
    try:
        section = envelope_shape.slice(axis_vector, slice_position)
        
        if not section:
            return 1.0  # Fallback
        
        min_dist = float('inf')
        for edge in section:
            for vertex in edge.Vertexes:
                # Distance from centroid in the perpendicular plane
                vec_to_vertex = vertex.Point - centroid
                # Remove component along principal axis
                dist_along_axis = vec_to_vertex.dot(axis_vector)
                perp_vec = vec_to_vertex - axis_vector * dist_along_axis
                dist = perp_vec.Length
                
                if dist > 0.5:  # Ignore vertices very close to centroid
                    min_dist = min(min_dist, dist)
        
        if min_dist < float('inf') and min_dist > 1.0:
            return min_dist
        else:
            return 1.0  # Fallback
            
    except Exception:
        return 1.0  # Fallback


def find_envelope_end_center(envelope_shape: Part.Shape,
                              led_pos: FreeCAD.Vector,
                              led_direction: FreeCAD.Vector) -> FreeCAD.Vector:
    """Find the geometric center of the envelope at its far end.
    
    Uses slicing along the LED direction to find where the envelope ends,
    then returns the centroid of the last valid slice.
    
    Args:
        envelope_shape: The FreeCAD Part.Shape of the envelope solid
        led_pos: Position of the LED
        led_direction: Normalized direction the LED is pointing
        
    Returns:
        FreeCAD.Vector at the center of the envelope's far end
    """
    bbox = envelope_shape.BoundBox
    
    # Normalize LED direction
    led_dir_norm = FreeCAD.Vector(led_direction)
    if led_dir_norm.Length > 0.001:
        led_dir_norm.normalize()
    else:
        # Fallback: use direction from LED to bbox center
        led_dir_norm = FreeCAD.Vector(
            (bbox.XMin + bbox.XMax) / 2 - led_pos.x,
            (bbox.YMin + bbox.YMax) / 2 - led_pos.y,
            (bbox.ZMin + bbox.ZMax) / 2 - led_pos.z
        )
        if led_dir_norm.Length > 0.001:
            led_dir_norm.normalize()
        else:
            return bbox.Center  # Complete fallback
    
    # Find the search range by projecting bbox corners onto LED direction
    corners = [
        FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMin),
        FreeCAD.Vector(bbox.XMax, bbox.YMin, bbox.ZMin),
        FreeCAD.Vector(bbox.XMin, bbox.YMax, bbox.ZMin),
        FreeCAD.Vector(bbox.XMax, bbox.YMax, bbox.ZMin),
        FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMax),
        FreeCAD.Vector(bbox.XMax, bbox.YMin, bbox.ZMax),
        FreeCAD.Vector(bbox.XMin, bbox.YMax, bbox.ZMax),
        FreeCAD.Vector(bbox.XMax, bbox.YMax, bbox.ZMax),
    ]
    
    led_proj = led_pos.dot(led_dir_norm)
    projections = [(corner.dot(led_dir_norm) - led_proj) for corner in corners]
    min_proj = min(projections)
    max_proj = max(projections)
    
    # Search from LED toward far end, finding last valid slice centroid
    num_search_steps = 50
    search_range = max_proj - min_proj
    step_size = search_range / num_search_steps
    
    last_valid_centroid = None
    
    for i in range(num_search_steps + 1):
        # Start from just past LED and go toward far end
        offset = min_proj + i * step_size
        slice_pos = led_proj + offset
        
        centroid = compute_slice_centroid(envelope_shape, slice_pos, led_dir_norm)
        if centroid is not None:
            last_valid_centroid = centroid
    
    if last_valid_centroid is not None:
        return last_valid_centroid
    
    # Fallback: return center of far bbox face
    bbox_center = FreeCAD.Vector(
        (bbox.XMin + bbox.XMax) / 2,
        (bbox.YMin + bbox.YMax) / 2,
        (bbox.ZMin + bbox.ZMax) / 2
    )
    far_offset = max_proj * 0.95  # 95% toward far end
    far_center = led_pos + led_dir_norm * far_offset
    return far_center


def compute_smoothed_tangent(points: List[CenterlinePoint], 
                              index: int, 
                              axis_vector: FreeCAD.Vector, 
                              blend_strength: float = 0.3) -> FreeCAD.Vector:
    """Compute a smoothed tangent using multiple neighbors.
    
    Uses a weighted average of differences to neighboring points,
    with closer neighbors having more influence. At the endpoints,
    blends with the principal axis direction for stability.
    
    Args:
        points: List of CenterlinePoint objects
        index: Index of the point to compute tangent for
        axis_vector: Principal axis direction vector
        blend_strength: How much to blend with axis at endpoints (0-1)
        
    Returns:
        Normalized tangent vector
    """
    n = len(points)
    WINDOW = 3  # Use up to 3 points on each side
    
    # Collect weighted differences
    tangent = FreeCAD.Vector(0, 0, 0)
    total_weight = 0.0
    
    for offset in range(1, WINDOW + 1):
        weight = 1.0 / offset  # Closer = higher weight
        
        # Forward neighbor
        if index + offset < n:
            diff = points[index + offset].position - points[index].position
            tangent = tangent + diff * weight
            total_weight += weight
        
        # Backward neighbor
        if index - offset >= 0:
            diff = points[index].position - points[index - offset].position
            tangent = tangent + diff * weight
            total_weight += weight
    
    if total_weight > 0:
        tangent = tangent * (1.0 / total_weight)
    
    # Normalize
    if tangent.Length > 0.001:
        tangent.normalize()
    else:
        tangent = FreeCAD.Vector(axis_vector)  # Copy axis vector
    
    # Blend with principal axis at endpoints for stability
    t = points[index].t  # 0.0 to 1.0
    if t < 0.2:
        # Near start: blend more with axis
        blend = blend_strength * (1.0 - t / 0.2)
        tangent = tangent * (1.0 - blend) + axis_vector * blend
        if tangent.Length > 0.001:
            tangent.normalize()
    elif t > 0.8:
        # Near end: blend more with axis
        blend = blend_strength * (t - 0.8) / 0.2
        tangent = tangent * (1.0 - blend) + axis_vector * blend
        if tangent.Length > 0.001:
            tangent.normalize()
    
    return tangent


def extract_curved_centerline(envelope_shape: Part.Shape, 
                               num_samples: int = 25) -> Tuple[List[CenterlinePoint], str]:
    """Extract a curved centerline by connecting slice centroids.
    
    This follows the actual shape of the envelope rather than using a straight
    line through the bounding box center.
    
    Args:
        envelope_shape: The FreeCAD Part.Shape of the envelope solid
        num_samples: Number of slices to take along the principal axis
        
    Returns:
        Tuple of:
        - List of CenterlinePoint objects defining the curved centerline
        - principal_axis: String 'X', 'Y', or 'Z' indicating which axis is principal
    """
    # Use world-space bounding box to determine principal axis
    bbox = envelope_shape.BoundBox
    
    FreeCAD.Console.PrintMessage(f"  BBox: X=[{bbox.XMin:.2f}, {bbox.XMax:.2f}], "
                                 f"Y=[{bbox.YMin:.2f}, {bbox.YMax:.2f}], "
                                 f"Z=[{bbox.ZMin:.2f}, {bbox.ZMax:.2f}]\n")
    FreeCAD.Console.PrintMessage(f"  Dimensions: X={bbox.XLength:.2f}, "
                                 f"Y={bbox.YLength:.2f}, Z={bbox.ZLength:.2f}\n")
    
    # Find the longest axis - this is the principal axis for slicing
    dims = [
        ('X', bbox.XLength, bbox.XMin, bbox.XMax),
        ('Y', bbox.YLength, bbox.YMin, bbox.YMax),
        ('Z', bbox.ZLength, bbox.ZMin, bbox.ZMax)
    ]
    dims.sort(key=lambda x: x[1], reverse=True)
    
    principal_axis = dims[0][0]
    axis_length = dims[0][1]
    axis_min = dims[0][2]
    axis_max = dims[0][3]
    
    # Get axis vector for slicing
    if principal_axis == 'X':
        axis_vector = FreeCAD.Vector(1, 0, 0)
    elif principal_axis == 'Y':
        axis_vector = FreeCAD.Vector(0, 1, 0)
    else:
        axis_vector = FreeCAD.Vector(0, 0, 1)
    
    FreeCAD.Console.PrintMessage(f"  Principal axis: {principal_axis} ({axis_length:.2f}mm)\n")
    
    # Slice at multiple positions and compute centroids
    START_MARGIN = 0.05  # 5% inset from start
    END_MARGIN = 0.05    # 5% inset from end
    
    centerline_points = []
    
    for i in range(num_samples):
        # Map i from [0, num_samples-1] to t from [START_MARGIN, 1-END_MARGIN]
        t = START_MARGIN + (i / (num_samples - 1)) * (1.0 - START_MARGIN - END_MARGIN)
        slice_pos = axis_min + axis_length * t
        
        # Compute centroid of this slice
        centroid = compute_slice_centroid(envelope_shape, slice_pos, axis_vector)
        
        if centroid is None:
            # Fallback: use bounding box center for this slice
            if principal_axis == 'X':
                centroid = FreeCAD.Vector(slice_pos, 
                                          (bbox.YMin + bbox.YMax) / 2,
                                          (bbox.ZMin + bbox.ZMax) / 2)
            elif principal_axis == 'Y':
                centroid = FreeCAD.Vector((bbox.XMin + bbox.XMax) / 2,
                                          slice_pos,
                                          (bbox.ZMin + bbox.ZMax) / 2)
            else:
                centroid = FreeCAD.Vector((bbox.XMin + bbox.XMax) / 2,
                                          (bbox.YMin + bbox.YMax) / 2,
                                          slice_pos)
        
        # Compute max radius at this centroid
        max_radius = compute_slice_max_radius(envelope_shape, slice_pos, axis_vector, centroid)
        
        centerline_points.append(CenterlinePoint(
            position=centroid,
            t=t,
            max_radius=max_radius,
            tangent=None
        ))
    
    # Compute smoothed tangent directions for each point
    for i, point in enumerate(centerline_points):
        point.tangent = compute_smoothed_tangent(centerline_points, i, axis_vector)
    
    # Log centerline info
    start = centerline_points[0].position
    end = centerline_points[-1].position
    centerline_length = sum(
        (centerline_points[i+1].position - centerline_points[i].position).Length 
        for i in range(len(centerline_points) - 1)
    )
    
    FreeCAD.Console.PrintMessage(f"  Curved centerline: {len(centerline_points)} points, "
                                 f"{centerline_length:.2f}mm length\n")
    FreeCAD.Console.PrintMessage(f"  Start: ({start.x:.2f}, {start.y:.2f}, {start.z:.2f})\n")
    FreeCAD.Console.PrintMessage(f"  End: ({end.x:.2f}, {end.y:.2f}, {end.z:.2f})\n")
    
    radii = [p.max_radius for p in centerline_points]
    FreeCAD.Console.PrintMessage(f"  Radius range: {min(radii):.2f} - {max(radii):.2f} mm\n")
    
    return centerline_points, principal_axis


def extract_centerline_from_envelope(envelope_shape: Part.Shape) -> Tuple[List[CenterlinePoint], str]:
    """Extract curved centerline from envelope using slice centroids.
    
    DEPRECATED: Use extract_curved_centerline() or extract_curved_centerline_auto() instead.
    """
    return extract_curved_centerline(envelope_shape)


def extract_centerline_from_sketch(sketch_obj, envelope_shape: Part.Shape, 
                                    led_pos: FreeCAD.Vector,
                                    num_samples: int = 25) -> List[CenterlinePoint]:
    """Extract centerline points from a user-provided Sketch with BSpline.
    
    Samples points along the BSpline curve and computes the tangent and max radius
    at each point. The direction is auto-detected so the centerline starts near
    the LED position.
    
    Args:
        sketch_obj: FreeCAD Sketch object containing a BSpline curve
        envelope_shape: Envelope solid for computing max radii
        led_pos: LED position vector (used to determine curve direction)
        num_samples: Number of points to sample along the curve
        
    Returns:
        List of CenterlinePoint objects defining the centerline
    """
    FreeCAD.Console.PrintMessage("Extracting centerline from user-provided sketch...\n")
    
    # Get the shape from the sketch
    sketch_shape = sketch_obj.Shape
    
    if not sketch_shape.Edges:
        FreeCAD.Console.PrintError(f"Sketch '{sketch_obj.Label}' has no edges!\n")
        return []
    
    # Find the longest edge (assuming it's the main BSpline)
    main_edge = None
    max_length = 0
    
    for edge in sketch_shape.Edges:
        if edge.Length > max_length:
            max_length = edge.Length
            main_edge = edge
    
    if main_edge is None:
        FreeCAD.Console.PrintError("Could not find a valid edge in the sketch!\n")
        return []
    
    # Get curve type info for logging
    curve_type = "Unknown"
    if hasattr(main_edge, 'Curve'):
        curve_type = main_edge.Curve.TypeId.split('::')[-1]
    
    FreeCAD.Console.PrintMessage(f"  Using edge: {curve_type}, length={max_length:.2f}mm\n")
    
    # Get parameter range for sampling
    first_param = main_edge.FirstParameter
    last_param = main_edge.LastParameter
    
    # Check if curve needs reversing (start should be near LED)
    start_point = main_edge.valueAt(first_param)
    end_point = main_edge.valueAt(last_param)
    
    dist_start_to_led = (start_point - led_pos).Length
    dist_end_to_led = (end_point - led_pos).Length
    
    # If end is closer to LED, reverse the sampling direction
    reverse_direction = dist_end_to_led < dist_start_to_led
    if reverse_direction:
        first_param, last_param = last_param, first_param
        FreeCAD.Console.PrintMessage("  Reversed centerline direction (start near LED)\n")
    
    # Sample points along the curve
    centerline_points = []
    
    for i in range(num_samples):
        # Normalized position (0 to 1)
        t_norm = i / (num_samples - 1)
        
        # Parameter along the curve
        param = first_param + t_norm * (last_param - first_param)
        
        # Get point and tangent at this parameter
        point = main_edge.valueAt(param)
        tangent = main_edge.tangentAt(param)
        
        # If reversed, flip tangent direction
        if reverse_direction:
            tangent = tangent * -1
        
        if tangent.Length > 0.001:
            tangent.normalize()
        else:
            # Fallback: use direction to next/previous point
            if i < num_samples - 1:
                next_param = first_param + (t_norm + 1/(num_samples-1)) * (last_param - first_param)
                tangent = main_edge.valueAt(next_param) - point
            else:
                prev_param = first_param + (t_norm - 1/(num_samples-1)) * (last_param - first_param)
                tangent = point - main_edge.valueAt(prev_param)
            if tangent.Length > 0.001:
                tangent.normalize()
            else:
                tangent = FreeCAD.Vector(0, 1, 0)  # Default fallback
        
        # Compute max radius using envelope slicing
        slice_position = point.dot(tangent)
        slice_centroid = compute_slice_centroid(envelope_shape, slice_position, tangent)
        centroid_for_radius = slice_centroid if slice_centroid else point
        
        max_radius = compute_slice_max_radius(envelope_shape, 
                                               slice_position,
                                               tangent, 
                                               centroid_for_radius)
        
        centerline_points.append(CenterlinePoint(
            position=point,
            t=t_norm,
            max_radius=max_radius,
            tangent=tangent
        ))
    
    # Log centerline info
    if centerline_points:
        start = centerline_points[0].position
        end = centerline_points[-1].position
        radii = [p.max_radius for p in centerline_points]
        
        FreeCAD.Console.PrintMessage(f"  Sampled {len(centerline_points)} points along centerline\n")
        FreeCAD.Console.PrintMessage(f"  Start: ({start.x:.2f}, {start.y:.2f}, {start.z:.2f})\n")
        FreeCAD.Console.PrintMessage(f"  End: ({end.x:.2f}, {end.y:.2f}, {end.z:.2f})\n")
        FreeCAD.Console.PrintMessage(f"  Radius range: {min(radii):.2f} - {max(radii):.2f} mm\n")
    
    return centerline_points


def compute_radius_at_point_perpendicular(envelope_shape: Part.Shape, 
                                           point: FreeCAD.Vector, 
                                           tangent: FreeCAD.Vector) -> float:
    """Compute the inscribed radius at a point by measuring perpendicular to tangent.
    
    Creates a plane perpendicular to the tangent at the given point and finds
    the minimum distance from the point to the envelope boundary in that plane.
    
    Args:
        envelope_shape: The envelope solid
        point: Center point to measure from
        tangent: Direction vector (radius is measured perpendicular to this)
        
    Returns:
        Maximum inscribed radius at this point
    """
    try:
        # Get two perpendicular basis vectors
        if abs(tangent.z) < 0.9:
            ref = FreeCAD.Vector(0, 0, 1)
        else:
            ref = FreeCAD.Vector(1, 0, 0)
        
        u = tangent.cross(ref)
        if u.Length > 0.001:
            u.normalize()
        else:
            u = FreeCAD.Vector(1, 0, 0)
        
        v = tangent.cross(u)
        if v.Length > 0.001:
            v.normalize()
        else:
            v = FreeCAD.Vector(0, 1, 0)
        
        # Sample in 8 directions perpendicular to tangent
        min_dist = float('inf')
        num_dirs = 8
        
        for i in range(num_dirs):
            angle = 2 * math.pi * i / num_dirs
            direction = u * math.cos(angle) + v * math.sin(angle)
            
            # Cast a ray from point in this direction
            ray_length = 50.0  # Max ray length
            ray_end = point + direction * ray_length
            
            # Create a line and find intersection with envelope
            line = Part.makeLine(point, ray_end)
            
            try:
                # Find intersection points with envelope
                intersections = envelope_shape.section(line)
                
                if intersections.Vertexes:
                    for vertex in intersections.Vertexes:
                        dist = (vertex.Point - point).Length
                        if dist > 0.5:  # Ignore very close points
                            min_dist = min(min_dist, dist)
            except Exception:
                pass
        
        if min_dist < float('inf') and min_dist > 1.0:
            return min_dist
        else:
            return 5.0  # Fallback
            
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Radius computation failed: {e}\n")
        return 5.0


def extract_curved_centerline_auto(envelope_shape: Part.Shape, 
                                    led_pos: FreeCAD.Vector,
                                    led_direction: FreeCAD.Vector = None,
                                    num_points: int = 25) -> List[CenterlinePoint]:
    """Extract a curved centerline automatically by marching through the envelope.
    
    This function follows the envelope's natural curve by iteratively slicing,
    computing centroids, and marching from the LED end to the far end. This
    works for organic/curved envelope shapes without requiring a user sketch.
    
    Algorithm:
    1. Find entry point (closest envelope surface point to LED)
    2. Find exit point (farthest point from LED along envelope)
    3. Initialize direction using LED direction (if provided) for smooth start
    4. March through envelope:
       - Slice perpendicular to current direction
       - Compute centroid of slice (this is the centerline point)
       - Compute max_radius as min distance from centroid to slice edges
       - Update direction toward next expected position (blending with LED direction)
       - Step forward
    
    Args:
        envelope_shape: The FreeCAD Part.Shape of the envelope solid
        led_pos: Position of the LED (determines which end is "start")
        led_direction: Direction the LED is pointing (used for initial tangent)
        num_points: Number of centerline points to generate
        
    Returns:
        List of CenterlinePoint objects defining the curved centerline
    """
    FreeCAD.Console.PrintMessage("Extracting centerline automatically from envelope geometry...\n")
    
    # Step 1: Find entry and exit points
    bbox = envelope_shape.BoundBox
    
    # Get normalized LED direction for entry/exit point detection
    led_dir_norm = None
    if led_direction is not None:
        led_dir_norm = FreeCAD.Vector(led_direction)
        if led_dir_norm.Length > 0.001:
            led_dir_norm.normalize()
    
    # Find entry point: closest envelope surface point to LED
    # Sample vertices to find closest
    entry_point = None
    min_led_dist = float('inf')
    
    for vertex in envelope_shape.Vertexes:
        dist = (vertex.Point - led_pos).Length
        if dist < min_led_dist:
            min_led_dist = dist
            entry_point = vertex.Point
    
    if entry_point is None:
        FreeCAD.Console.PrintWarning("  Could not find entry point, using bounding box\n")
        entry_point = FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMin)
    
    # Exit point will be determined naturally by marching - set to None initially
    # This allows the centerline to follow the envelope's natural curve
    exit_point = None
    
    # Estimate total length for step sizing using bounding box diagonal
    # This is a rough estimate; the actual path may be longer if curved
    bbox_diagonal = FreeCAD.Vector(
        bbox.XMax - bbox.XMin,
        bbox.YMax - bbox.YMin, 
        bbox.ZMax - bbox.ZMin
    ).Length
    
    # Use distance from entry point to far bbox corner as length estimate
    far_corner = FreeCAD.Vector(bbox.XMax, bbox.YMax, bbox.ZMax)
    near_corner = FreeCAD.Vector(bbox.XMin, bbox.YMin, bbox.ZMin)
    # Choose the corner farthest from entry point
    if (far_corner - entry_point).Length > (near_corner - entry_point).Length:
        estimated_far = far_corner
    else:
        estimated_far = near_corner
    
    full_length = (estimated_far - entry_point).Length
    CENTERLINE_END_MARGIN = 0.02  # 2% margin from envelope end (closer to edge)
    total_length = full_length * (1.0 - CENTERLINE_END_MARGIN)
    step_size = total_length / (num_points - 1)
    
    FreeCAD.Console.PrintMessage(f"  Entry point: ({entry_point.x:.2f}, {entry_point.y:.2f}, {entry_point.z:.2f})\n")
    FreeCAD.Console.PrintMessage(f"  Estimated length: {full_length:.2f}mm, using {total_length:.2f}mm (2% margin)\n")
    
    # Step 2: Initialize direction
    # Start with LED direction (if provided) for correct initial tangent,
    # then let marching naturally follow the envelope's curve
    if led_direction is not None:
        direction = FreeCAD.Vector(led_direction)
        direction.normalize()
        overall_direction = FreeCAD.Vector(direction)  # Initial estimate
        FreeCAD.Console.PrintMessage(f"  Using LED direction for initial tangent: ({direction.x:.2f}, {direction.y:.2f}, {direction.z:.2f})\n")
    else:
        # Fallback: use direction from entry point toward bbox center
        bbox_center = FreeCAD.Vector(
            (bbox.XMin + bbox.XMax) / 2,
            (bbox.YMin + bbox.YMax) / 2,
            (bbox.ZMin + bbox.ZMax) / 2
        )
        overall_direction = bbox_center - entry_point
        if overall_direction.Length > 0.001:
            overall_direction.normalize()
        else:
            overall_direction = FreeCAD.Vector(0, 1, 0)  # Default to Y-up
        direction = FreeCAD.Vector(overall_direction)
    
    # Step 3: March through envelope collecting centerline points
    centerline_points = []
    current_pos = FreeCAD.Vector(entry_point)  # Copy entry point
    
    # Direction smoothing parameters
    DIRECTION_SMOOTHING = 0.3  # Blend factor for direction updates
    
    for i in range(num_points):
        t = i / (num_points - 1)
        
        # Slice perpendicular to current direction at current position
        slice_pos = current_pos.dot(direction)
        
        # Add offset for first/last slices to avoid edge failures
        if i == 0:
            slice_pos += 0.5  # Offset first slice slightly into envelope
        elif i == num_points - 1:
            slice_pos -= 0.5  # Offset last slice slightly before envelope end
        
        # Get slice and compute centroid
        centroid = compute_slice_centroid(envelope_shape, slice_pos, direction)
        
        if centroid is None:
            # Fallback: use current position
            centroid = FreeCAD.Vector(current_pos)  # Copy current position
            FreeCAD.Console.PrintWarning(f"  Slice {i+1}/{num_points} failed, using fallback position\n")
        
        # Compute max_radius as minimum distance from centroid to slice edges
        max_radius = compute_slice_max_radius(envelope_shape, slice_pos, direction, centroid)
        
        # Ensure minimum radius
        if max_radius < 1.0:
            max_radius = 1.0
        
        # Debug logging for first/last few points
        if i < 3 or i >= num_points - 2:
            FreeCAD.Console.PrintMessage(f"    Point {i+1}: slice_pos={slice_pos:.2f}, "
                                         f"centroid=({centroid.x:.2f}, {centroid.y:.2f}, {centroid.z:.2f}), "
                                         f"max_r={max_radius:.2f}mm\n")
        
        # Store the centerline point (tangent will be computed later)
        centerline_points.append(CenterlinePoint(
            position=centroid,
            t=t,
            max_radius=max_radius,
            tangent=FreeCAD.Vector(direction)  # Copy direction
        ))
        
        # Step 4: Update direction for next iteration (smooth following)
        if i < num_points - 1:
            # Look ahead: estimate where the next centroid should be
            next_pos = current_pos + direction * step_size
            next_slice_pos = next_pos.dot(direction)
            next_centroid = compute_slice_centroid(envelope_shape, next_slice_pos, direction)
            
            if next_centroid is not None:
                # Update direction toward next centroid (smoothed)
                new_direction = (next_centroid - centroid)
                if new_direction.Length > 0.1:
                    new_direction.normalize()
                    
                    # Blend with LED direction based on position along centerline
                    # At start (t=0): heavily favor LED direction
                    # At end: let marching naturally follow the envelope's curve
                    if led_direction is not None:
                        led_blend = max(0, 1.0 - t * 2)  # 1.0 at t=0, 0 at t>=0.5
                        if led_blend > 0:
                            # Blend LED direction with computed direction
                            led_dir_norm = FreeCAD.Vector(led_direction)
                            led_dir_norm.normalize()
                            new_direction = new_direction * (1 - led_blend) + led_dir_norm * led_blend
                            new_direction.normalize()
                    
                    # No end convergence - let marching naturally follow the envelope's curve
                    # The exit_point will be set after marching completes
                    
                    # Blend with current direction to prevent oscillation
                    direction = direction * (1 - DIRECTION_SMOOTHING) + new_direction * DIRECTION_SMOOTHING
                    direction.normalize()
        
        # Step forward
        current_pos = centroid + direction * step_size
    
    # Set exit_point to the naturally-found end of the centerline
    # This is where the marching algorithm ended up following the envelope's curve
    exit_point = centerline_points[-1].position
    FreeCAD.Console.PrintMessage(f"  Natural exit point: ({exit_point.x:.2f}, {exit_point.y:.2f}, {exit_point.z:.2f})\n")
    
    # Update overall_direction now that we know the actual path
    overall_direction = exit_point - entry_point
    if overall_direction.Length > 0.001:
        overall_direction.normalize()
    
    # Step 5: Compute smoothed tangents based on neighbor points
    # This ensures smooth profile orientation even with noisy centroids
    for i in range(len(centerline_points)):
        if i == 0:
            # First point: use LED direction if available for correct initial tangent
            if led_direction is not None:
                tangent = FreeCAD.Vector(led_direction)
            elif len(centerline_points) > 1:
                tangent = centerline_points[1].position - centerline_points[0].position
            else:
                tangent = FreeCAD.Vector(overall_direction)
        elif i == len(centerline_points) - 1:
            # Last point: use direction from previous points (following the natural curve)
            # Since exit_point IS this point, we use the incoming direction
            tangent = centerline_points[i].position - centerline_points[i-1].position
            if tangent.Length < 0.001:
                # Final fallback: use overall direction
                tangent = FreeCAD.Vector(overall_direction)
        else:
            # Middle points: average of forward and backward directions
            fwd = centerline_points[i+1].position - centerline_points[i].position
            bwd = centerline_points[i].position - centerline_points[i-1].position
            tangent = fwd + bwd
            
            # For early points, blend with LED direction
            t = centerline_points[i].t
            if led_direction is not None and t < 0.3:
                led_blend = 1.0 - (t / 0.3)  # 1.0 at t=0, 0 at t=0.3
                led_dir_norm = FreeCAD.Vector(led_direction)
                led_dir_norm.normalize()
                if tangent.Length > 0.001:
                    tangent.normalize()
                    tangent = tangent * (1 - led_blend) + led_dir_norm * led_blend
        
        if tangent.Length > 0.001:
            tangent.normalize()
        else:
            tangent = FreeCAD.Vector(direction)  # Fallback to overall direction
        
        centerline_points[i].tangent = tangent
    
    # Force first profiles to use LED direction for horizontal orientation
    # This ensures the first profile is perpendicular to the LED direction
    if led_direction is not None:
        led_dir_norm = FreeCAD.Vector(led_direction)
        led_dir_norm.normalize()
        
        # First 2 points: use pure LED direction
        for i in range(min(2, len(centerline_points))):
            centerline_points[i].tangent = FreeCAD.Vector(led_dir_norm)
        
        # Points 2-4: blend LED direction with computed tangent
        for i in range(2, min(5, len(centerline_points))):
            blend = (i - 2) / 3.0  # 0.0 at i=2, 0.33 at i=3, 0.67 at i=4
            computed = centerline_points[i].tangent
            if computed.Length > 0.001:
                computed.normalize()
            blended = led_dir_norm * (1 - blend) + computed * blend
            if blended.Length > 0.001:
                blended.normalize()
            centerline_points[i].tangent = blended
    
    # End tangents follow the natural curve - no forced convergence needed
    # The marching algorithm already found the natural path through the envelope
    
    # Log results
    start = centerline_points[0].position
    end = centerline_points[-1].position
    centerline_length = sum(
        (centerline_points[i+1].position - centerline_points[i].position).Length 
        for i in range(len(centerline_points) - 1)
    )
    
    radii = [p.max_radius for p in centerline_points]
    FreeCAD.Console.PrintMessage(f"  Auto centerline: {len(centerline_points)} points, {centerline_length:.2f}mm\n")
    FreeCAD.Console.PrintMessage(f"  Start: ({start.x:.2f}, {start.y:.2f}, {start.z:.2f})\n")
    FreeCAD.Console.PrintMessage(f"  End: ({end.x:.2f}, {end.y:.2f}, {end.z:.2f})\n")
    FreeCAD.Console.PrintMessage(f"  Radius range: {min(radii):.2f} - {max(radii):.2f} mm\n")
    
    return centerline_points
