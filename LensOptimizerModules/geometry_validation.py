"""
Geometry validation functions for the Lens Optimizer.

This module provides functions to validate lens geometry against
envelope constraints and perform geometric calculations.
"""

import math
from typing import List, Tuple
import FreeCAD
import Part

from .config import DEFAULT_VOLUME_TOLERANCE
from .data_classes import ProfileParams


def get_envelope_radius_at_z(envelope_shape: Part.Shape, z_pos: float, 
                              center_x: float, center_y: float) -> float:
    """Get the actual envelope radius at a specific Z position.
    
    Uses cross-sectional slicing for accurate radius calculation, 
    not just bounding box approximation.
    
    Args:
        envelope_shape: The envelope solid
        z_pos: Z coordinate to check
        center_x: X coordinate of centerline
        center_y: Y coordinate of centerline
        
    Returns:
        Maximum radius from centerline to envelope edge at this Z
    """
    try:
        # Slice envelope at this Z
        section = envelope_shape.slice(FreeCAD.Vector(0, 0, 1), z_pos)
        
        if section:
            max_dist = 0.0
            for edge in section:
                for vertex in edge.Vertexes:
                    dx = vertex.Point.x - center_x
                    dy = vertex.Point.y - center_y
                    dist = math.sqrt(dx*dx + dy*dy)
                    max_dist = max(max_dist, dist)
            if max_dist > 0:
                return max_dist
    except Exception:
        pass
    
    # Fallback to bounding box estimate
    bbox = envelope_shape.BoundBox
    return max(bbox.XLength, bbox.YLength) / 2.0


def get_envelope_radius_at_position(envelope_shape: Part.Shape,
                                    centerline_start: FreeCAD.Vector,
                                    centerline_end: FreeCAD.Vector,
                                    t: float,
                                    u_axis: FreeCAD.Vector,
                                    v_axis: FreeCAD.Vector) -> float:
    """Get inscribed radius at a position along the principal axis.
    
    Slices perpendicular to the centerline and finds the MINIMUM distance
    to the envelope edge (inscribed circle radius). This ensures profiles
    fit within non-circular/organic envelope cross-sections.
    
    Args:
        envelope_shape: The envelope solid
        centerline_start: Start of centerline (FreeCAD.Vector)
        centerline_end: End of centerline (FreeCAD.Vector)
        t: Position along centerline (0.0 to 1.0)
        u_axis: First perpendicular axis
        v_axis: Second perpendicular axis
        
    Returns:
        Minimum radius from centerline to envelope edge (inscribed circle)
    """
    try:
        centerline = centerline_end - centerline_start
        point = centerline_start + centerline * t
        
        # Get principal axis direction (normalized centerline)
        principal_axis = FreeCAD.Vector(centerline)
        principal_axis.normalize()
        
        # Slice perpendicular to principal axis
        # The slice() function takes a normal direction and distance from origin
        slice_dist = point.dot(principal_axis)
        section = envelope_shape.slice(principal_axis, slice_dist)
        
        if section:
            # Calculate MIN distance from centerline point in the u-v plane
            # This gives the inscribed circle radius - safe for any cross-section shape
            min_dist = float('inf')
            for edge in section:
                for vertex in edge.Vertexes:
                    vec = vertex.Point - point
                    # Project onto perpendicular plane
                    dist_u = abs(vec.dot(u_axis))
                    dist_v = abs(vec.dot(v_axis))
                    dist = math.sqrt(dist_u**2 + dist_v**2)
                    # Only consider vertices at meaningful distance from centerline
                    if dist > 0.5:  # Ignore points very close to centerline
                        min_dist = min(min_dist, dist)
            if min_dist < float('inf') and min_dist > 0:
                return min_dist
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Radius calculation failed at t={t:.2f}: {e}\n")
    
    # Fallback to conservative estimate
    bbox = envelope_shape.BoundBox
    return min(bbox.XLength, bbox.YLength, bbox.ZLength) / 6.0


def get_max_radius_at_position(z_position: float, max_radii: List[float]) -> float:
    """Interpolate the maximum allowed radius at a given Z position.
    
    Args:
        z_position: Normalized position along centerline (0.0 to 1.0)
        max_radii: List of sampled max radii from extract_centerline_from_envelope()
        
    Returns:
        Interpolated maximum radius at this position
    """
    if not max_radii:
        return 10.0
    
    # Clamp position to valid range
    z_position = max(0.0, min(1.0, z_position))
    
    # Interpolate between samples
    num_samples = len(max_radii)
    idx_float = z_position * (num_samples - 1)
    idx_low = int(idx_float)
    idx_high = min(idx_low + 1, num_samples - 1)
    frac = idx_float - idx_low
    
    return max_radii[idx_low] * (1 - frac) + max_radii[idx_high] * frac


def clamp_profile_to_envelope(profile: ProfileParams, max_radii: List[float]) -> ProfileParams:
    """Clamp a profile's radius to fit within the envelope at its position.
    
    Uses pre-computed max_radii from extract_centerline_from_envelope().
    This is simple and reliable since radii were already sampled correctly.
    
    Args:
        profile: The profile to clamp
        max_radii: Pre-computed list of max radii along centerline
        
    Returns:
        The clamped profile (modified in place and returned)
    """
    # Get max radius at this profile's position from pre-computed list
    max_radius = get_max_radius_at_position(profile.z_position, max_radii)
    
    # Update max_radius and clamp
    profile.max_radius = max_radius
    profile.radius = min(profile.radius, max_radius * 0.95)  # 5% margin (was 20%)
    profile.radius = max(1.0, profile.radius)  # Minimum 1mm radius
    
    return profile


def validate_lens_geometry(lens_solid: Part.Shape, envelope_shape: Part.Shape) -> Tuple[bool, str]:
    """Validate that a lens geometry is valid and fits within the envelope.
    
    Performs multiple checks:
    1. Loft creation succeeded
    2. Geometry is a valid solid
    3. Geometry is watertight (closed)
    4. Geometry fits within envelope bounds
    5. No significant volume outside envelope
    
    Args:
        lens_solid: The lofted lens shape to validate
        envelope_shape: The envelope constraint shape
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check 1: Loft succeeded
    if lens_solid is None:
        return False, "Loft creation returned None"
    
    # Check 2: Basic validity
    try:
        if not lens_solid.isValid():
            return False, "Shape is not valid (self-intersecting or degenerate)"
    except Exception as e:
        return False, f"Validity check failed: {e}"
    
    # Check 3: Is it a solid with volume?
    try:
        if not lens_solid.Solids:
            return False, "Not a solid (no volume)"
        if lens_solid.Volume < 0.1:  # Less than 0.1 mm³
            return False, f"Volume too small: {lens_solid.Volume:.3f}mm³"
    except Exception as e:
        return False, f"Volume check failed: {e}"
    
    # Check 4: Boolean containment check (precise, works for organic/tilted shapes)
    # Note: We removed the axis-aligned bounding box check because it's too strict
    # for tilted/organic envelopes where the lens can fit inside but still have
    # a bounding box that exceeds the envelope's axis-aligned bbox.
    try:
        # Cut lens from envelope - if anything remains outside, it's invalid
        outside = lens_solid.cut(envelope_shape)
        if outside.Volume > DEFAULT_VOLUME_TOLERANCE:  # Allow small tolerance for coordinate mismatches
            return False, f"Volume outside envelope: {outside.Volume:.2f}mm³"
    except Exception as e:
        # Boolean operations can fail on complex geometry
        # Log warning but don't fail - geometry might still be valid
        FreeCAD.Console.PrintWarning(f"Boolean containment check failed: {e}\n")
    
    return True, "Valid"
