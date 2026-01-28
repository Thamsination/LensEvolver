"""
Lofting functions for the Lens Optimizer.

This module provides functions to create solid lens geometry by lofting
or sweeping profiles along centerlines.
"""

from typing import List, Optional
import FreeCAD
import Part

from .data_classes import ProfileParams, CenterlinePoint
from .profile_creation import (
    create_polygon_profile,
    create_profile,
    interpolate_centerline_at_t
)


def create_lofted_lens(profiles: List[ProfileParams], 
                       centerline_points: List[CenterlinePoint]) -> Optional[Part.Shape]:
    """Create a solid lens by lofting profile polygons along a curved centerline.
    
    Uses ruled lofting with tangent control profiles at start/end to create
    predictable surface behavior and reduce bulging.
    
    Args:
        profiles: List of ProfileParams defining cross-sections
        centerline_points: List of CenterlinePoint objects defining the curved centerline
        
    Returns:
        Part.Shape solid following the centerline, or None on failure
    """
    if len(profiles) < 2:
        FreeCAD.Console.PrintError("Need at least 2 profiles for loft\n")
        return None
    
    if len(centerline_points) < 2:
        FreeCAD.Console.PrintError("Need at least 2 centerline points\n")
        return None
    
    # Sort profiles by position along centerline
    sorted_profiles = sorted(profiles, key=lambda p: p.z_position)
    
    # --- Step 1: Create B-spline spine from centerline points ---
    spine_points = [cp.position for cp in centerline_points]
    try:
        spine_curve = Part.BSplineCurve()
        spine_curve.interpolate(spine_points)
        spine_edge = spine_curve.toShape()
        spine_wire = Part.Wire([spine_edge])
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Failed to create spine curve: {e}, falling back to loft\n")
        spine_wire = None
    
    # --- Step 2: Create wires for each profile ---
    wires = []
    profile_centers = []  # Store centers for potential debugging
    for profile in sorted_profiles:
        # Find the centerline point closest to this profile's z_position
        t = profile.z_position  # Normalized position (0 to 1)
        
        # Find the two centerline points to interpolate between
        idx = 0
        for i, cp in enumerate(centerline_points):
            if cp.t >= t:
                idx = i
                break
            idx = i
        
        if idx == 0:
            center = centerline_points[0].position
            tangent = centerline_points[0].tangent
        elif idx >= len(centerline_points) - 1:
            center = centerline_points[-1].position
            tangent = centerline_points[-1].tangent
        else:
            cp0 = centerline_points[idx - 1]
            cp1 = centerline_points[idx]
            
            if cp1.t - cp0.t > 0.001:
                frac = (t - cp0.t) / (cp1.t - cp0.t)
            else:
                frac = 0.5
            
            center = cp0.position + (cp1.position - cp0.position) * frac
            tangent = cp0.tangent + (cp1.tangent - cp0.tangent) * frac
            if tangent.Length > 0.001:
                tangent.normalize()
            else:
                tangent = cp0.tangent
        
        # Create polygon perpendicular to tangent
        wire = create_polygon_profile(profile, center, tangent)
        wires.append(wire)
        profile_centers.append(center)
    
    # --- Step 3: Create ruled loft for predictable surface behavior ---
    try:
        loft = Part.makeLoft(wires, solid=True, ruled=True)
        
        if loft.isValid():
            return loft
        else:
            FreeCAD.Console.PrintWarning("Ruled loft invalid, trying smooth loft...\n")
            loft = Part.makeLoft(wires, solid=True, ruled=False)
            return loft if loft.isValid() else None
            
    except Exception as e:
        FreeCAD.Console.PrintError(f"Loft creation failed: {e}\n")
        return None


def create_swept_lens(profiles: List[ProfileParams], 
                      centerline_points: List[CenterlinePoint],
                      use_circle: bool = False) -> Optional[Part.Shape]:
    """Create a solid lens by lofting profiles along the centerline.
    
    Uses Part.makeLoft with smooth interpolation (ruled=False) to create
    clean geometry matching the manual Part::Loft behavior in FreeCAD.
    
    Args:
        profiles: List of ProfileParams defining cross-sections
        centerline_points: List of CenterlinePoint objects defining the curved centerline
        use_circle: If True, use circular profiles; if False, use polygon profiles
        
    Returns:
        Part.Shape solid following the centerline, or None on failure
    """
    if len(profiles) < 2:
        FreeCAD.Console.PrintError("Need at least 2 profiles for loft\n")
        return None
    
    if len(centerline_points) < 2:
        FreeCAD.Console.PrintError("Need at least 2 centerline points\n")
        return None
    
    num_sides = profiles[0].sides
    profile_type = "circular" if use_circle else f"{num_sides}-sided"
    
    # Sort profiles by position along centerline
    sorted_profiles = sorted(profiles, key=lambda p: p.z_position)
    
    # The last profile always has 1mm radius (creates dome-like tip)
    sorted_profiles[-1].radius = 1.0
    
    # Create profile wires at their positions along centerline
    wires = []
    for profile in sorted_profiles:
        # Interpolate position and tangent at this profile's z_position
        t = profile.z_position
        center, tangent = interpolate_centerline_at_t(centerline_points, t)
        
        # Create profile perpendicular to tangent (circle or polygon)
        wire = create_profile(profile, center, tangent, use_circle)
        wires.append(wire)
    
    # Create smooth loft (matching manual Part::Loft behavior)
    try:
        loft = Part.makeLoft(wires, solid=True, ruled=False, closed=False)
        
        if loft is not None and loft.isValid():
            FreeCAD.Console.PrintMessage(
                f"Smooth loft created successfully ({profile_type} profiles, "
                f"{len(wires)} sections, last=1mm)\n")
            return loft
        else:
            FreeCAD.Console.PrintWarning("Smooth loft invalid, trying ruled loft...\n")
            loft = Part.makeLoft(wires, solid=True, ruled=True, closed=False)
            if loft is not None and loft.isValid():
                FreeCAD.Console.PrintMessage(f"Ruled loft created ({profile_type} profiles, last=1mm)\n")
                return loft
            
    except Exception as e:
        FreeCAD.Console.PrintError(f"Loft creation failed: {e}\n")
    
    return None
