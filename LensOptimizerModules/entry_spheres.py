"""
Entry sphere cutting functions for the Lens Optimizer.

This module provides functions to cut spherical entry surfaces at LED
positions for proper optical light entry into the lens.
"""

from typing import List, Optional
import FreeCAD
import Part

from .data_classes import ProfileParams, CenterlinePoint
from .lofting import create_swept_lens


def cut_lens_entry_spheres(lens_solid: Part.Shape,
                           led_positions: List[FreeCAD.Vector],
                           led_directions: List[FreeCAD.Vector],
                           first_profile_radius: float,
                           centerline_start: FreeCAD.Vector,
                           sphere_depth_factor: float = 0.3,
                           debug_save_spheres: bool = True,
                           result_group=None) -> Optional[Part.Shape]:
    """Cut spherical lens entry surfaces at LED positions.
    
    Creates concave spherical entry surfaces at each LED position by subtracting
    spheres from the lens solid.
    
    Args:
        lens_solid: The lofted lens solid to modify
        led_positions: List of LED position vectors
        led_directions: List of LED direction vectors (pointing into the lens)
        first_profile_radius: Radius of the first profile (maximum allowed sphere radius)
        centerline_start: Position of the first centerline point (lens entry)
        sphere_depth_factor: How deep the sphere cuts relative to radius (0.0-0.5)
        debug_save_spheres: If True, save spheres as visible geometry for debugging
        result_group: Optional parent group to contain debug geometry
    
    Returns:
        Modified lens solid with spherical entry surfaces, or original if cut fails
    """
    if lens_solid is None:
        return None
    
    if not led_positions or not led_directions:
        FreeCAD.Console.PrintWarning("No LED positions/directions provided for spherical cut\n")
        return lens_solid
    
    # Log input data for debugging
    FreeCAD.Console.PrintMessage(f"\n  Spherical Entry Cut Debug:\n")
    FreeCAD.Console.PrintMessage(f"    Number of LED positions: {len(led_positions)}\n")
    FreeCAD.Console.PrintMessage(f"    Number of LED directions: {len(led_directions)}\n")
    FreeCAD.Console.PrintMessage(f"    First profile radius: {first_profile_radius:.2f}mm\n")
    FreeCAD.Console.PrintMessage(f"    Centerline start: ({centerline_start.x:.2f}, "
                                 f"{centerline_start.y:.2f}, {centerline_start.z:.2f})\n")
    
    for i, (pos, dir) in enumerate(zip(led_positions, led_directions)):
        FreeCAD.Console.PrintMessage(f"    LED {i+1} pos: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}), "
                                     f"dir: ({dir.x:.2f}, {dir.y:.2f}, {dir.z:.2f})\n")
    
    # Constrain sphere radius to not exceed first profile radius
    max_sphere_radius = first_profile_radius * 0.95
    
    # Clamp depth factor to valid range
    sphere_depth_factor = max(0.05, min(0.45, sphere_depth_factor))
    
    result = lens_solid.copy()
    
    # Get document for debug geometry
    doc = FreeCAD.ActiveDocument
    debug_group = None
    if debug_save_spheres and doc:
        debug_group_name = "EntrySphere_Debug"
        debug_group = doc.getObject(debug_group_name)
        if debug_group is None:
            debug_group = doc.addObject("App::DocumentObjectGroup", debug_group_name)
            # Add to result group if provided
            if result_group is not None:
                result_group.addObject(debug_group)
        # Clear old debug objects
        for obj in debug_group.Group:
            doc.removeObject(obj.Name)
    
    for i, (led_pos, led_dir) in enumerate(zip(led_positions, led_directions)):
        try:
            # Normalize LED direction
            led_dir_norm = FreeCAD.Vector(led_dir)
            if led_dir_norm.Length > 0.001:
                led_dir_norm.normalize()
            else:
                FreeCAD.Console.PrintWarning(f"LED {i+1}: Invalid direction, skipping spherical cut\n")
                continue
            
            # Calculate sphere parameters
            sphere_center = FreeCAD.Vector(led_pos)
            
            # Sphere radius is constrained to not intersect with the first profile's side walls
            sphere_radius = max_sphere_radius * sphere_depth_factor * 2.0
            sphere_radius = min(sphere_radius, max_sphere_radius)
            
            FreeCAD.Console.PrintMessage(f"    LED {i+1} sphere center (=LED pos): "
                                         f"({sphere_center.x:.2f}, {sphere_center.y:.2f}, {sphere_center.z:.2f}), "
                                         f"r={sphere_radius:.2f}mm, max_r={max_sphere_radius:.2f}mm\n")
            
            # Create the sphere
            sphere = Part.makeSphere(sphere_radius, sphere_center)
            
            if not sphere.isValid():
                FreeCAD.Console.PrintWarning(f"LED {i+1}: Invalid sphere created, skipping\n")
                continue
            
            # Save sphere as debug geometry
            if debug_save_spheres and doc and debug_group:
                sphere_obj = doc.addObject("Part::Feature", f"EntrySphere_LED{i+1}")
                sphere_obj.Shape = sphere
                sphere_obj.ViewObject.ShapeColor = (1.0, 0.5, 0.0)  # Orange
                sphere_obj.ViewObject.Transparency = 70
                debug_group.addObject(sphere_obj)
                FreeCAD.Console.PrintMessage(f"    Saved debug sphere: {sphere_obj.Name}\n")
            
            # Cut the sphere from the lens
            try:
                cut_result = result.cut(sphere)
                
                if cut_result.isValid() and cut_result.Volume > 0.1:
                    result = cut_result
                    cut_depth = sphere_radius * sphere_depth_factor
                    FreeCAD.Console.PrintMessage(
                        f"  LED {i+1}: Spherical entry cut SUCCESS (r={sphere_radius:.2f}mm, "
                        f"depth={cut_depth:.2f}mm)\n"
                    )
                else:
                    FreeCAD.Console.PrintWarning(
                        f"LED {i+1}: Spherical cut resulted in invalid geometry, skipping\n"
                    )
            except Exception as cut_error:
                FreeCAD.Console.PrintWarning(f"LED {i+1}: Cut operation failed: {cut_error}\n")
                continue
                
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"LED {i+1}: Spherical entry creation failed: {e}\n")
            continue
    
    return result


def create_lens_with_entry_spheres(profiles: List[ProfileParams],
                                    centerline_points: List[CenterlinePoint],
                                    led_positions: List[FreeCAD.Vector],
                                    led_directions: List[FreeCAD.Vector],
                                    sphere_depth_factor: float = 0.3,
                                    use_circle: bool = False,
                                    result_group=None) -> Optional[Part.Shape]:
    """Create a lens with spherical entry surfaces at LED positions.
    
    This is a convenience function that combines create_swept_lens() with
    cut_lens_entry_spheres() to create a lens with proper optical entry surfaces.
    
    Args:
        profiles: List of ProfileParams defining cross-sections
        centerline_points: List of CenterlinePoint objects defining the curved centerline
        led_positions: List of LED position vectors
        led_directions: List of LED direction vectors (pointing into the lens)
        sphere_depth_factor: How deep the sphere cuts (0.0-0.5)
        use_circle: If True, use circular profiles; if False, use polygon profiles
        result_group: Optional parent group to contain debug geometry
    
    Returns:
        Lens solid with spherical entry surfaces, or None on failure
    """
    # Create the base lens
    lens_solid = create_swept_lens(profiles, centerline_points, use_circle)
    
    if lens_solid is None:
        return None
    
    # Get first profile radius for constraint
    sorted_profiles = sorted(profiles, key=lambda p: p.z_position)
    first_profile_radius = sorted_profiles[0].radius
    
    # Get centerline start position
    centerline_start = centerline_points[0].position
    
    # Cut spherical entry surfaces
    result = cut_lens_entry_spheres(
        lens_solid,
        led_positions,
        led_directions,
        first_profile_radius,
        centerline_start,
        sphere_depth_factor,
        result_group=result_group
    )
    
    return result


def transform_lens_to_envelope(lens_shape: Part.Shape, 
                                transform: Optional[FreeCAD.Placement]) -> Part.Shape:
    """Transform a lens to match envelope orientation, or return as-is if no transform needed.
    
    Args:
        lens_shape: The lens solid (may already be at world position)
        transform: FreeCAD.Placement with rotation and translation, or None
        
    Returns:
        Part.Shape - transformed if transform provided, or original if transform is None
    """
    if transform is None:
        return lens_shape
    
    transformed = lens_shape.copy()
    transformed.Placement = transform
    return transformed
