"""
Mesh operations for the Lens Optimizer.

This module provides functions for converting between FreeCAD solids and meshes,
mesh subdivision, smoothing, and export.
"""

import numpy as np
import FreeCAD
import Part
import Mesh
import MeshPart


def solid_to_mesh(solid, resolution=0.5):
    """Convert FreeCAD solid to mesh with given resolution.
    
    Args:
        solid: FreeCAD solid shape
        resolution: Linear deflection (smaller = finer mesh)
        
    Returns:
        Mesh.Mesh object
    """
    mesh = MeshPart.meshFromShape(
        Shape=solid,
        LinearDeflection=resolution,
        AngularDeflection=0.5,  # radians
        Relative=False
    )
    
    return mesh


def subdivide_mesh(mesh, iterations=2):
    """Subdivide mesh to create more uniform vertex distribution.
    
    Uses Loop subdivision to add intermediate vertices, creating
    horizontal rings on cylindrical surfaces.
    
    Args:
        mesh: FreeCAD Mesh.Mesh object
        iterations: Number of subdivision passes (default 2)
        
    Returns:
        Subdivided Mesh.Mesh object
    """
    import trimesh
    
    # Convert FreeCAD mesh to trimesh
    vertices = np.array([[p.x, p.y, p.z] for p in mesh.Points])
    faces = np.array([[f[0], f[1], f[2]] for f in mesh.Topology[1]])
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    original_faces = len(tm.faces)
    
    # Apply subdivision iterations
    for _ in range(iterations):
        tm = tm.subdivide()
    
    FreeCAD.Console.PrintMessage(
        f"  Subdivided mesh: {original_faces} -> {len(tm.faces)} faces "
        f"({len(tm.vertices)} vertices)\n"
    )
    
    # Convert back to FreeCAD mesh
    triangles = []
    for face in tm.faces:
        v0 = tm.vertices[face[0]].tolist()
        v1 = tm.vertices[face[1]].tolist()
        v2 = tm.vertices[face[2]].tolist()
        triangles.append([v0, v1, v2])
    
    return Mesh.Mesh(triangles)


def mesh_to_numpy(mesh):
    """Convert FreeCAD Mesh to numpy arrays.
    
    Returns:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of triangle indices
    """
    vertices = np.array([[p.x, p.y, p.z] for p in mesh.Points])
    faces = np.array([[f[0], f[1], f[2]] for f in mesh.Topology[1]])
    
    return vertices, faces


def numpy_to_mesh(vertices, faces):
    """Convert numpy arrays back to FreeCAD Mesh.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of triangle indices
        
    Returns:
        Mesh.Mesh object
    """
    facets = []
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        facets.append([
            FreeCAD.Vector(v0[0], v0[1], v0[2]),
            FreeCAD.Vector(v1[0], v1[1], v1[2]),
            FreeCAD.Vector(v2[0], v2[1], v2[2])
        ])
    
    mesh = Mesh.Mesh(facets)
    return mesh


def mesh_to_solid(mesh):
    """Convert mesh back to FreeCAD solid.
    
    Creates a shell from the mesh and then a solid.
    """
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, 0.1)
    
    try:
        shape = shape.copy()
        shape.sewShape()
        
        if shape.Shells:
            solid = Part.makeSolid(shape.Shells[0])
            return solid
        else:
            FreeCAD.Console.PrintWarning("Could not create shells from mesh\n")
            return shape
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Could not create solid: {e}\n")
        return shape


def export_mesh_to_stl(mesh, filepath):
    """Export mesh to STL file."""
    mesh.write(filepath)


def smooth_mesh_vertices(vertices, faces, smoothing_factor=0.3, iterations=1):
    """Apply Laplacian smoothing to mesh vertices.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of triangle indices
        smoothing_factor: How much to move toward neighbors (0-1)
        iterations: Number of smoothing passes
        
    Returns:
        Smoothed vertices array
    """
    num_vertices = len(vertices)
    
    # Build adjacency list
    adjacency = [set() for _ in range(num_vertices)]
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Smooth
    smoothed = vertices.copy()
    for _ in range(iterations):
        new_positions = smoothed.copy()
        for i in range(num_vertices):
            if adjacency[i]:
                neighbors = list(adjacency[i])
                neighbor_center = np.mean(smoothed[neighbors], axis=0)
                new_positions[i] = (1 - smoothing_factor) * smoothed[i] + smoothing_factor * neighbor_center
        smoothed = new_positions
    
    return smoothed


def scale_mesh_from_center(vertices, scale_factor=0.8):
    """Scale mesh vertices toward center of mass.
    
    This creates a smaller starting shape inside the envelope,
    allowing bidirectional deformation (both inward and outward).
    
    Args:
        vertices: Nx3 array of vertex positions
        scale_factor: Scale factor (0.8 = 80% of original size)
        
    Returns:
        Scaled vertices array
    """
    center = np.mean(vertices, axis=0)
    scaled = center + (vertices - center) * scale_factor
    return scaled


def get_bounding_box(vertices):
    """Get bounding box of vertex array.
    
    Returns:
        Dict with 'min' and 'max' arrays
    """
    return {
        'min': np.min(vertices, axis=0),
        'max': np.max(vertices, axis=0)
    }


def constrain_to_bbox(vertices, bbox, margin=0.01):
    """Constrain vertices to lie within bounding box."""
    vertices = np.clip(vertices, bbox['min'] + margin, bbox['max'] - margin)
    return vertices


def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals from mesh.
    
    Returns:
        Nx3 array of vertex normals
    """
    # Initialize normal accumulator
    vertex_normals = np.zeros_like(vertices)
    
    # Compute face normals and accumulate at vertices
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        
        # Accumulate at each vertex
        for idx in face:
            vertex_normals[idx] += normal
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)  # Avoid division by zero
    vertex_normals = vertex_normals / norms
    
    return vertex_normals
