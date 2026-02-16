try:
    import os
    os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin/x64'))
except Exception:
    pass


"""
UV LED Ray Tracing through Light Pipe using CUDA
Supports STEP model import and GPU-accelerated ray-triangle intersection
"""

import numpy as np
import sys
import warnings

# Import standard library modules FIRST - these must work before any filtering
# If these fail, Python's environment is broken
try:
    import argparse
    import traceback
except ImportError as e:
    print(f"ERROR: Cannot import standard library modules: {e}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    raise

# Suppress CUDA compiler warnings
warnings.filterwarnings('ignore', message='The CUDA compiler succeeded')
warnings.filterwarnings('ignore', category=UserWarning, module='pycuda')

# CRITICAL: Filter sys.path to remove FreeCAD bin directories BEFORE importing PyCUDA
# This must happen AFTER all standard library imports (which are done above)
# but BEFORE PyCUDA import, so we find system Python's complete PyCUDA, not FreeCAD's incomplete one
if sys.path:
    filtered_paths = []
    for p in sys.path:
        if not p:
            filtered_paths.append(p)  # Keep empty string (current directory)
            continue
        p_lower = p.lower()
        # Only remove if it's VERY clearly a FreeCAD bin directory
        # Must contain 'freecad', contain 'bin', AND be in a FreeCAD installation path
        is_freecad_bin = (
            'freecad' in p_lower and
            'bin' in p_lower and
            ('program files' in p_lower or ('freecad' in os.path.basename(os.path.dirname(p)).lower() if os.path.dirname(p) else False)) and
            (p_lower.endswith('bin') or p_lower.endswith('bin\\') or p_lower.endswith('bin/') or
             '\\bin\\' in p_lower or '/bin/' in p_lower)
        )
        if not is_freecad_bin:
            filtered_paths.append(p)
    sys.path = filtered_paths

try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    
    # Check if this is FreeCAD's incomplete PyCUDA by checking the module location
    pycuda_path = ''
    try:
        if hasattr(cuda, '__file__') and cuda.__file__:
            pycuda_path = os.path.dirname(os.path.abspath(cuda.__file__))
    except:
        pass
    
    if pycuda_path and 'freecad' in pycuda_path.lower():
        # This is FreeCAD's incomplete PyCUDA - reject it
        raise ImportError(f"FreeCAD's incomplete PyCUDA found at {pycuda_path}. System Python's PyCUDA is required.")
    
    # Try to actually use PyCUDA - this will fail if it's FreeCAD's incomplete version
    try:
        cuda.init()
        # If we get here, PyCUDA works
        CUDA_AVAILABLE = True
    except Exception as e:
        # PyCUDA imported but can't be used (likely FreeCAD's incomplete version)
        CUDA_AVAILABLE = False
        print("ERROR: PyCUDA imported but cannot be used (may be FreeCAD's incomplete version).", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print("ERROR: CUDA is required. Aborting operation.", file=sys.stderr)
        sys.exit(1)
        
except ImportError as e:
    CUDA_AVAILABLE = False
    error_msg = str(e)
    if 'freecad' in error_msg.lower():
        print(f"ERROR: {error_msg}", file=sys.stderr)
    else:
        print("ERROR: PyCUDA not available.", file=sys.stderr)
        print(f"ERROR: ImportError: {e}", file=sys.stderr)
    print("ERROR: CUDA is required. Aborting operation.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Catch any other errors during PyCUDA import/initialization
    CUDA_AVAILABLE = False
    print("ERROR: PyCUDA not available.", file=sys.stderr)
    print(f"ERROR: {e}", file=sys.stderr)
    print("ERROR: CUDA is required. Aborting operation.", file=sys.stderr)
    sys.exit(1)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. STEP import will not work.")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, List
import struct


# ============ ADD CUDA INITIALIZATION HERE ============
if CUDA_AVAILABLE:
    try:
        # Try to actually use PyCUDA - this will fail if it's FreeCAD's incomplete version
        cuda.init()
        device = cuda.Device(0)
        try:
            ctx = cuda.Device(0).retain_primary_context()
            print(f"✓ Using existing CUDA context on {device.name()}")
        except:
            ctx = device.make_context()
            print(f"✓ Created new CUDA context on {device.name()}")

        print(f"✓ CUDA initialized: {device.name()}")
        print(f"✓ Compute Capability: {device.compute_capability()}")
        print(f"✓ Total Memory: {device.total_memory() // 1024**2} MB")
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:
        # PyCUDA imported but can't be used (likely FreeCAD's incomplete version)
        CUDA_AVAILABLE = False
        print("ERROR: PyCUDA imported but cannot be used (may be FreeCAD's incomplete version).", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print("ERROR: CUDA is required. Aborting operation.", file=sys.stderr)
        sys.exit(1)


@dataclass
class Material:
    """Material properties for ray tracing"""
    refractive_index: float = 1.5  # Typical for acrylic/PMMA
    absorption_coeff: float = 0.01  # UV absorption per unit length
    reflectance: float = 0.04  # Fresnel reflectance at normal incidence


# ============================================================================
# NVSU119CT LED DIRECTIVITY PATTERNS (from datasheet pages 13-16)
# ============================================================================
# Measured at Ts=25°C, IF=700mA
# Angle (degrees from optical axis) -> Relative radiant intensity (0-1)
# Each model has a unique radiation pattern based on die and lens characteristics

LED_DIRECTIVITY = {
    # U375 (375nm) - Page 13: Slightly narrower beam pattern
    "U375": {
        "angles": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float32),
        "values": np.array([1.00, 0.97, 0.92, 0.85, 0.75, 0.62, 0.48, 0.32, 0.18, 0.08], dtype=np.float32),
    },
    # U385 (385nm) - Page 14: Intermediate pattern
    "U385": {
        "angles": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float32),
        "values": np.array([1.00, 0.98, 0.94, 0.88, 0.80, 0.70, 0.55, 0.38, 0.22, 0.09], dtype=np.float32),
    },
    # U395 (395nm) - Page 15: Similar to U385
    "U395": {
        "angles": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float32),
        "values": np.array([1.00, 0.98, 0.95, 0.90, 0.83, 0.73, 0.58, 0.40, 0.24, 0.09], dtype=np.float32),
    },
    # U405 (405nm) - Page 16: Widest beam pattern (original data)
    "U405": {
        "angles": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float32),
        "values": np.array([1.00, 0.98, 0.96, 0.95, 0.93, 0.89, 0.75, 0.55, 0.30, 0.10], dtype=np.float32),
    },
}

# Default model for backward compatibility
DEFAULT_LED_MODEL = "U405"


def get_directivity(angle_deg, model: str = None):
    """Get relative radiant intensity at given angle using linear interpolation.
    
    Uses the NVSU119CT directivity pattern from the datasheet for the specified model.
    
    Args:
        angle_deg: Angle from optical axis in degrees (0-90)
        model: LED model identifier ("U375", "U385", "U395", or "U405")
               If None, uses DEFAULT_LED_MODEL
        
    Returns:
        Relative intensity (0-1), where 1.0 is on-axis intensity
    """
    if model is None:
        model = DEFAULT_LED_MODEL
    
    if model not in LED_DIRECTIVITY:
        print(f"Warning: Unknown LED model '{model}', using {DEFAULT_LED_MODEL}")
        model = DEFAULT_LED_MODEL
    
    directivity = LED_DIRECTIVITY[model]
    
    # Clamp angle to valid range
    angle_deg = np.clip(angle_deg, 0, 90)
    # Linear interpolation
    return np.interp(angle_deg, directivity["angles"], directivity["values"])


@dataclass
class LEDSource:
    """UV LED light source properties
    
    Designed to accurately model high-power UV LEDs like the Nichia NVSU119CT series.
    Supports both point source and realistic area source emission.
    
    Emission distribution options:
    - use_datasheet_directivity=True: Uses actual measured directivity pattern (most accurate)
    - use_datasheet_directivity=False, lambertian=True: Lambertian (cosine-weighted) approximation
    - use_datasheet_directivity=False, lambertian=False: Uniform distribution within cone
    """
    position: np.ndarray  # 3D position (center of emitting area)
    direction: np.ndarray  # Main emission direction (surface normal)
    power: float = 1.0  # Radiant power in mW (e.g., 1420 for NVSU119CT-U405 @ 700mA)
    wavelength: float = 405e-9  # Peak wavelength in meters (405nm for NVSU119CT-U405)
    half_angle: float = 70.0  # Half-angle of emission cone in degrees (70° = 140° viewing angle)
    emitter_size: float = 1.0  # Size of square emitting area in mm (typical ~1mm for high-power LEDs)
    lambertian: bool = True  # Use Lambertian (cosine-weighted) distribution (if not using datasheet)
    use_datasheet_directivity: bool = True  # Use actual NVSU119CT directivity pattern (most accurate)
    model: str = "U405"  # LED model for directivity pattern selection (U375/U385/U395/U405)


class CUDARayTracer:
    """GPU-accelerated ray tracer using CUDA"""
    
    def __init__(self, mesh_vertices, mesh_faces, material: Material):
        """
        Initialize CUDA ray tracer
        
        Args:
            mesh_vertices: Nx3 array of vertex positions
            mesh_faces: Mx3 array of triangle indices
            material: Material properties
        """
        self.vertices = np.array(mesh_vertices, dtype=np.float32)
        self.faces = np.array(mesh_faces, dtype=np.int32)
        self.material = material
        
        if not CUDA_AVAILABLE:
            print("ERROR: CUDA not available. CUDA is required for ray tracing.", file=sys.stderr)
            print("ERROR: Aborting operation.", file=sys.stderr)
            sys.exit(1)
        
        cuda.Context.push(cuda.Device(0).retain_primary_context())

        from pycuda.compiler import compile as cuda_compile
        import pycuda.driver as drv
        
        kernel_code = """
        __device__ bool ray_triangle_intersect(
            float3 ray_origin, float3 ray_dir,
            float3 v0, float3 v1, float3 v2,
            float* t, float* u, float* v)
        {
            const float EPSILON = 0.0000001f;
            
            float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            
            float3 h = make_float3(
                ray_dir.y * edge2.z - ray_dir.z * edge2.y,
                ray_dir.z * edge2.x - ray_dir.x * edge2.z,
                ray_dir.x * edge2.y - ray_dir.y * edge2.x
            );
            
            float a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;
            
            if (a > -EPSILON && a < EPSILON)
                return false;
            
            float f = 1.0f / a;
            float3 s = make_float3(
                ray_origin.x - v0.x,
                ray_origin.y - v0.y,
                ray_origin.z - v0.z
            );
            
            *u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
            if (*u < 0.0f || *u > 1.0f)
                return false;
            
            float3 q = make_float3(
                s.y * edge1.z - s.z * edge1.y,
                s.z * edge1.x - s.x * edge1.z,
                s.x * edge1.y - s.y * edge1.x
            );
            
            *v = f * (ray_dir.x * q.x + ray_dir.y * q.y + ray_dir.z * q.z);
            if (*v < 0.0f || *u + *v > 1.0f)
                return false;
            
            *t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);
            
            return *t > EPSILON;
        }
        
        __global__ void trace_rays(
            float* ray_origins, float* ray_directions,
            float* vertices, int* faces,
            int num_rays, int num_triangles,
            float* hit_distances, int* hit_triangle_ids,
            float* hit_points)
        {
            int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (ray_idx >= num_rays)
                return;
            
            float3 ray_origin = make_float3(
                ray_origins[ray_idx * 3],
                ray_origins[ray_idx * 3 + 1],
                ray_origins[ray_idx * 3 + 2]
            );
            
            float3 ray_dir = make_float3(
                ray_directions[ray_idx * 3],
                ray_directions[ray_idx * 3 + 1],
                ray_directions[ray_idx * 3 + 2]
            );
            
            float closest_t = 1e10f;
            int closest_triangle = -1;
            
            for (int tri_idx = 0; tri_idx < num_triangles; tri_idx++) {
                int v0_idx = faces[tri_idx * 3];
                int v1_idx = faces[tri_idx * 3 + 1];
                int v2_idx = faces[tri_idx * 3 + 2];
                
                float3 v0 = make_float3(
                    vertices[v0_idx * 3],
                    vertices[v0_idx * 3 + 1],
                    vertices[v0_idx * 3 + 2]
                );
                float3 v1 = make_float3(
                    vertices[v1_idx * 3],
                    vertices[v1_idx * 3 + 1],
                    vertices[v1_idx * 3 + 2]
                );
                float3 v2 = make_float3(
                    vertices[v2_idx * 3],
                    vertices[v2_idx * 3 + 1],
                    vertices[v2_idx * 3 + 2]
                );
                
                float t, u, v;
                if (ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2, &t, &u, &v)) {
                    if (t < closest_t) {
                        closest_t = t;
                        closest_triangle = tri_idx;
                    }
                }
            }
            
            hit_distances[ray_idx] = closest_t;
            hit_triangle_ids[ray_idx] = closest_triangle;
            
            if (closest_triangle >= 0) {
                hit_points[ray_idx * 3] = ray_origin.x + ray_dir.x * closest_t;
                hit_points[ray_idx * 3 + 1] = ray_origin.y + ray_dir.y * closest_t;
                hit_points[ray_idx * 3 + 2] = ray_origin.z + ray_dir.z * closest_t;
            }
        }
        """
        
        import tempfile
        import os

        ptx_code = cuda_compile(
            kernel_code,
            nvcc='nvcc',
            target='ptx',
            arch='sm_86',
            keep=False,
            cache_dir=False
        )

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.ptx', delete=False, encoding='utf-8')
        temp_file.write(ptx_code if isinstance(ptx_code, str) else ptx_code.decode('utf-8'))
        temp_file.close()

        self.cuda_kernel = drv.module_from_file(temp_file.name)
        os.unlink(temp_file.name)

        self.trace_rays_kernel = self.cuda_kernel.get_function("trace_rays")
    
    def trace(self, ray_origins, ray_directions):
        """Trace rays through the geometry"""
        if not CUDA_AVAILABLE:
            print("ERROR: CUDA not available in trace() method. CUDA is required.", file=sys.stderr)
            print("ERROR: Aborting operation.", file=sys.stderr)
            sys.exit(1)
        
        num_rays = len(ray_origins)
        num_triangles = len(self.faces)
        
        hit_distances = np.ones(num_rays, dtype=np.float32) * 1e10
        hit_triangle_ids = np.ones(num_rays, dtype=np.int32) * -1
        hit_points = np.zeros((num_rays, 3), dtype=np.float32)
        
        ray_origins_gpu = cuda.mem_alloc(ray_origins.nbytes)
        ray_directions_gpu = cuda.mem_alloc(ray_directions.nbytes)
        vertices_gpu = cuda.mem_alloc(self.vertices.nbytes)
        faces_gpu = cuda.mem_alloc(self.faces.nbytes)
        hit_distances_gpu = cuda.mem_alloc(hit_distances.nbytes)
        hit_triangle_ids_gpu = cuda.mem_alloc(hit_triangle_ids.nbytes)
        hit_points_gpu = cuda.mem_alloc(hit_points.nbytes)
        
        cuda.memcpy_htod(ray_origins_gpu, ray_origins)
        cuda.memcpy_htod(ray_directions_gpu, ray_directions)
        cuda.memcpy_htod(vertices_gpu, self.vertices)
        cuda.memcpy_htod(faces_gpu, self.faces)
        
        block_size = 256
        grid_size = (num_rays + block_size - 1) // block_size
        
        self.trace_rays_kernel(
            ray_origins_gpu, ray_directions_gpu,
            vertices_gpu, faces_gpu,
            np.int32(num_rays), np.int32(num_triangles),
            hit_distances_gpu, hit_triangle_ids_gpu, hit_points_gpu,
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        cuda.memcpy_dtoh(hit_distances, hit_distances_gpu)
        cuda.memcpy_dtoh(hit_triangle_ids, hit_triangle_ids_gpu)
        cuda.memcpy_dtoh(hit_points, hit_points_gpu)
        
        return hit_distances, hit_triangle_ids, hit_points
    
    def _trace_cpu(self, ray_origins, ray_directions):
        """CPU fallback for ray tracing"""
        num_rays = len(ray_origins)
        hit_distances = np.ones(num_rays) * 1e10
        hit_triangle_ids = np.ones(num_rays, dtype=int) * -1
        hit_points = np.zeros((num_rays, 3))
        
        for ray_idx in range(num_rays):
            origin = ray_origins[ray_idx]
            direction = ray_directions[ray_idx]
            
            closest_t = 1e10
            closest_tri = -1
            
            for tri_idx, face in enumerate(self.faces):
                v0, v1, v2 = self.vertices[face]
                t, hit = self._ray_triangle_intersect(origin, direction, v0, v1, v2)
                
                if hit and t < closest_t:
                    closest_t = t
                    closest_tri = tri_idx
            
            hit_distances[ray_idx] = closest_t
            hit_triangle_ids[ray_idx] = closest_tri
            if closest_tri >= 0:
                hit_points[ray_idx] = origin + direction * closest_t
        
        return hit_distances, hit_triangle_ids, hit_points
    
    @staticmethod
    def _ray_triangle_intersect(origin, direction, v0, v1, v2):
        """Möller-Trumbore ray-triangle intersection"""
        EPSILON = 1e-7
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(direction, edge2)
        a = np.dot(edge1, h)
        
        if abs(a) < EPSILON:
            return float('inf'), False
        
        f = 1.0 / a
        s = origin - v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return float('inf'), False
        
        q = np.cross(s, edge1)
        v = f * np.dot(direction, q)
        
        if v < 0.0 or u + v > 1.0:
            return float('inf'), False
        
        t = f * np.dot(edge2, q)
        
        if t > EPSILON:
            return t, True
        
        return float('inf'), False


class UVLEDRayTracer:
    """Main ray tracing simulation class"""
    
    def __init__(self, step_file_path: str = None):
        self.mesh = None
        self.material = Material()
        self.led_source = None
        self.ray_tracer = None
        
        if step_file_path:
            self.load_step_model(step_file_path)
    
    def load_step_model(self, file_path: str):
        """Load geometry from STEP or STL file"""
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for STEP/STL file import")
        
        file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        print(f"Loading {'STL' if file_ext == 'stl' else 'STEP'} model from {file_path}...")
        try:
            self.mesh = trimesh.load(file_path, force='mesh')
            # STL from FreeCAD is in mm - no scaling needed
            # STEP files may be in meters - scale 1000x to mm
            if file_ext != 'stl':
                self.mesh.apply_scale(1000.0)
                print(f"Scaled STEP mesh 1000x (m → mm)")
            print(f"Loaded mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
            
            self.ray_tracer = CUDARayTracer(
                self.mesh.vertices,
                self.mesh.faces,
                self.material
            )
        except Exception as e:
            print(f"Error loading STEP file: {e}")
            raise
        
    def setup_led_source(self, position, direction, power=1.0, wavelength=405e-9, half_angle=70.0,
                         emitter_size=1.0, lambertian=True, use_datasheet_directivity=True,
                         model="U405"):
        """Configure UV LED source with realistic emission model.
        
        Args:
            position: 3D position of LED center (mm)
            direction: Main emission direction (will be normalized)
            power: Radiant power in mW (default 1420 for NVSU119CT-U405)
            wavelength: Peak wavelength in meters (default 405nm)
            half_angle: Half-angle of emission cone in degrees (default 70° = 140° viewing angle)
            emitter_size: Size of square emitting area in mm (default 1.0mm)
            lambertian: Use Lambertian (cosine-weighted) distribution (if not using datasheet)
            use_datasheet_directivity: Use actual NVSU119CT directivity pattern (default True, most accurate)
            model: LED model identifier for directivity pattern (U375/U385/U395/U405)
        """
        self.led_source = LEDSource(
            position=np.array(position, dtype=np.float32),
            direction=np.array(direction, dtype=np.float32),
            power=power,
            wavelength=wavelength,
            half_angle=half_angle,
            emitter_size=emitter_size,
            lambertian=lambertian,
            use_datasheet_directivity=use_datasheet_directivity,
            model=model
        )
        self.led_source.direction /= np.linalg.norm(self.led_source.direction)
    
    def generate_led_rays(self, num_rays=10000):
        """Generate rays from LED source with realistic emission model.
        
        Features:
        - NVSU119CT-U405 datasheet directivity pattern (most accurate, default)
        - Lambertian (cosine-weighted) angular distribution (approximation)
        - Uniform distribution within cone (simplest)
        - Finite emitting area (square chip) when emitter_size > 0
        - Properly bounded by half_angle viewing cone
        
        The directivity pattern uses rejection sampling with the actual
        measured radiation pattern from the NVSU119CT-U405 datasheet.
        """
        if self.led_source is None:
            raise ValueError("LED source not configured. Call setup_led_source() first.")
        
        # Pre-compute coordinate system once (more efficient)
        z_axis = self.led_source.direction
        
        if abs(z_axis[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        x_axis = np.cross(temp, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Pre-compute values
        half_angle_rad = np.radians(self.led_source.half_angle)
        max_cos_theta = np.cos(half_angle_rad)  # For clamping Lambertian to viewing cone
        emitter_half = self.led_source.emitter_size / 2.0
        
        rays_origin = []
        rays_direction = []
        
        for _ in range(num_rays):
            # === ANGULAR DISTRIBUTION ===
            if self.led_source.use_datasheet_directivity:
                # Use actual NVSU119CT directivity pattern with rejection sampling
                # This produces the most accurate emission distribution for each model
                max_directivity = 1.0  # Maximum is at 0 degrees
                while True:
                    # Generate candidate angle (uniform in degrees for simplicity)
                    theta_deg = np.random.uniform(0, 90)
                    theta = np.radians(theta_deg)
                    
                    # Get directivity at this angle for the specific LED model
                    directivity = get_directivity(theta_deg, self.led_source.model)
                    
                    # Accept with probability proportional to directivity
                    if np.random.uniform(0, max_directivity) < directivity:
                        # Also check if within half-angle cone
                        if theta <= half_angle_rad:
                            break
                        # If outside half-angle but within 90°, keep trying
            elif self.led_source.lambertian:
                # Lambertian (cosine-weighted) distribution
                # Uses inverse transform sampling: theta = arcsin(sqrt(u))
                # This produces more rays near the normal (realistic LED behavior)
                # Clamp to viewing angle cone
                u = np.random.uniform(0, 1)
                # Scale u to only sample within the half-angle cone
                # For Lambertian: intensity ~ cos(theta), so CDF ~ sin²(theta)
                max_sin_sq = np.sin(half_angle_rad) ** 2
                scaled_u = u * max_sin_sq
                theta = np.arcsin(np.sqrt(scaled_u))
            else:
                # Uniform distribution within cone (original behavior)
                theta = np.random.uniform(0, half_angle_rad)
            
            phi = np.random.uniform(0, 2 * np.pi)
            
            # Convert spherical to Cartesian (local coordinates)
            local_dir = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            # Transform to world coordinates
            world_dir = (local_dir[0] * x_axis + 
                        local_dir[1] * y_axis + 
                        local_dir[2] * z_axis)
            
            # === SPATIAL DISTRIBUTION (Area Source) ===
            if self.led_source.emitter_size > 0:
                # Randomize origin within square emitting area
                # Offset perpendicular to emission direction
                offset_x = np.random.uniform(-emitter_half, emitter_half)
                offset_y = np.random.uniform(-emitter_half, emitter_half)
                origin = (self.led_source.position + 
                         offset_x * x_axis + 
                         offset_y * y_axis)
            else:
                # Point source (emitter_size = 0)
                origin = self.led_source.position
            
            rays_origin.append(origin)
            rays_direction.append(world_dir)
        
        return np.array(rays_origin, dtype=np.float32), np.array(rays_direction, dtype=np.float32)
    
    def _calculate_refraction(self, ray_dir, normal, n1, n2):
        """Calculate refraction and reflection at interface using Snell's Law and Fresnel"""
        cos_i = -np.dot(normal, ray_dir)
        if cos_i < 0:
            normal = -normal
            cos_i = -cos_i
            n1, n2 = n2, n1
        
        n_ratio = n1 / n2
        sin_t_squared = n_ratio * n_ratio * (1.0 - cos_i * cos_i)
        
        if sin_t_squared > 1.0:
            reflected_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
            return reflected_dir, None, 1.0
        
        cos_t = np.sqrt(1.0 - sin_t_squared)
        refracted_dir = n_ratio * ray_dir + (n_ratio * cos_i - cos_t) * normal
        refracted_dir = refracted_dir / np.linalg.norm(refracted_dir)
        
        r0 = ((n1 - n2) / (n1 + n2)) ** 2
        reflectance = r0 + (1 - r0) * (1 - cos_i) ** 5
        
        reflected_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
        
        return reflected_dir, refracted_dir, reflectance
    
    def visualize_interactive_full(self, ray_paths=None, show_bounces=None, only_exit_rays=False, max_segments=None):
        """Simple 2-color visualization: RED for initial, BLUE for rest + spheres at exits"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.lines import Line2D
        
        # Downsample if too many segments
        if ray_paths is not None and max_segments and len(ray_paths) > max_segments:
            print(f"  Downsampling 3D viz: {len(ray_paths):,} → {max_segments:,} segments")
            indices = np.random.choice(len(ray_paths), max_segments, replace=False)
            ray_paths = [ray_paths[i] for i in sorted(indices)]
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if self.mesh is not None:
            vertices = self.mesh.vertices
            faces = self.mesh.faces
            
            mesh_faces = [vertices[face] for face in faces]
            mesh_collection = Poly3DCollection(mesh_faces, 
                                            alpha=0.02,
                                            facecolor='lightgray', 
                                            edgecolor='gray',   
                                            linewidth=0.2,
                                            rasterized=True)  # Rasterize mesh for speed
            ax.add_collection3d(mesh_collection)
        
        exit_points = []
        
        if ray_paths is not None and len(ray_paths) > 0:
            initial_count = 0
            other_count = 0
            exit_count = 0
            
            for path in ray_paths:
                bounce_num = path.get('bounce', 0)
                is_exit = path.get('is_exit', False)
                
                if show_bounces is not None and bounce_num not in show_bounces:
                    continue
                
                if only_exit_rays and not is_exit:
                    continue
                
                distance = np.linalg.norm(path['end'] - path['start'])
                if distance < 0.01:
                    continue
                
                if bounce_num == 0:
                    color = 'red'
                    linewidth = 2.0
                    alpha = 0.9
                    initial_count += 1
                else:
                    color = 'blue'
                    linewidth = 1.5
                    alpha = 0.7
                    other_count += 1
                
                if is_exit:
                    exit_points.append(path['end'])
                    exit_count += 1
                
                ax.plot3D(
                    [path['start'][0], path['end'][0]],
                    [path['start'][1], path['end'][1]],
                    [path['start'][2], path['end'][2]],
                    color=color, 
                    alpha=alpha,
                    linewidth=linewidth
                )
            
            print(f"\nSegments plotted:")
            print(f"  🔴 Initial (red):  {initial_count}")
            print(f"  🔵 Other (blue):   {other_count}")
            print(f"  ● Exit points:     {exit_count}")
            print(f"  Total:             {initial_count + other_count}")
        
        if len(exit_points) > 0:
            exit_points = np.array(exit_points)
            ax.scatter(exit_points[:, 0], 
                      exit_points[:, 1], 
                      exit_points[:, 2],
                      c='yellow',
                      s=100,
                      alpha=0.9,
                      edgecolors='orange',
                      linewidths=2,
                      marker='o',
                      label='Exit Points')
            print(f"  Plotted {len(exit_points)} yellow exit spheres")
        
        if self.led_source is not None:
            ax.scatter(*self.led_source.position, c='darkred', s=150, marker='o', 
                      label='UV LED', edgecolors='black', linewidths=2, zorder=100)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        title = 'UV LED Ray Tracing'
        if show_bounces is not None:
            title += f' (Bounces: {show_bounces})'
        if only_exit_rays:
            title += ' [EXIT RAYS ONLY]'
        ax.set_title(title)
        
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Initial ray (air)'),
            Line2D([0], [0], color='blue', lw=1.5, label='All other segments'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                   markersize=10, markeredgecolor='orange', markeredgewidth=2, 
                   label='Exit points', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_box_aspect([1,1,2])
        plt.tight_layout()
        return fig
    
    def analyze_exit_hotspots(self, ray_paths, rays_entered=None, rays_exited=None, rays_absorbed=None, exit_events=None):
        """Analyze where light exits the cylinder and create heatmaps"""
        exit_points = []
        exit_intensities = []
        exit_bounces = []
        
        for path in ray_paths:
            if not path.get('is_exit', False):
                continue
            
            distance = np.linalg.norm(path['end'] - path['start'])
            if distance < 0.01:
                continue
            
            exit_points.append(path['end'])
            exit_intensities.append(path['intensity'])
            exit_bounces.append(path.get('bounce', 0))
        
        if len(exit_points) == 0:
            print("No exit points found!")
            return None
        
        exit_points = np.array(exit_points)
        exit_intensities = np.array(exit_intensities)
        exit_bounces = np.array(exit_bounces)
        
        print(f"Found {len(exit_points)} exit points")
        
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        colors = ['white', 'lavender', 'mediumpurple', 'darkviolet', 'indigo']
        cmap = LinearSegmentedColormap.from_list('purple_intensity', colors, N=100)
        
        fig = plt.figure(figsize=(18, 10))
        
        # Z-distribution
        ax1 = fig.add_subplot(2, 3, 1)
        z_bins = np.linspace(exit_points[:, 2].min(), exit_points[:, 2].max(), 30)
        z_hist, z_edges = np.histogram(exit_points[:, 2], bins=z_bins, weights=exit_intensities)
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        ax1.bar(z_centers, z_hist, width=np.diff(z_edges), color='mediumpurple', alpha=0.7, edgecolor='indigo')
        ax1.set_xlabel('Z Position (mm)')
        ax1.set_ylabel('Light Intensity')
        ax1.set_title('Light Output Along Pipe Length')
        ax1.grid(True, alpha=0.3)
        
        # Angular distribution
        ax2 = fig.add_subplot(2, 3, 2, projection='polar')
        theta = np.arctan2(exit_points[:, 1], exit_points[:, 0])
        theta_bins = np.linspace(-np.pi, np.pi, 36)
        theta_hist, theta_edges = np.histogram(theta, bins=theta_bins, weights=exit_intensities)
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        ax2.bar(theta_centers, theta_hist, width=np.diff(theta_edges), color='mediumpurple', alpha=0.7, edgecolor='indigo')
        ax2.set_title('Circumferential Distribution', pad=20)
        
        # Reflection number
        ax3 = fig.add_subplot(2, 3, 3)
        bounce_bins = np.arange(0, max(exit_bounces) + 2)
        bounce_hist, _ = np.histogram(exit_bounces, bins=bounce_bins, weights=exit_intensities)
        ax3.bar(bounce_bins[:-1], bounce_hist, color='mediumpurple', alpha=0.7, edgecolor='indigo')
        ax3.set_xlabel('Reflection Number')
        ax3.set_ylabel('Light Intensity')
        ax3.set_title('Light Output vs Reflection Number')
        ax3.grid(True, alpha=0.3)
        
        # 3D scatter
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        
        if self.mesh is not None:
            vertices = self.mesh.vertices
            faces = self.mesh.faces
            mesh_faces = [vertices[face] for face in faces]
            mesh_collection = Poly3DCollection(mesh_faces, 
                                            alpha=0.08,
                                            facecolor='lightgray', 
                                            edgecolor='darkslategray',
                                            linewidth=0.8)
            ax4.add_collection3d(mesh_collection)
        
        scatter = ax4.scatter(exit_points[:, 0], exit_points[:, 1], exit_points[:, 2],
                            c=exit_intensities, cmap=cmap, s=50, alpha=0.8)
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        ax4.set_zlabel('Z (mm)')
        ax4.set_title('3D Exit Points')
        ax4.view_init(elev=20, azim=45)
        ax4.set_box_aspect([1,1,2])
        plt.colorbar(scatter, ax=ax4, label='Intensity', shrink=0.6)
        
        # 2D heatmap
        ax5 = fig.add_subplot(2, 3, 5)
        theta_deg = np.degrees(theta)
        H, xedges, yedges = np.histogram2d(exit_points[:, 2], theta_deg, bins=[30, 36], weights=exit_intensities)
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        im = ax5.imshow(H, extent=extent, origin='lower', aspect='auto', cmap=cmap, interpolation='bilinear')
        ax5.set_xlabel('Angle (degrees)')
        ax5.set_ylabel('Z Position (mm)')
        ax5.set_title('Exit Intensity Heatmap')
        plt.colorbar(im, ax=ax5, label='Intensity')
        
        # Statistics - show full simulation report
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        # Build comprehensive stats text
        stats_lines = ["SIMULATION REPORT", "═"*35, ""]
        
        if rays_entered is not None:
            stats_lines.append("Overall Efficiency:")
            stats_lines.append(f"  Rays entered:    {rays_entered:,}")
            stats_lines.append(f"  Rays exited:     {rays_exited:,} ({rays_exited/rays_entered*100:.1f}%)")
            stats_lines.append(f"  Rays absorbed:   {rays_absorbed:,} ({rays_absorbed/rays_entered*100:.1f}%)")
            if exit_events and rays_exited:
                stats_lines.append(f"  Exit cycles:     {exit_events/rays_exited:.1f}x avg")
            stats_lines.append("")
        
        stats_lines.append("Exit Hotspot Analysis:")
        stats_lines.append(f"  Exit points:     {len(exit_points):,}")
        stats_lines.append(f"  Light output:    {np.sum(exit_intensities):.3f}")
        stats_lines.append("")
        stats_lines.append("Z-Position:")
        stats_lines.append(f"  Peak at:         {z_centers[np.argmax(z_hist)]:.1f} mm")
        stats_lines.append(f"  Mean:            {np.average(exit_points[:, 2], weights=exit_intensities):.1f} mm")
        stats_lines.append("")
        stats_lines.append("Reflections:")
        stats_lines.append(f"  Most exits at:   {bounce_bins[np.argmax(bounce_hist)]}")
        stats_lines.append(f"  Mean bounces:    {np.average(exit_bounces, weights=exit_intensities):.1f}")
        
        stats_text = "\n".join(stats_lines)
        ax6.text(0.05, 0.5, stats_text, fontsize=9, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        return fig

    def simulate(self, num_rays=10000, max_bounces=1000):
        """Run ray tracing simulation with proper refraction.
        
        Intensity convention:
            Each ray is initialized with intensity = led_source.power (mW).
            After bouncing, intensities decrease due to absorption and Fresnel losses.
            Exit intensities are in mW. To get total power on absorber:
                total_power_mW = sum(exit_intensities) / num_rays
            This should equal efficiency * led_power_mW (power budget).
        
        Returns:
            ray_paths: List of dicts with 'start', 'end', 'intensity', 'bounce', 'is_exit'
            intensities: Final intensity array (for rays still active)
        """
        if self.ray_tracer is None:
            raise ValueError("Geometry not loaded.")
        
        print(f"Simulating {num_rays} rays with up to {max_bounces} bounces...")
        
        ray_origins, ray_directions = self.generate_led_rays(num_rays)
        ray_paths = []
        
        # Intensity convention: Each ray carries the full LED power (mW).
        # Downstream, irradiance is normalized by num_rays so that:
        #   - integrated_power = sum(exit_intensities) / num_rays  [mW]
        #   - This equals efficiency * led_power when power is conserved.
        # This allows irradiance (mW/cm²) to be computed as:
        #   grid_intensities / cell_area_cm2 / num_rays
        intensities = np.ones(num_rays) * self.led_source.power
        ray_in_material = np.zeros(num_rays, dtype=bool)
        
        for bounce in range(max_bounces):
            print(f"  Bounce {bounce + 1}/{max_bounces}...")
            
            if num_rays == 0:
                break
            
            hit_distances, hit_triangle_ids, hit_points = self.ray_tracer.trace(
                ray_origins, ray_directions
            )
            
            for i in range(num_rays):
                if hit_triangle_ids[i] >= 0:
                    distance = hit_distances[i]
                    intensities[i] *= np.exp(-self.material.absorption_coeff * distance)
            
            new_origins = []
            new_directions = []
            new_intensities = []
            new_in_material = []
            exit_flags = []

            for i in range(num_rays):
                if hit_triangle_ids[i] >= 0:
                    tri_idx = hit_triangle_ids[i]
                    face = self.ray_tracer.faces[tri_idx]
                    v0, v1, v2 = self.ray_tracer.vertices[face]
                    
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    normal = np.cross(edge1, edge2)
                    normal = normal / np.linalg.norm(normal)
                    
                    in_material = ray_in_material[i]
                    n1 = self.material.refractive_index if in_material else 1.0
                    n2 = 1.0 if in_material else self.material.refractive_index
                    
                    reflected_dir, refracted_dir, reflectance = self._calculate_refraction(
                        ray_directions[i], normal, n1, n2
                    )
                    
                    if np.random.random() < reflectance or refracted_dir is None:
                        new_origins.append(hit_points[i].copy())
                        new_directions.append(reflected_dir)
                        new_intensities.append(intensities[i] * reflectance)
                        new_in_material.append(in_material)
                        exit_flags.append(False)
                    else:
                        new_origins.append(hit_points[i].copy())
                        new_directions.append(refracted_dir)
                        new_intensities.append(intensities[i] * (1 - reflectance))
                        new_in_material.append(not in_material)
                        exit_flags.append(in_material)
                        
            for i in range(num_rays):
                if hit_triangle_ids[i] >= 0:
                    is_exit = exit_flags[i] if i < len(exit_flags) else False
                    
                    ray_paths.append({
                        'start': ray_origins[i].copy(),
                        'end': hit_points[i].copy(),
                        'intensity': intensities[i],
                        'bounce': bounce,
                        'is_exit': is_exit
                    })

            if len(new_origins) == 0:
                break

            ray_origins = np.array(new_origins, dtype=np.float32)
            ray_directions = np.array(new_directions, dtype=np.float32)
            intensities = np.array(new_intensities)
            ray_in_material = np.array(new_in_material, dtype=bool)
            num_rays = len(ray_origins)
        
        exit_count = sum(1 for r in ray_paths if r.get('is_exit', False))
        print(f"Simulation complete. Traced {len(ray_paths)} ray segments. Exits: {exit_count}")
        
        return ray_paths, intensities


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='UV LED Ray Tracing Simulation - Analyze light pipe optical performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with STEP file:
  python %(prog)s lightpipe.step
  
  # Custom ray count and bounce limit:
  python %(prog)s lightpipe.step --rays 10000 --bounces 500
  
  # Set LED position and direction:
  python %(prog)s lightpipe.step --led-pos 0 0 5 --led-dir 0 0 1
  
  # Custom LED half-angle and wavelength:
  python %(prog)s lightpipe.step --led-angle 45 --wavelength 365
  
  # Custom material properties (acrylic):
  python %(prog)s lightpipe.step --refractive-index 1.49 --absorption 0.015
  
  # Full custom simulation:
  python %(prog)s lightpipe.step --rays 50000 --bounces 2000 \\
      --led-pos 0 0 2 --led-dir 0 0 1 --led-angle 67.5 \\
      --refractive-index 1.5 --absorption 0.01
        """
    )
    
    # Positional argument
    parser.add_argument('step_file', 
                       help='Path to STEP model file (.step or .stp)')
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--rays', '-r', 
                          type=int, 
                          default=50000,
                          help='Number of rays to simulate (default: 50000)')
    sim_group.add_argument('--bounces', '-b', 
                          type=int, 
                          default=1000,
                          help='Maximum number of bounces per ray (default: 1000)')
    
    # LED source parameters
    led_group = parser.add_argument_group('LED Source Parameters')
    led_group.add_argument('--led-pos', 
                          nargs=3, 
                          type=float, 
                          default=[0, 0, 2],
                          metavar=('X', 'Y', 'Z'),
                          help='LED position in mm (default: 0 0 2)')
    led_group.add_argument('--led-dir', 
                          nargs=3, 
                          type=float, 
                          default=[0, 0, 1],
                          metavar=('X', 'Y', 'Z'),
                          help='LED direction vector (default: 0 0 1)')
    led_group.add_argument('--led-power', 
                          type=float, 
                          default=1420.0,
                          help='LED radiant power in mW (default: 1420 for NVSU119CT-U405)')
    led_group.add_argument('--led-angle', 
                          type=float, 
                          default=70.0,
                          help='LED half-angle in degrees (default: 70 for 140° viewing angle)')
    led_group.add_argument('--wavelength', '-w', 
                          type=float, 
                          default=405,
                          help='LED peak wavelength in nm (default: 405 for NVSU119CT-U405)')
    led_group.add_argument('--emitter-size', 
                          type=float, 
                          default=1.0,
                          help='LED emitter size in mm (default: 1.0, use 0 for point source)')
    led_group.add_argument('--lambertian', 
                          action='store_true',
                          default=True,
                          help='Use Lambertian (cosine-weighted) emission (fallback if not using datasheet)')
    led_group.add_argument('--no-lambertian', 
                          action='store_false',
                          dest='lambertian',
                          help='Use uniform emission within cone (disable Lambertian)')
    led_group.add_argument('--datasheet-directivity', 
                          action='store_true',
                          default=True,
                          help='Use actual NVSU119CT-U405 directivity pattern from datasheet (default: True, most accurate)')
    led_group.add_argument('--no-datasheet-directivity', 
                          action='store_false',
                          dest='datasheet_directivity',
                          help='Disable datasheet directivity (use Lambertian or uniform instead)')
    
    # Material parameters
    mat_group = parser.add_argument_group('Material Properties (PMMA/Acrylic defaults)')
    mat_group.add_argument('--refractive-index', '-n', 
                          type=float, 
                          default=1.5,
                          help='Refractive index (default: 1.5 for PMMA)')
    mat_group.add_argument('--absorption', '-a', 
                          type=float, 
                          default=0.01,
                          help='Absorption coefficient per mm (default: 0.01)')
    
    # Output options
    out_group = parser.add_argument_group('Output Options')
    out_group.add_argument('--output-prefix', '-o', 
                          default=None,
                          help='Prefix for output files (default: uses STEP filename)')
    out_group.add_argument('--no-display', 
                          action='store_true',
                          help='Do not display plots (save only)')
    out_group.add_argument('--skip-3d-viz',
                          action='store_true',
                          help='Skip 3D visualization (faster, only generate hotspot analysis)')
    out_group.add_argument('--max-viz-segments',
                          type=int,
                          default=50000,
                          help='Max segments to plot in 3D visualization (default: 50000)')
    out_group.add_argument('--dpi',
                          type=int,
                          default=150,
                          help='Plot DPI (default: 150, use 100 for faster)')
    
    return parser.parse_args()


def main():
    """Main entry point with CLI"""
    import os
    import time
    
    start_time = time.time()
    
    args = parse_arguments()
    
    # Set output prefix from STEP filename if not specified
    if args.output_prefix is None:
        args.output_prefix = os.path.splitext(os.path.basename(args.step_file))[0].replace('.', '_')
    
    # Validate STEP file
    if not args.step_file.lower().endswith(('.step', '.stp')):
        print(f"Warning: '{args.step_file}' doesn't have .step/.stp extension")
    
    # Print configuration
    print("\n" + "="*70)
    print("UV LED RAY TRACING SIMULATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  STEP file:         {args.step_file}")
    print(f"  Rays:              {args.rays:,}")
    print(f"  Max bounces:       {args.bounces:,}")
    print(f"\nLED Source:")
    print(f"  Position:          ({args.led_pos[0]}, {args.led_pos[1]}, {args.led_pos[2]}) mm")
    print(f"  Direction:         ({args.led_dir[0]}, {args.led_dir[1]}, {args.led_dir[2]})")
    print(f"  Power:             {args.led_power} mW")
    print(f"  Half-angle:        {args.led_angle}° ({args.led_angle * 2}° viewing angle)")
    print(f"  Wavelength:        {args.wavelength} nm")
    print(f"  Emitter size:      {args.emitter_size} mm" + (" (area source)" if args.emitter_size > 0 else " (point source)"))
    if args.datasheet_directivity:
        dist_type = "NVSU119CT-U405 datasheet pattern (most accurate)"
    elif args.lambertian:
        dist_type = "Lambertian (cosine-weighted)"
    else:
        dist_type = "Uniform"
    print(f"  Distribution:      {dist_type}")
    print(f"\nMaterial:")
    print(f"  Refractive index:  {args.refractive_index}")
    print(f"  Absorption coeff:  {args.absorption} /mm")
    print("="*70 + "\n")
    
    # Create ray tracer
    tracer = UVLEDRayTracer()
    
    # Set material properties
    tracer.material.refractive_index = args.refractive_index
    tracer.material.absorption_coeff = args.absorption
    
    # Load geometry
    try:
        tracer.load_step_model(args.step_file)
    except Exception as e:
        print(f"\n❌ Error loading STEP file: {e}")
        return 1
    
    # Setup LED with realistic emission model
    tracer.setup_led_source(
        position=args.led_pos,
        direction=args.led_dir,
        power=args.led_power,
        wavelength=args.wavelength * 1e-9,  # Convert nm to meters
        half_angle=args.led_angle,
        emitter_size=args.emitter_size,
        lambertian=args.lambertian,
        use_datasheet_directivity=args.datasheet_directivity
    )
    
    # Run simulation
    try:
        ray_paths, intensities = tracer.simulate(num_rays=args.rays, max_bounces=args.bounces)
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        return 1
    
    # Generate report
    print("\n" + "="*70)
    print("RAY TRACING SIMULATION REPORT")
    print("="*70)
    
    total_segments = len(ray_paths)
    initial_rays = sum(1 for p in ray_paths if p.get('bounce', 0) == 0)
    exit_segments = sum(1 for p in ray_paths if p.get('is_exit', False))
    blue_segments = sum(1 for p in ray_paths if p.get('bounce', 0) > 0)
    blue_non_exit = sum(1 for p in ray_paths if p.get('bounce', 0) > 0 and not p.get('is_exit', False))
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total ray segments:        {total_segments:,}")
    print(f"  Initial rays (RED):        {initial_rays:,}")
    print(f"  Blue segments total:       {blue_segments:,}")
    print(f"  Blue segments (exits):     {exit_segments:,}")
    print(f"  Blue segments (internal):  {blue_non_exit:,}")
    
    max_bounce_seen = max((p.get('bounce', 0) for p in ray_paths), default=0)
    rays_at_max = sum(1 for p in ray_paths 
                     if p.get('bounce', 0) > 0 
                     and not p.get('is_exit', False) 
                     and p.get('bounce', 0) >= args.bounces - 5)  # Check against actual limit
    
    print(f"\n🔄 BOUNCE ANALYSIS:")
    print(f"  Max bounces observed:      {max_bounce_seen}")
    print(f"  Max bounces allowed:       {args.bounces}")
    print(f"  Rays near limit:           {rays_at_max}")
    if rays_at_max > 0:
        print(f"  ⚠️  {rays_at_max} rays hit bounce limit! Consider increasing --bounces")
    elif max_bounce_seen > args.bounces * 0.8:
        print(f"  ⚠️  Max observed ({max_bounce_seen}) is >80% of limit. Consider more bounces.")
    else:
        print(f"  ✓ Good! Simulation completed naturally (max: {max_bounce_seen}/{args.bounces})")
    
    # Absorption analysis
    print(f"\n💀 ABSORPTION ANALYSIS:")
    low_intensity_rays = [p for p in ray_paths 
                         if p.get('intensity', 1.0) < 0.001 
                         and not p.get('is_exit', False) 
                         and p.get('bounce', 0) > 10]
    absorbed_count = len(low_intensity_rays)
    print(f"  Absorbed ray segments:     {absorbed_count}")
    if absorbed_count > 0:
        avg_bounce = np.mean([p['bounce'] for p in low_intensity_rays])
        print(f"  Avg bounces to death:      {avg_bounce:.1f}")
    
    # Simple ray accounting
    print(f"\n✨ LIGHT PIPE EFFICIENCY:")
    
    rays_entered = sum(1 for p in ray_paths if p.get('bounce', 0) == 0)
    rays_absorbed = len(low_intensity_rays)
    rays_exited = rays_entered - rays_absorbed  # Everything that didn't die, exited!
    exit_events = sum(1 for p in ray_paths if p.get('is_exit', False))
    
    print(f"  Rays entered:              {rays_entered:,}")
    print(f"  Rays exited:               {rays_exited:,} ({rays_exited/rays_entered*100:.1f}%)")
    print(f"  Rays absorbed (died):      {rays_absorbed:,} ({rays_absorbed/rays_entered*100:.1f}%)")
    print(f"  Total exit events:         {exit_events:,} (includes re-entries)")
    if exit_events > 0 and rays_exited > 0:
        print(f"  Avg exits per escaping ray:{exit_events/rays_exited:.1f}x")
    
    # Intensity statistics from ray segments (not final array which is empty)
    print(f"\n📉 INTENSITY STATISTICS:")
    all_intensities = [p['intensity'] for p in ray_paths]
    print(f"  Average intensity:         {np.mean(all_intensities):.4f}")
    print(f"  Min intensity:             {np.min(all_intensities):.4f}")
    print(f"  Max intensity:             {np.max(all_intensities):.4f}")
    
    print("\n" + "="*70 + "\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    if not args.skip_3d_viz:
        fig1 = tracer.visualize_interactive_full(ray_paths, 
                                                  only_exit_rays=False,
                                                  max_segments=args.max_viz_segments)
        output_file1 = f"{args.output_prefix}_{args.refractive_index}_{args.rays}_{args.bounces}_visualization.png"
        plt.savefig(output_file1, dpi=args.dpi, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file1}")
    else:
        print("  Skipping 3D visualization (--skip-3d-viz)")
    
    fig2 = tracer.analyze_exit_hotspots(ray_paths, 
                                          rays_entered=rays_entered,
                                          rays_exited=rays_exited, 
                                          rays_absorbed=rays_absorbed,
                                          exit_events=exit_events)
    if fig2:
        output_file2 = f"{args.output_prefix}_{args.refractive_index}_{args.rays}_{args.bounces}_hotspots.png"
        plt.savefig(output_file2, dpi=args.dpi, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file2}")
    
    if not args.no_display:
        print("\nDisplaying plots...")
        plt.show()
    else:
        plt.close('all')
    
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    
    if minutes > 0:
        print(f"\n✅ Simulation complete! Completed in {minutes}m {seconds:.1f}s\n")
    else:
        print(f"\n✅ Simulation complete! Completed in {seconds:.1f}s\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        if CUDA_AVAILABLE:
            ctx.pop()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if CUDA_AVAILABLE:
            ctx.pop()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        if CUDA_AVAILABLE:
            ctx.pop()
        sys.exit(1)