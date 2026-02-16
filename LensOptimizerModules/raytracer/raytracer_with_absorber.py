"""
Extended Ray Tracer with Absorber Material Support
=================================================
This extends raytracer-v2.2.0.py to support absorber materials that stop rays.

Key additions:
- Absorber face tracking
- Rays stop when hitting absorber faces
- Better exit detection
"""

import sys
import os
import traceback  # Import standard library modules
import argparse   # Import standard library modules

# CRITICAL: Filter sys.path to remove FreeCAD bin directories BEFORE loading raytracer-v2.2.0.py
# This must happen AFTER standard library imports but BEFORE the raytracer module is loaded
# because raytracer-v2.2.0.py imports PyCUDA at module level
# NOTE: This filtering is only needed if FreeCAD's Python is accidentally used.
# The FreeCAD macro should now use system Python 3.14 directly.
if sys.path:
    filtered_paths = []
    for p in sys.path:
        if not p:
            filtered_paths.append(p)  # Keep empty string (current directory)
            continue
        p_lower = p.lower()
        # Only remove if it's VERY clearly a FreeCAD bin directory
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

# Import the original raytracer
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Load the original raytracer module
import importlib.util
raytracer_path = os.path.join(script_dir, "raytracer-v2.2.0.py")

if not os.path.exists(raytracer_path):
    raise FileNotFoundError(f"raytracer-v2.2.0.py not found at {raytracer_path}")

spec = importlib.util.spec_from_file_location("raytracer_v2_2_0", raytracer_path)
raytracer_module = importlib.util.module_from_spec(spec)

# Store reference to original module for cleanup
_original_raytracer_module = raytracer_module

# Set UTF-8 encoding for stdout/stderr
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Execute the module
import contextlib

@contextlib.contextmanager
def safe_stdout_stderr():
    """Context manager to safely handle Unicode in stdout/stderr"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class SafeTextIO:
        def __init__(self, original):
            self.original = original
            self.buffer = original.buffer if hasattr(original, 'buffer') else None
        
        def write(self, text):
            try:
                self.original.write(text)
            except UnicodeEncodeError:
                safe_text = text.encode('ascii', 'replace').decode('ascii')
                self.original.write(safe_text)
        
        def flush(self):
            self.original.flush()
        
        def __getattr__(self, name):
            return getattr(self.original, name)
    
    sys.stdout = SafeTextIO(original_stdout)
    sys.stderr = SafeTextIO(original_stderr)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

try:
    with safe_stdout_stderr():
        spec.loader.exec_module(raytracer_module)
except Exception as e:
    print(f"ERROR: Failed to load raytracer module: {e}", file=sys.stderr)
    traceback.print_exc()  # traceback already imported at top
    sys.exit(1)

# Import classes from the original module
UVLEDRayTracer = raytracer_module.UVLEDRayTracer
Material = raytracer_module.Material
LEDSource = raytracer_module.LEDSource
CUDARayTracer = raytracer_module.CUDARayTracer
import numpy as np


class AbsorberMaterial(Material):
    """
    Absorber material with optical properties.
    Light can enter, bounce within, and exit the absorber.
    We track EXIT points as the efficiency measurement.
    """
    def __init__(self, refractive_index=1.5, absorption_coeff=0.1):
        super().__init__()
        self.is_absorber = True
        self.refractive_index = refractive_index  # Similar to glass/acrylic
        self.absorption_coeff = absorption_coeff  # Light absorption inside


class UVLEDRayTracerWithAbsorber(UVLEDRayTracer):
    """
    Extended ray tracer with absorber material support.
    
    The absorber is treated as a material with optical properties:
    - Light can ENTER the absorber (refraction at surface)
    - Light can BOUNCE within the absorber (internal reflections)
    - Light can EXIT the absorber (refraction out)
    
    We track EXITS from absorber as the efficiency measurement.
    
    RECOMMENDED: Use separate STL files for lens and absorber (absorber_stl parameter).
    This ensures exact face identification without bounding box guessing.
    """
    
    def __init__(self, step_file_path: str = None, absorber_stl: str = None,
                 absorber_face_indices=None, absorber_bbox=None, lens_bbox=None,
                 absorber_refractive_index=1.5, absorber_absorption=0.1):
        """
        Initialize ray tracer with absorber support
        
        Args:
            step_file_path: Path to lens STL/STEP file (optional, can load later)
            absorber_stl: Path to SEPARATE absorber STL file (RECOMMENDED - exact face identification)
            absorber_face_indices: Set of face indices that belong to absorber material (DEPRECATED)
            absorber_bbox: Dict with xmin,xmax,ymin,ymax,zmin,zmax for position-based detection (DEPRECATED)
            lens_bbox: Dict with xmin,xmax,ymin,ymax,zmin,zmax to EXCLUDE from absorber detection (DEPRECATED)
            absorber_refractive_index: Refractive index of absorber material (default 1.5)
            absorber_absorption: Absorption coefficient inside absorber (default 0.1)
        """
        # Initialize parent without loading step file yet
        self.mesh = None
        self.material = Material()
        self.led_source = None
        self.ray_tracer = None
        
        # NEW: Separate absorber STL file (most reliable method)
        self.absorber_stl = absorber_stl
        self.lens_face_count = 0  # Number of lens faces (absorber faces come after)
        
        # OPTIMIZATION: Cache absorber mesh to avoid reloading from disk on every lens change
        self._cached_absorber_mesh = None
        self._cached_absorber_stl_path = None
        
        # DEPRECATED: Position-based absorber detection (unreliable for curved shapes)
        self.absorber_bbox = absorber_bbox
        self.lens_bbox = lens_bbox  # Faces in this bbox are NOT absorber
        self.absorber_face_indices = set()  # Will be populated in load_step_model
        self.absorber_material = AbsorberMaterial(
            refractive_index=absorber_refractive_index,
            absorption_coeff=absorber_absorption
        )
        
        # Log initialization method
        if absorber_stl:
            print(f"Absorber material: using SEPARATE STL file (most reliable)")
            print(f"  Absorber STL: {absorber_stl}")
            print(f"  Refractive index: {absorber_refractive_index}")
            print(f"  Absorption coeff: {absorber_absorption}")
        elif absorber_bbox:
            print(f"Absorber material: using bounding box detection (DEPRECATED)")
            print(f"  Refractive index: {absorber_refractive_index}")
            print(f"  Absorption coeff: {absorber_absorption}")
            if lens_bbox:
                print(f"  Lens faces will be excluded from absorber")
        elif absorber_face_indices:
            self.absorber_face_indices = absorber_face_indices if isinstance(absorber_face_indices, set) else set(absorber_face_indices)
            print(f"Absorber material (face indices, DEPRECATED): {len(self.absorber_face_indices)} faces")
        
        if step_file_path:
            self.load_step_model(step_file_path)
    
    def load_step_model(self, file_path: str):
        """Load geometry from STEP or STL file.
        
        If absorber_stl was provided in __init__, this will:
        1. Load the lens mesh from file_path
        2. Load the absorber mesh from absorber_stl
        3. Merge them into a single mesh
        4. Track face index ranges: faces 0 to N-1 are lens, faces N to M-1 are absorber
        
        This approach is more reliable than bounding box detection for any shape.
        """
        import trimesh
        import os
        
        # Normalize the file path (handle Windows backslashes and relative paths)
        original_path = file_path
        file_path = os.path.abspath(os.path.normpath(file_path))
        
        # Support both STEP and STL files
        file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        file_type = "STL" if file_ext == 'stl' else "STEP"
        print(f"Loading {file_type} model from {file_path}...")
        
        # Verify file exists with helpful error message
        if not os.path.exists(file_path):
            # Try original path in case normalization broke it
            if os.path.exists(original_path):
                file_path = original_path
                file_path = os.path.abspath(os.path.normpath(file_path))
            else:
                error_msg = f"File not found: {file_path}"
                if original_path != file_path:
                    error_msg += f" (original: {original_path})"
                raise FileNotFoundError(error_msg)
        
        file_size = os.path.getsize(file_path)
        print(f"  File exists: ✓")
        print(f"  File size: {file_size:,} bytes")
        
        try:
            # Load lens mesh
            if file_ext == 'stl':
                lens_mesh = trimesh.load(file_path, force='mesh')
                self._mesh_scale = 1.0  # No scaling for STL
                print(f"Loaded lens STL mesh (assuming mm units, no scaling)")
            else:
                with open(file_path, 'rb') as f:
                    lens_mesh = trimesh.load(f, file_type=file_ext, force='mesh')
                lens_mesh.apply_scale(1000.0)  # m → mm
                self._mesh_scale = 1000.0
                print(f"Loaded lens STEP mesh (scaled 1000x: m → mm)")
            
            self.lens_face_count = len(lens_mesh.faces)
            print(f"  Lens: {len(lens_mesh.vertices)} vertices, {self.lens_face_count} faces")
            
            # NEW: Load separate absorber STL if provided (most reliable method)
            if self.absorber_stl and os.path.exists(self.absorber_stl):
                absorber_path = os.path.abspath(os.path.normpath(self.absorber_stl))
                
                # OPTIMIZATION: Use cached absorber mesh if available and path matches
                if self._cached_absorber_mesh is not None and self._cached_absorber_stl_path == absorber_path:
                    # Reuse cached absorber mesh (fast!)
                    absorber_mesh = self._cached_absorber_mesh
                    absorber_face_count = len(absorber_mesh.faces)
                    print(f"  Absorber: CACHED ({absorber_face_count} faces)")
                else:
                    # Load from disk and cache for future use
                    print(f"Loading SEPARATE absorber STL: {absorber_path}")
                    absorber_mesh = trimesh.load(absorber_path, force='mesh')
                    absorber_face_count = len(absorber_mesh.faces)
                    print(f"  Absorber: {len(absorber_mesh.vertices)} vertices, {absorber_face_count} faces")
                    
                    # Cache the absorber mesh for future lens loads
                    self._cached_absorber_mesh = absorber_mesh
                    self._cached_absorber_stl_path = absorber_path
                
                # Merge meshes: lens first, then absorber
                # Face indices: 0 to lens_face_count-1 = lens
                #               lens_face_count to total-1 = absorber
                self.mesh = trimesh.util.concatenate([lens_mesh, absorber_mesh])
                
                # Set absorber face indices based on merged mesh
                # All faces starting at lens_face_count are absorber faces
                self.absorber_face_indices = set(range(self.lens_face_count, len(self.mesh.faces)))
                
                print(f"  MERGED MESH: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
                print(f"  Face index ranges:")
                print(f"    Lens faces: 0 to {self.lens_face_count - 1}")
                print(f"    Absorber faces: {self.lens_face_count} to {len(self.mesh.faces) - 1}")
                print(f"  ✓ Absorber identification: {len(self.absorber_face_indices)} faces (exact, no guessing)")
            else:
                # No separate absorber file - use lens mesh only
                self.mesh = lens_mesh
                print(f"Loaded mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
                
                # Fall back to bounding box detection if provided (DEPRECATED)
                if self.absorber_bbox:
                    self._identify_absorber_faces_by_bbox()
                elif self.absorber_face_indices:
                    # Legacy: Verify absorber face indices if using old method
                    max_face_idx = len(self.mesh.faces) - 1
                    valid_indices = [idx for idx in self.absorber_face_indices if 0 <= idx <= max_face_idx]
                    invalid_indices = [idx for idx in self.absorber_face_indices if idx < 0 or idx > max_face_idx]
                    if invalid_indices:
                        print(f"Warning: {len(invalid_indices)} invalid absorber face indices (mesh has {len(self.mesh.faces)} faces)")
                    self.absorber_face_indices = set(valid_indices)
            
            # Create ray tracer with merged mesh
            self.ray_tracer = CUDARayTracer(
                self.mesh.vertices,
                self.mesh.faces,
                self.material
            )
            
            if self.absorber_face_indices:
                print(f"✓ Absorber faces ready: {len(self.absorber_face_indices)} faces")
                
        except Exception as e:
            print(f"Error loading {file_type} file: {e}")
            raise
    
    def _identify_absorber_faces_by_bbox(self):
        """DEPRECATED: Identify absorber faces using bounding box (unreliable for curved shapes)"""
        absorber_bbox = self.absorber_bbox
        scale = getattr(self, '_mesh_scale', 1.0)
        absorber_min = np.array([absorber_bbox['xmin'] * scale, absorber_bbox['ymin'] * scale, absorber_bbox['zmin'] * scale])
        absorber_max = np.array([absorber_bbox['xmax'] * scale, absorber_bbox['ymax'] * scale, absorber_bbox['zmax'] * scale])
        print(f"WARNING: Using DEPRECATED bounding box detection (unreliable for curved shapes)")
        print(f"Absorber bbox (scaled {scale}x): [{absorber_min[0]:.1f}, {absorber_max[0]:.1f}] x [{absorber_min[1]:.1f}, {absorber_max[1]:.1f}] x [{absorber_min[2]:.1f}, {absorber_max[2]:.1f}]")
        
        # Also get lens bounding box if available (faces in lens bbox are NOT absorber)
        lens_min = None
        lens_max = None
        if self.lens_bbox:
            lens_min = np.array([self.lens_bbox['xmin'] * scale, self.lens_bbox['ymin'] * scale, self.lens_bbox['zmin'] * scale])
            lens_max = np.array([self.lens_bbox['xmax'] * scale, self.lens_bbox['ymax'] * scale, self.lens_bbox['zmax'] * scale])
            print(f"Lens bbox (excluded): [{lens_min[0]:.1f}, {lens_max[0]:.1f}] x [{lens_min[1]:.1f}, {lens_max[1]:.1f}] x [{lens_min[2]:.1f}, {lens_max[2]:.1f}]")
        
        # Find faces whose centroids are within the absorber bbox but NOT in lens bbox
        self.absorber_face_indices = set()
        tolerance = 0.5 * scale  # 0.5mm tolerance (scaled appropriately)
        lens_tolerance = 1.0 * scale  # Generous lens exclusion
        
        for face_idx, face in enumerate(self.mesh.faces):
            v0, v1, v2 = self.mesh.vertices[face]
            centroid = (v0 + v1 + v2) / 3.0
            
            in_absorber_bbox = (
                absorber_min[0] - tolerance <= centroid[0] <= absorber_max[0] + tolerance and
                absorber_min[1] - tolerance <= centroid[1] <= absorber_max[1] + tolerance and
                absorber_min[2] - tolerance <= centroid[2] <= absorber_max[2] + tolerance
            )
            
            in_lens_bbox = False
            if lens_min is not None and lens_max is not None:
                in_lens_bbox = (
                    lens_min[0] - lens_tolerance <= centroid[0] <= lens_max[0] + lens_tolerance and
                    lens_min[1] - lens_tolerance <= centroid[1] <= lens_max[1] + lens_tolerance and
                    lens_min[2] - lens_tolerance <= centroid[2] <= lens_max[2] + lens_tolerance
                )
            
            if in_absorber_bbox and not in_lens_bbox:
                self.absorber_face_indices.add(face_idx)
        
        print(f"Absorber faces (from bounding box, lens excluded): {len(self.absorber_face_indices)} of {len(self.mesh.faces)} total faces")
    
    def simulate(self, num_rays=10000, max_bounces=1000, max_ray_length=200.0):
        """
        Run ray tracing simulation with absorber as a material.
        
        The absorber is treated like any optical material:
        - Rays can ENTER the absorber (refraction)
        - Rays can bounce INSIDE the absorber (internal reflection)
        - Rays can EXIT the absorber (refraction out) - THIS IS WHAT WE TRACK
        
        FIXED: Segments are now recorded IMMEDIATELY upon hit detection,
        ensuring all ray paths are continuous and connected.
        
        Returns:
            ray_paths: List of ray segments with metadata
            intensities: Final intensities
        """
        if self.ray_tracer is None:
            raise ValueError("Geometry not loaded.")
        
        print(f"Simulating {num_rays} rays with up to {max_bounces} bounces...")
        if self.absorber_face_indices:
            print(f"  Absorber faces: {len(self.absorber_face_indices)} (rays can enter/exit)")
            print(f"  Absorber refractive index: {self.absorber_material.refractive_index}")
        print(f"  Max ray length: {max_ray_length} mm")
        
        # Small offset to prevent self-intersection when ray continues from hit point
        ORIGIN_OFFSET = 0.001  # 0.001mm = 1 micron
        
        ray_origins, ray_directions = self.generate_led_rays(num_rays)
        ray_paths = []
        intensities = np.ones(num_rays) * self.led_source.power
        
        # Track material state for each ray:
        # 0 = in air, 1 = in lens material, 2 = in absorber material
        ray_material_state = np.zeros(num_rays, dtype=np.int32)
        
        # Statistics
        absorber_entry_count = 0
        absorber_exit_count = 0
        lens_exit_count = 0
        
        # Path length and absorption tracking
        total_lens_path_length = 0.0
        total_absorber_path_length = 0.0
        total_air_path_length = 0.0
        initial_total_power = np.sum(intensities)
        initial_num_rays = num_rays  # Save original count
        fresnel_loss_total = 0.0
        
        for bounce in range(max_bounces):
            # Print progress every 10 bounces to reduce console spam
            if bounce % 10 == 0 or bounce < 5:
                print(f"  Bounce {bounce + 1}/{max_bounces}... ({num_rays} active rays)")
            
            if num_rays == 0:
                print(f"  All rays terminated at bounce {bounce + 1}")
                break
            
            hit_distances, hit_triangle_ids, hit_points = self.ray_tracer.trace(
                ray_origins, ray_directions
            )
            
            # Apply material absorption for distance traveled (fully vectorized for performance)
            valid_hits = hit_triangle_ids >= 0
            if np.any(valid_hits):
                # Track path lengths by material (vectorized)
                air_mask = valid_hits & (ray_material_state == 0)
                lens_mask = valid_hits & (ray_material_state == 1)
                absorber_mask = valid_hits & (ray_material_state == 2)
                
                total_air_path_length += float(np.sum(hit_distances[air_mask]))
                total_lens_path_length += float(np.sum(hit_distances[lens_mask]))
                total_absorber_path_length += float(np.sum(hit_distances[absorber_mask]))
                
                # Apply absorption (fully vectorized - much faster than loop)
                # Lens absorption
                if np.any(lens_mask):
                    intensities[lens_mask] *= np.exp(-self.material.absorption_coeff * hit_distances[lens_mask])
                # Absorber absorption
                if np.any(absorber_mask):
                    intensities[absorber_mask] *= np.exp(-self.absorber_material.absorption_coeff * hit_distances[absorber_mask])
            
            new_origins = []
            new_directions = []
            new_intensities = []
            new_material_states = []

            for i in range(num_rays):
                current_state = ray_material_state[i]
                
                # === HANDLE RAYS THAT DIDN'T HIT ANYTHING ===
                # Record segment for ALL rays that miss (not just air) - FIX for disconnected rays
                if hit_triangle_ids[i] < 0:
                    direction_normalized = ray_directions[i] / np.linalg.norm(ray_directions[i])
                    extended_end = ray_origins[i] + direction_normalized * max_ray_length
                    
                    ray_paths.append({
                        'start': ray_origins[i].tolist(),
                        'end': extended_end.tolist(),
                        'intensity': float(intensities[i]),
                        'bounce': bounce,
                        'is_exit': True,  # Ray exits to infinity
                        'is_absorber_hit': False,
                        'is_absorber_exit': False,
                        'is_absorber_entry': False
                    })
                    continue  # Ray is terminated
                
                # === RAY HIT SOMETHING ===
                tri_idx = hit_triangle_ids[i]
                is_absorber_face = tri_idx in self.absorber_face_indices
                
                # Get face normal
                face = self.ray_tracer.faces[tri_idx]
                v0, v1, v2 = self.ray_tracer.vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal_len = np.linalg.norm(normal)
                
                # Handle degenerate face - STILL RECORD SEGMENT (FIX for disconnected rays)
                if normal_len == 0:
                    ray_paths.append({
                        'start': ray_origins[i].tolist(),
                        'end': hit_points[i].tolist(),
                        'intensity': float(intensities[i]),
                        'bounce': bounce,
                        'is_exit': False,
                        'is_absorber_hit': is_absorber_face,
                        'is_absorber_exit': False,
                        'is_absorber_entry': False
                    })
                    continue  # Can't continue ray (no valid normal), but segment is recorded
                
                normal = normal / normal_len
                
                # Determine refractive indices based on what we're hitting
                if is_absorber_face:
                    # Hitting absorber surface
                    if current_state == 2:
                        # EXITING absorber (inside absorber -> outside)
                        n1 = self.absorber_material.refractive_index
                        n2 = 1.0  # Exit to air
                        new_state_if_refracted = 0  # Will be in air
                        is_absorber_exit = True
                        is_absorber_entry = False
                    else:
                        # ENTERING absorber (from air or lens)
                        n1 = self.material.refractive_index if current_state == 1 else 1.0
                        n2 = self.absorber_material.refractive_index
                        new_state_if_refracted = 2  # Will be in absorber
                        is_absorber_exit = False
                        is_absorber_entry = True
                else:
                    # Hitting lens surface (normal material interaction)
                    if current_state == 1:
                        # Exiting lens
                        n1 = self.material.refractive_index
                        n2 = 1.0
                        new_state_if_refracted = 0
                    else:
                        # Entering lens
                        n1 = 1.0
                        n2 = self.material.refractive_index
                        new_state_if_refracted = 1
                    is_absorber_exit = False
                    is_absorber_entry = False
                
                # Calculate refraction/reflection
                reflected_dir, refracted_dir, reflectance = self._calculate_refraction(
                    ray_directions[i], normal, n1, n2
                )
                
                # Determine segment metadata
                is_lens_exit = (current_state == 1 and new_state_if_refracted == 0 and not is_absorber_face)
                
                # === RECORD SEGMENT IMMEDIATELY (FIX: before ray continuation decision) ===
                ray_paths.append({
                    'start': ray_origins[i].tolist(),
                    'end': hit_points[i].tolist(),
                    'intensity': float(intensities[i]),
                    'bounce': bounce,
                    'is_exit': is_lens_exit,
                    'is_lens_exit': is_lens_exit,  # Explicit lens exit flag
                    'is_absorber_hit': False,
                    'is_absorber_exit': is_absorber_exit,
                    'is_absorber_entry': is_absorber_entry
                })
                
                # Track statistics
                if is_absorber_entry:
                    absorber_entry_count += 1
                if is_absorber_exit:
                    absorber_exit_count += 1
                    continue  # Ray stops at absorber exit - don't continue
                if is_lens_exit:
                    lens_exit_count += 1
                
                # === NOW decide ray continuation ===
                if np.random.random() < reflectance or refracted_dir is None:
                    # Reflection (or total internal reflection)
                    # Track Fresnel loss: ray loses (1-R) of its energy
                    fresnel_loss_total += float(intensities[i] * (1 - reflectance))
                    if reflected_dir is not None:
                        reflected_dir = np.asarray(reflected_dir, dtype=np.float32).flatten()
                        if reflected_dir.shape[0] == 3:
                            # Apply small offset along new direction to prevent self-intersection
                            new_origin = np.asarray(hit_points[i], dtype=np.float32).flatten()
                            new_origin = new_origin + reflected_dir * ORIGIN_OFFSET
                            
                            new_origins.append(new_origin)
                            new_directions.append(reflected_dir)
                            new_intensities.append(float(intensities[i] * reflectance))
                            new_material_states.append(int(current_state))  # Stay in same material
                else:
                    # Refraction - ray continues into new medium
                    # Track Fresnel loss: ray loses R of its energy
                    fresnel_loss_total += float(intensities[i] * reflectance)
                    refracted_dir = np.asarray(refracted_dir, dtype=np.float32).flatten()
                    if refracted_dir.shape[0] == 3:
                        # Apply small offset along new direction to prevent self-intersection
                        new_origin = np.asarray(hit_points[i], dtype=np.float32).flatten()
                        new_origin = new_origin + refracted_dir * ORIGIN_OFFSET
                        
                        new_origins.append(new_origin)
                        new_directions.append(refracted_dir)
                        new_intensities.append(float(intensities[i] * (1 - reflectance)))
                        new_material_states.append(new_state_if_refracted)

            if len(new_origins) == 0:
                break

            # Validate directions
            valid_indices = []
            for idx in range(len(new_directions)):
                dir_vec = new_directions[idx]
                if isinstance(dir_vec, np.ndarray) and dir_vec.shape == (3,):
                    valid_indices.append(idx)
                elif isinstance(dir_vec, (list, tuple)) and len(dir_vec) == 3:
                    new_directions[idx] = np.array(dir_vec, dtype=np.float32)
                    valid_indices.append(idx)
            
            if len(valid_indices) == 0:
                break
            
            if len(valid_indices) < len(new_origins):
                new_origins = [new_origins[i] for i in valid_indices]
                new_directions = [new_directions[i] for i in valid_indices]
                new_intensities = [new_intensities[i] for i in valid_indices]
                new_material_states = [new_material_states[i] for i in valid_indices]
            
            ray_origins = np.array(new_origins, dtype=np.float32)
            ray_directions = np.array(new_directions, dtype=np.float32)
            intensities = np.array(new_intensities)
            ray_material_state = np.array(new_material_states, dtype=np.int32)
            num_rays = len(ray_origins)
        
        # Statistics
        lens_exits = sum(1 for r in ray_paths if r.get('is_exit', False) and not r.get('is_absorber_exit', False))
        absorber_exits = sum(1 for r in ray_paths if r.get('is_absorber_exit', False))
        
        # Calculate final power on absorber exits and lens exits
        absorber_exit_rays = [r for r in ray_paths if r.get('is_absorber_exit', False)]
        final_absorber_power = sum(r.get('intensity', 0) for r in absorber_exit_rays)
        
        lens_exit_rays = [r for r in ray_paths if r.get('is_lens_exit', False)]
        final_lens_exit_power = sum(r.get('intensity', 0) for r in lens_exit_rays)
        
        # Calculate absorption losses (use initial ray count for averages)
        avg_lens_path = total_lens_path_length / initial_num_rays if initial_num_rays > 0 else 0
        avg_absorber_path = total_absorber_path_length / initial_num_rays if initial_num_rays > 0 else 0
        
        # Theoretical absorption loss based on path lengths (Beer-Lambert)
        lens_transmission = np.exp(-self.material.absorption_coeff * avg_lens_path) if avg_lens_path > 0 else 1.0
        absorber_transmission = np.exp(-self.absorber_material.absorption_coeff * avg_absorber_path) if avg_absorber_path > 0 else 1.0
        
        print(f"\nSimulation complete. Traced {len(ray_paths)} ray segments.")
        print(f"  Lens exits: {lens_exits}")
        print(f"  Absorber entries: {absorber_entry_count}")
        print(f"  Absorber exits (EFFICIENCY METRIC): {absorber_exits}")
        
        print(f"\n  === PATH LENGTH ANALYSIS ===")
        print(f"  Total lens path:     {total_lens_path_length:.1f} mm (avg {avg_lens_path:.2f} mm/ray)")
        print(f"  Total absorber path: {total_absorber_path_length:.1f} mm (avg {avg_absorber_path:.2f} mm/ray)")
        print(f"  Total air path:      {total_air_path_length:.1f} mm")
        
        print(f"\n  === ABSORPTION ANALYSIS ===")
        print(f"  Lens absorption coeff:     {self.material.absorption_coeff:.4f} /mm")
        print(f"  Absorber absorption coeff: {self.absorber_material.absorption_coeff:.4f} /mm")
        print(f"  Avg lens transmission:     {lens_transmission*100:.1f}% (per ray avg path)")
        print(f"  Avg absorber transmission: {absorber_transmission*100:.1f}% (per ray avg path)")
        
        print(f"\n  === POWER BUDGET ===")
        print(f"  Initial total power:  {initial_total_power:.2f} mW")
        print(f"  Fresnel losses:       {fresnel_loss_total:.2f} mW ({fresnel_loss_total/initial_total_power*100:.1f}%)")
        print(f"  Lens exit power:      {final_lens_exit_power:.2f} mW ({final_lens_exit_power/initial_total_power*100:.1f}%)")
        print(f"  Final absorber power: {final_absorber_power:.2f} mW ({final_absorber_power/initial_total_power*100:.1f}%)")
        
        # Store diagnostics for external access
        self.diagnostics = {
            'total_lens_path_mm': total_lens_path_length,
            'total_absorber_path_mm': total_absorber_path_length,
            'total_air_path_mm': total_air_path_length,
            'avg_lens_path_mm': avg_lens_path,
            'avg_absorber_path_mm': avg_absorber_path,
            'lens_absorption_coeff': self.material.absorption_coeff,
            'absorber_absorption_coeff': self.absorber_material.absorption_coeff,
            'lens_transmission_pct': lens_transmission * 100,
            'absorber_transmission_pct': absorber_transmission * 100,
            'initial_total_power_mW': initial_total_power,
            'fresnel_loss_mW': fresnel_loss_total,
            'fresnel_loss_pct': fresnel_loss_total / initial_total_power * 100 if initial_total_power > 0 else 0,
            'lens_exit_power_mW': final_lens_exit_power,
            'lens_exit_power_pct': final_lens_exit_power / initial_total_power * 100 if initial_total_power > 0 else 0,
            'lens_exit_count': len(lens_exit_rays),
            'final_absorber_power_mW': final_absorber_power,
            'final_absorber_power_pct': final_absorber_power / initial_total_power * 100 if initial_total_power > 0 else 0,
            'initial_num_rays': initial_num_rays,
            'absorber_exit_count': len(absorber_exit_rays),
        }
        
        return ray_paths, intensities


# Make it available for import
# Store reference for cleanup
raytracer_module = _original_raytracer_module

if __name__ != "__main__":
    pass

