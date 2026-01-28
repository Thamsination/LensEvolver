"""
Raytracer Wrapper Script
========================
This script wraps raytracer-v2.2.0.py to output JSON results for FreeCAD integration.

Usage:
    python raytracer_wrapper.py <step_file> [options] --output-json <output.json>

All options from raytracer-v2.2.0.py are supported, plus --output-json for JSON export.
"""

import sys
import os

# Import standard library modules first
import json
import numpy as np

# NOTE: We don't filter sys.path here - it can break standard library imports
# Instead, raytracer-v2.2.0.py will detect and reject FreeCAD's incomplete PyCUDA
# by checking the module path and trying to initialize CUDA

# Set UTF-8 encoding for stdout/stderr to handle Unicode characters
if sys.platform == 'win32':
    import io
    # Reconfigure stdout/stderr to use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    else:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the directory containing raytracer-v2.2.0.py to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import will be done in main() after loading the module

def cleanup_cuda_context(raytracer_module):
    """Clean up CUDA context properly"""
    try:
        # Try to import pycuda directly
        try:
            import pycuda.driver as cuda
            # Pop all contexts from stack until empty
            max_pops = 10  # Safety limit
            pop_count = 0
            while pop_count < max_pops:
                try:
                    cuda.Context.pop()
                    pop_count += 1
                except (RuntimeError, AttributeError, cuda.Error):
                    # Stack is empty or no context
                    break
            if pop_count > 0:
                print(f"Cleaned up {pop_count} CUDA context(s)", file=sys.stderr)
        except ImportError:
            # PyCUDA not available, nothing to clean
            pass
        except Exception as e:
            # Ignore cleanup errors silently
            pass
        
        # Also try to pop the module's context if it exists
        if hasattr(raytracer_module, 'ctx'):
            try:
                raytracer_module.ctx.pop()
            except:
                pass
        
        # Try to access CUDA_AVAILABLE and ctx from the original module if it exists
        if hasattr(raytracer_module, 'CUDA_AVAILABLE') and raytracer_module.CUDA_AVAILABLE:
            if hasattr(raytracer_module, 'ctx'):
                try:
                    raytracer_module.ctx.pop()
                except:
                    pass
    except Exception:
        # Ignore all cleanup errors
        pass


def main():
    """Main wrapper function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Raytracer wrapper for FreeCAD integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input geometry (either step_file OR --lens-stl required)
    parser.add_argument('step_file', nargs='?', default=None, help='Path to STEP model file (or use --lens-stl)')
    parser.add_argument('--lens-stl', type=str, default=None,
                       help='Path to lens STL file (alternative to step_file)')
    parser.add_argument('--output-json', required=True, help='Output JSON file for ray paths')
    
    # Simulation parameters
    parser.add_argument('--rays', type=int, default=10000, help='Number of rays')
    parser.add_argument('--bounces', type=int, default=1000, help='Max bounces')
    parser.add_argument('--max-ray-length', type=float, default=200.0, 
                       help='Maximum ray length in mm when no obstacle is hit (default: 200)')
    
    # LED parameters (defaults for Nichia NVSU119CT-U405)
    parser.add_argument('--led-pos', nargs=3, type=float, default=[0, 0, 2],
                       metavar=('X', 'Y', 'Z'), help='LED position in mm')
    parser.add_argument('--led-dir', nargs=3, type=float, default=[0, 0, 1],
                       metavar=('X', 'Y', 'Z'), help='LED direction vector')
    parser.add_argument('--led-power', type=float, default=1420.0, 
                       help='LED radiant power in mW (default: 1420 for NVSU119CT-U405)')
    parser.add_argument('--led-angle', type=float, default=70.0, 
                       help='LED half-angle in degrees (default: 70 for 140° viewing angle)')
    parser.add_argument('--wavelength', type=float, default=405, 
                       help='LED peak wavelength in nm (default: 405 for NVSU119CT-U405)')
    parser.add_argument('--emitter-size', type=float, default=1.0,
                       help='LED emitter size in mm (default: 1.0, use 0 for point source)')
    parser.add_argument('--lambertian', action='store_true', default=True,
                       help='Use Lambertian (cosine-weighted) emission (fallback if not using datasheet)')
    parser.add_argument('--no-lambertian', action='store_false', dest='lambertian',
                       help='Use uniform emission within cone (disable Lambertian)')
    parser.add_argument('--datasheet-directivity', action='store_true', default=True,
                       help='Use actual NVSU119CT directivity pattern from datasheet (default: True)')
    parser.add_argument('--no-datasheet-directivity', action='store_false', dest='datasheet_directivity',
                       help='Disable datasheet directivity (use Lambertian or uniform instead)')
    parser.add_argument('--led-model', type=str, default='U405',
                       choices=['U375', 'U385', 'U395', 'U405'],
                       help='LED model for directivity pattern (U375/U385/U395/U405, default: U405)')
    
    # Lens material parameters
    parser.add_argument('--refractive-index', type=float, default=1.535, 
                       help='Lens refractive index (default: 1.535 for ZEONEX K26R)')
    parser.add_argument('--absorption', type=float, default=0.001, 
                       help='Lens absorption coefficient /mm (default: 0.001 for COP)')
    
    # Absorber material parameters
    parser.add_argument('--absorber-refractive-index', type=float, default=1.585,
                       help='Absorber refractive index (default: 1.585 for TRIREX 3020MD PC)')
    parser.add_argument('--absorber-absorption', type=float, default=0.05,
                       help='Absorber absorption coefficient /mm (default: 0.05 for PC)')
    
    # Absorber geometry parameters
    # NEW: Separate absorber STL file (most reliable method - exact face identification)
    parser.add_argument('--absorber-stl', type=str, default=None,
                       help='Path to separate absorber STL file (most reliable method)')
    
    # DEPRECATED: Bounding box detection (unreliable for curved shapes)
    parser.add_argument('--absorber-faces', nargs='+', type=int, default=None,
                       help='Face indices that belong to absorber (DEPRECATED)')
    parser.add_argument('--absorber-bbox', nargs=6, type=float, default=None,
                       metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                       help='Absorber bounding box in mm (DEPRECATED: use --absorber-stl)')
    parser.add_argument('--lens-bbox', nargs=6, type=float, default=None,
                       metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                       help='Lens bounding box in mm (DEPRECATED: use --absorber-stl)')
    
    # Other options (passed through but not used)
    parser.add_argument('--no-display', action='store_true', help='Do not display plots')
    parser.add_argument('--skip-3d-viz', action='store_true', help='Skip 3D visualization')
    
    args = parser.parse_args()
    
    # Validate input: require either step_file or --lens-stl
    if not args.step_file and not args.lens_stl:
        print("ERROR: Must provide either step_file or --lens-stl", file=sys.stderr)
        sys.exit(1)
    
    # Import raytracer module (loaded at bottom of file)
    raytracer_module = sys.modules.get('raytracer_v2_2_0')
    if raytracer_module is None:
        print("ERROR: Raytracer module not loaded", file=sys.stderr)
        sys.exit(1)
    
    # Try to use UVLEDRayTracerWithAbsorber if available, otherwise fall back to original
    if hasattr(raytracer_module, 'UVLEDRayTracerWithAbsorber'):
        UVLEDRayTracer = raytracer_module.UVLEDRayTracerWithAbsorber
        print("Using UVLEDRayTracerWithAbsorber (absorber material support)", file=sys.stderr)
    else:
        UVLEDRayTracer = raytracer_module.UVLEDRayTracer
        print("Using UVLEDRayTracer (original, no absorber support)", file=sys.stderr)
    
    Material = raytracer_module.Material
    
    # Create ray tracer with absorber support if available
    # Priority: absorber_stl (separate file, most reliable) > absorber_bbox (deprecated) > absorber_faces (deprecated)
    absorber_stl = args.absorber_stl
    
    # Legacy: absorber bounding box (deprecated, unreliable for curved shapes)
    absorber_bbox = None
    if args.absorber_bbox and not absorber_stl:
        absorber_bbox = {
            'xmin': args.absorber_bbox[0],
            'xmax': args.absorber_bbox[1],
            'ymin': args.absorber_bbox[2],
            'ymax': args.absorber_bbox[3],
            'zmin': args.absorber_bbox[4],
            'zmax': args.absorber_bbox[5]
        }
        print(f"WARNING: Using deprecated absorber_bbox method (unreliable for curved shapes)", file=sys.stderr)
        print(f"Absorber bounding box: X[{absorber_bbox['xmin']:.2f}, {absorber_bbox['xmax']:.2f}], "
              f"Y[{absorber_bbox['ymin']:.2f}, {absorber_bbox['ymax']:.2f}], "
              f"Z[{absorber_bbox['zmin']:.2f}, {absorber_bbox['zmax']:.2f}]", file=sys.stderr)
    
    # Legacy: lens bounding box (deprecated)
    lens_bbox = None
    if args.lens_bbox and not absorber_stl:
        lens_bbox = {
            'xmin': args.lens_bbox[0],
            'xmax': args.lens_bbox[1],
            'ymin': args.lens_bbox[2],
            'ymax': args.lens_bbox[3],
            'zmin': args.lens_bbox[4],
            'zmax': args.lens_bbox[5]
        }
        print(f"Lens bounding box (excluded from absorber): X[{lens_bbox['xmin']:.2f}, {lens_bbox['xmax']:.2f}], "
              f"Y[{lens_bbox['ymin']:.2f}, {lens_bbox['ymax']:.2f}], "
              f"Z[{lens_bbox['zmin']:.2f}, {lens_bbox['zmax']:.2f}]", file=sys.stderr)
    
    # Use absorber-enabled tracer if available
    if hasattr(UVLEDRayTracer, '__init__'):
        # Check if it accepts absorber_stl parameter (new, preferred method)
        import inspect
        sig = inspect.signature(UVLEDRayTracer.__init__)
        if 'absorber_stl' in sig.parameters:
            # New method: separate absorber STL file with material properties
            tracer = UVLEDRayTracer(
                step_file_path=None, 
                absorber_stl=absorber_stl,
                absorber_refractive_index=args.absorber_refractive_index,
                absorber_absorption=args.absorber_absorption
            )
            if absorber_stl:
                print(f"Using separate absorber STL file: {absorber_stl}", file=sys.stderr)
                print(f"Absorber material: n={args.absorber_refractive_index}, α={args.absorber_absorption}/mm", file=sys.stderr)
        elif 'absorber_bbox' in sig.parameters:
            # Legacy: bounding box detection
            tracer = UVLEDRayTracer(step_file_path=None, absorber_bbox=absorber_bbox, lens_bbox=lens_bbox)
        elif 'absorber_face_indices' in sig.parameters:
            # Fallback to old method
            absorber_face_set = set(args.absorber_faces) if args.absorber_faces else None
            tracer = UVLEDRayTracer(step_file_path=None, absorber_face_indices=absorber_face_set)
        else:
            tracer = UVLEDRayTracer()
    else:
        tracer = UVLEDRayTracer()
    
    # Set lens material properties
    tracer.material.refractive_index = args.refractive_index
    tracer.material.absorption_coeff = args.absorption
    print(f"Lens material: n={args.refractive_index}, α={args.absorption}/mm", file=sys.stderr)
    
    # Check for required packages before loading geometry
    missing_packages = []
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import trimesh
    except ImportError:
        missing_packages.append("trimesh")
    
    try:
        import pycuda
    except ImportError:
        missing_packages.append("pycuda")
    
    if missing_packages:
        print(f"ERROR: Missing required Python packages: {', '.join(missing_packages)}", file=sys.stderr)
        print(f"\nTo install missing packages, run:", file=sys.stderr)
        print(f"  pip install {' '.join(missing_packages)}", file=sys.stderr)
        print(f"\nOr install all required packages:", file=sys.stderr)
        print(f"  pip install numpy pycuda trimesh matplotlib", file=sys.stderr)
        print(f"\nMake sure you're using the same Python that FreeCAD is calling.", file=sys.stderr)
        sys.exit(1)
    
    # Load geometry (either STEP file or STL files)
    try:
        if args.lens_stl:
            # Load from separate lens and absorber STL files
            print(f"Loading lens from STL: {args.lens_stl}", file=sys.stderr)
            tracer.load_step_model(args.lens_stl)  # load_step_model handles STL via trimesh
        elif args.step_file:
            # Load from STEP file
            print(f"Loading geometry from STEP: {args.step_file}", file=sys.stderr)
            tracer.load_step_model(args.step_file)
        else:
            print("ERROR: No geometry file provided", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load geometry: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    # Setup LED with realistic emission model
    tracer.setup_led_source(
        position=args.led_pos,
        direction=args.led_dir,
        power=args.led_power,
        wavelength=args.wavelength * 1e-9,  # Convert nm to meters
        half_angle=args.led_angle,
        emitter_size=args.emitter_size,
        lambertian=args.lambertian,
        use_datasheet_directivity=args.datasheet_directivity,
        model=args.led_model
    )
    
    # Run simulation with max ray length
    try:
        ray_paths, intensities = tracer.simulate(
            num_rays=args.rays, 
            max_bounces=args.bounces,
            max_ray_length=args.max_ray_length
        )
    except Exception as e:
        print(f"ERROR: Simulation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Post-process ray paths to:
    # 1. Limit ray length to max_ray_length when no obstacle is hit
    # 2. Mark hit points (refraction points)
    # 3. Track absorber hits (if not already marked by raytracer)
    max_ray_length = args.max_ray_length
    hit_points = []  # Store hit points for visualization
    
    processed_paths = []
    absorber_hits_count = 0
    
    for i, path in enumerate(ray_paths):
        start = np.array(path['start']) if isinstance(path['start'], (list, np.ndarray)) else path['start']
        end = np.array(path['end']) if isinstance(path['end'], (list, np.ndarray)) else path['end']
        
        # Check if this is an absorber hit or exit (from raytracer)
        is_absorber_hit = path.get('is_absorber_hit', False)
        is_absorber_exit = path.get('is_absorber_exit', False)
        is_absorber_entry = path.get('is_absorber_entry', False)
        if is_absorber_hit or is_absorber_exit:
            absorber_hits_count += 1
        
        # Calculate ray direction and length
        direction = end - start
        ray_length = np.linalg.norm(direction)
        
        # If ray is very long (likely didn't hit anything), limit to max_ray_length
        if ray_length > max_ray_length * 1.5:  # Allow some tolerance
            # Normalize direction and set end point at max_ray_length
            if ray_length > 0:
                direction_normalized = direction / ray_length
                end = start + direction_normalized * max_ray_length
                ray_length = max_ray_length
        
        # Mark this as a hit point if:
        # - It's a refraction/reflection point (bounce > 0 means it hit a surface)
        # - It's NOT an absorber hit (absorber hits are tracked separately)
        is_hit_point = path.get('bounce', 0) > 0 and not is_absorber_hit
        
        if is_hit_point:
            hit_points.append({
                'position': end.tolist() if isinstance(end, np.ndarray) else list(end),
                'bounce': int(path.get('bounce', 0)),
                'is_exit': bool(path.get('is_exit', False)) if not isinstance(path.get('is_exit', False), np.bool_) else bool(path.get('is_exit', False)),
                'intensity': float(path.get('intensity', 1.0))
            })
        
        processed_paths.append({
            'start': start.tolist() if isinstance(start, np.ndarray) else list(start),
            'end': end.tolist() if isinstance(end, np.ndarray) else list(end),
            'intensity': float(path['intensity']),
            'bounce': int(path.get('bounce', 0)),
            'is_exit': bool(path.get('is_exit', False)) if not isinstance(path.get('is_exit', False), np.bool_) else bool(path.get('is_exit', False)),
            'is_hit_point': bool(is_hit_point),
            'is_absorber_hit': bool(is_absorber_hit),
            'is_absorber_exit': bool(is_absorber_exit),  # Key flag for efficiency metric
            'is_absorber_entry': bool(is_absorber_entry)
        })
    
    # Convert ray_paths to JSON-serializable format
    # Helper function to convert numpy types to Python native types (NumPy 2.0 compatible)
    def convert_to_python_type(value):
        # Check for numpy integer types (base class and specific types)
        if isinstance(value, np.integer):
            return int(value)
        # Check for numpy float types (base class and specific types, np.float_ removed in NumPy 2.0)
        elif isinstance(value, np.floating):
            return float(value)
        # Check for numpy bool types
        elif isinstance(value, np.bool_):
            return bool(value)
        # Check for numpy arrays
        elif isinstance(value, np.ndarray):
            return value.tolist()
        # Recursively convert lists and tuples
        elif isinstance(value, (list, tuple)):
            return [convert_to_python_type(item) for item in value]
        # Recursively convert dictionaries
        elif isinstance(value, dict):
            return {k: convert_to_python_type(v) for k, v in value.items()}
        else:
            return value
    
    # Convert all paths and hit points to ensure JSON serializability
    processed_paths_clean = [convert_to_python_type(p) for p in processed_paths]
    hit_points_clean = [convert_to_python_type(h) for h in hit_points]
    
    absorber_hits_total = sum(1 for p in processed_paths if p.get('is_absorber_hit', False))
    absorber_exits_total = sum(1 for p in processed_paths if p.get('is_absorber_exit', False))
    absorber_entries_total = sum(1 for p in processed_paths if p.get('is_absorber_entry', False))
    
    json_data = {
        'ray_paths': processed_paths_clean,
        'hit_points': hit_points_clean,
        'statistics': {
            'total_segments': int(len(processed_paths)),
            'initial_rays': int(sum(1 for p in processed_paths if p.get('bounce', 0) == 0)),
            'exit_segments': int(sum(1 for p in processed_paths if p.get('is_exit', False))),
            'hit_points_count': int(len(hit_points)),
            'absorber_hits': int(absorber_hits_total),
            'absorber_exits': int(absorber_exits_total),  # Key metric: rays exiting absorber
            'absorber_entries': int(absorber_entries_total),
            'max_bounce': int(max((p.get('bounce', 0) for p in processed_paths), default=0)),
        }
    }
    
    # Write JSON file first (before cleanup, in case cleanup causes issues)
    try:
        with open(args.output_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"SUCCESS: Results written to {args.output_json}", file=sys.stderr)
        print(f"Total segments: {len(processed_paths)}", file=sys.stderr)
        print(f"Hit points: {len(hit_points)}", file=sys.stderr)
        if absorber_hits_total > 0:
            print(f"Absorber hits (rays stopped): {absorber_hits_total}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to write JSON: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    # Clean up CUDA context after writing JSON (but before exit)
    # This prevents the "context stack not empty" error
    try:
        # First, try to clean up the main raytracer module
        raytracer_module = sys.modules.get('raytracer_v2_2_0')
        if raytracer_module:
            cleanup_cuda_context(raytracer_module)
            # If this is raytracer_with_absorber, it has a reference to the original module
            if hasattr(raytracer_module, 'raytracer_module'):
                cleanup_cuda_context(raytracer_module.raytracer_module)
            # Also check for _original_raytracer_module
            if hasattr(raytracer_module, '_original_raytracer_module'):
                cleanup_cuda_context(raytracer_module._original_raytracer_module)
        
        # Also try to clean up from any other raytracer modules
        for module_name in list(sys.modules.keys()):
            if 'raytracer' in module_name.lower() and module_name != 'raytracer_v2_2_0':
                mod = sys.modules.get(module_name)
                if mod:
                    # Check if this module has a reference to the original raytracer module
                    if hasattr(mod, 'raytracer_module'):
                        cleanup_cuda_context(mod.raytracer_module)
                    if hasattr(mod, '_original_raytracer_module'):
                        cleanup_cuda_context(mod._original_raytracer_module)
    except Exception:
        # Ignore cleanup errors - we're about to exit anyway
        pass
    
    return 0


if __name__ == "__main__":
    # Load raytracer module with absorber support
    import importlib.util
    
    # Try to use the new raytracer with absorber support
    raytracer_with_absorber_path = os.path.join(script_dir, "raytracer_with_absorber.py")
    raytracer_original_path = os.path.join(script_dir, "raytracer-v2.2.0.py")
    
    # Prefer the new raytracer with absorber support if it exists
    if os.path.exists(raytracer_with_absorber_path):
        raytracer_path = raytracer_with_absorber_path
        print("Using raytracer_with_absorber.py (absorber material support)", file=sys.stderr)
    elif os.path.exists(raytracer_original_path):
        raytracer_path = raytracer_original_path
        print("Using raytracer-v2.2.0.py (original, no absorber support)", file=sys.stderr)
    else:
        print(f"ERROR: No raytracer found. Checked:", file=sys.stderr)
        print(f"  {raytracer_with_absorber_path}", file=sys.stderr)
        print(f"  {raytracer_original_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("raytracer_v2_2_0", raytracer_path)
    raytracer_module = importlib.util.module_from_spec(spec)
    
    # Execute the module (this will initialize CUDA, etc.)
    # Wrap stdout/stderr to handle Unicode encoding errors
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
                    # Replace problematic Unicode characters
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Make it available globally
    sys.modules['raytracer_v2_2_0'] = raytracer_module
    
    # Now run main
    try:
        exit_code = main()
        # Additional cleanup after main (main() should have cleaned up, but be safe)
        try:
            cleanup_cuda_context(raytracer_module)
            # Also check if raytracer_with_absorber loaded the original module
            if hasattr(raytracer_module, 'raytracer_module'):
                cleanup_cuda_context(raytracer_module.raytracer_module)
            if hasattr(raytracer_module, '_original_raytracer_module'):
                cleanup_cuda_context(raytracer_module._original_raytracer_module)
        except:
            pass
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        try:
            cleanup_cuda_context(raytracer_module)
        except:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        try:
            cleanup_cuda_context(raytracer_module)
        except:
            pass
        sys.exit(1)


