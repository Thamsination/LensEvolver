"""
Raytracer integration for the Lens Optimizer.

This module provides functions to find Python with PyCUDA and run
raytracing evaluations on lens/absorber geometry.
"""

import os
import subprocess
import shutil
from typing import Optional, Dict
import numpy as np
import FreeCAD

from .config import (
    MANUAL_PYTHON_PATH,
    LED_MODELS,
    DEFAULT_LED_HALF_ANGLE,
    DEFAULT_LED_EMITTER_SIZE,
    DEFAULT_LED_CURRENT,
    DEFAULT_LED_WAVELENGTH,
    current_to_radiant_power
)
from .materials import (
    LENS_REFRACTIVE_INDEX,
    LENS_ABSORPTION_COEFF,
    ABSORBER_REFRACTIVE_INDEX,
    ABSORBER_ABSORPTION_COEFF
)
from .server_management import get_raytracer_client
from .config import RAYTRACER_PATH

# Cache the python executable so we don't search every time
_cached_python_exe = None


def verify_python_has_packages(python_exe):
    """Verify that a Python executable has the required packages (including PyCUDA)."""
    try:
        check_cmd = [
            python_exe, "-c",
            "import numpy; import trimesh; import pycuda; import matplotlib"
        ]
        result = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        return result.returncode == 0
    except:
        return False


def find_python_executable() -> Optional[str]:
    """Find system Python executable (not FreeCAD's Python) that has PyCUDA."""
    FreeCAD.Console.PrintMessage("Searching for Python with PyCUDA...\n")
    
    # Check if manual path is set
    if MANUAL_PYTHON_PATH and os.path.exists(MANUAL_PYTHON_PATH):
        FreeCAD.Console.PrintMessage(f"Checking manual Python path: {MANUAL_PYTHON_PATH}\n")
        try:
            result = subprocess.run(
                [MANUAL_PYTHON_PATH, "--version"],
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if result.returncode == 0:
                if verify_python_has_packages(MANUAL_PYTHON_PATH):
                    FreeCAD.Console.PrintMessage(f"Using manual Python path: {MANUAL_PYTHON_PATH}\n")
                    return MANUAL_PYTHON_PATH
                else:
                    FreeCAD.Console.PrintWarning("Manual Python path doesn't have required packages (numpy, trimesh, pycuda, matplotlib)\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Manual Python path failed: {e}\n")
    
    # Try common full paths first
    common_paths = [
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Python", "Python314", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Python", "Python313", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Python", "Python312", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Python", "Python311", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python314", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python313", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python312", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python311", "python.exe"),
        r"C:\Python314\python.exe",
        r"C:\Python313\python.exe",
        r"C:\Python312\python.exe",
        r"C:\Python311\python.exe",
        r"C:\Program Files\Python314\python.exe",
        r"C:\Program Files\Python313\python.exe",
        r"C:\Program Files\Python312\python.exe",
        r"C:\Program Files\Python311\python.exe",
    ]
    
    found_pythons = []  # Track what we found for diagnostic messages
    
    for python_path in common_paths:
        if os.path.exists(python_path):
            found_pythons.append(python_path)
            try:
                result = subprocess.run(
                    [python_path, "--version"],
                    capture_output=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                if result.returncode == 0:
                    version = result.stdout.decode().strip() if result.stdout else "unknown"
                    FreeCAD.Console.PrintMessage(f"  Found: {python_path} ({version})\n")
                    if verify_python_has_packages(python_path):
                        FreeCAD.Console.PrintMessage(f"  -> Has required packages (numpy, trimesh, pycuda, matplotlib)\n")
                        return python_path
                    else:
                        FreeCAD.Console.PrintMessage(f"  -> Missing required packages\n")
            except Exception as e:
                FreeCAD.Console.PrintMessage(f"  Error checking {python_path}: {e}\n")
                continue
    
    # Try PATH as last resort
    FreeCAD.Console.PrintMessage("Checking PATH for Python...\n")
    for cmd in ["python", "python3"]:
        python_path = shutil.which(cmd)
        if python_path:
            # Convert to absolute path to avoid issues with relative paths
            python_path = os.path.abspath(python_path)
            # Skip if it's in the FreeCAD or LensOptimizerModules directory (not a system Python)
            if 'FreeCAD' in python_path or 'LensOptimizerModules' in python_path:
                FreeCAD.Console.PrintMessage(f"  Skipping {python_path} (embedded Python)\n")
                continue
            try:
                result = subprocess.run(
                    [python_path, "--version"],
                    capture_output=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                version = result.stdout.decode().strip() if result.stdout else "unknown"
                FreeCAD.Console.PrintMessage(f"  Found in PATH: {python_path} ({version})\n")
            except:
                pass
            if verify_python_has_packages(python_path):
                FreeCAD.Console.PrintMessage(f"  -> Has required packages\n")
                return python_path
            else:
                FreeCAD.Console.PrintMessage(f"  -> Missing required packages\n")
    
    # Provide helpful error message
    FreeCAD.Console.PrintError("\n" + "="*60 + "\n")
    FreeCAD.Console.PrintError("ERROR: Could not find Python with PyCUDA installed!\n")
    FreeCAD.Console.PrintError("="*60 + "\n")
    FreeCAD.Console.PrintError("The raytracer requires these packages: numpy, trimesh, pycuda, matplotlib\n")
    FreeCAD.Console.PrintError("\nTo fix this:\n")
    FreeCAD.Console.PrintError("1. Open Command Prompt as Administrator\n")
    FreeCAD.Console.PrintError("2. Run: pip install numpy trimesh pycuda matplotlib\n")
    FreeCAD.Console.PrintError("\nOr set MANUAL_PYTHON_PATH in:\n")
    FreeCAD.Console.PrintError(f"   {os.path.join(os.path.dirname(__file__), 'config.py')}\n")
    if found_pythons:
        FreeCAD.Console.PrintError(f"\nPython installations found (but missing packages):\n")
        for p in found_pythons:
            FreeCAD.Console.PrintError(f"   {p}\n")
    FreeCAD.Console.PrintError("="*60 + "\n")
    return None


def _run_raytracer_via_server(lens_stl_path, absorber_stl_path, led_pos, led_dir,
                               num_rays, max_bounces, led_power, led_model, led_wavelength,
                               lens_n, lens_abs, absorber_n, absorber_abs) -> Optional[Dict]:
    """Run raytracing via the persistent server (internal helper).
    
    The server will automatically restart if it died (e.g., after Windows logout).
    
    Returns:
        dict with ray_paths, statistics, etc. or None on failure
    """
    _raytracer_client = get_raytracer_client()
    
    if _raytracer_client is None:
        FreeCAD.Console.PrintWarning("Raytracer client is None - server not started?\n")
        return None
    
    try:
        # Convert FreeCAD vectors to lists
        led_pos_list = [led_pos.x, led_pos.y, led_pos.z] if hasattr(led_pos, 'x') else list(led_pos)
        led_dir_list = [led_dir.x, led_dir.y, led_dir.z] if hasattr(led_dir, 'x') else list(led_dir)
        
        # Call the server (auto-restart is handled by the client)
        result = _raytracer_client.trace(
            lens_stl=lens_stl_path,
            absorber_stl=absorber_stl_path,
            led_pos=led_pos_list,
            led_dir=led_dir_list,
            rays=num_rays,
            bounces=max_bounces,
            led_power=led_power,
            led_angle=LED_MODELS.get(led_model, {}).get('half_angle', DEFAULT_LED_HALF_ANGLE),
            wavelength=led_wavelength,
            emitter_size=DEFAULT_LED_EMITTER_SIZE,
            lambertian=True,
            datasheet_directivity=True,
            led_model=led_model,
            refractive_index=lens_n,
            absorption=lens_abs,
            absorber_refractive_index=absorber_n,
            absorber_absorption=absorber_abs
        )
        
        # Check for errors
        if result is None:
            FreeCAD.Console.PrintWarning("Server returned None\n")
            return None
        
        success = result.get('success', False)
        error = result.get('error', None)
        
        if not success:
            if error:
                FreeCAD.Console.PrintError(f"Raytracer error: {error}\n")
            return None
        
        # Convert server response format to expected format
        # Server returns: absorber_exits = [{'end': [x,y,z], 'intensity': float}, ...]
        # Code expects: exit_positions = [[x,y,z], ...], exit_intensities = [float, ...]
        
        absorber_exits = result.get('absorber_exits', [])
        statistics = result.get('statistics', {})
        
        exit_positions = []
        exit_intensities = []
        
        for exit_data in absorber_exits:
            if isinstance(exit_data, dict):
                exit_positions.append(exit_data.get('end', [0, 0, 0]))
                exit_intensities.append(exit_data.get('intensity', 1.0))
        
        # Build the response in the expected format
        converted_result = {
            'success': True,
            'exit_positions': np.array(exit_positions) if exit_positions else np.array([]).reshape(0, 3),
            'exit_intensities': np.array(exit_intensities) if exit_intensities else np.array([]),
            'lens_hits': statistics.get('lens_entries', statistics.get('initial_rays', 0)),
            'absorber_entries': statistics.get('absorber_entries', 0),  # Rays that entered absorber
            'absorber_exits': statistics.get('absorber_exits', len(exit_positions)),
            'total_segments': statistics.get('total_segments', 0),
            'initial_rays': statistics.get('initial_rays', 0),
        }
        
        return converted_result
        
    except ConnectionError as e:
        # Server died and could not be restarted
        FreeCAD.Console.PrintError(f"Raytracer server connection failed: {e}\n")
        FreeCAD.Console.PrintWarning("The server may have been terminated (e.g., Windows logout).\n")
        FreeCAD.Console.PrintWarning("Please restart the optimization to restart the server.\n")
        return None
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Server raytracing failed: {e}\n")
        return None


def _run_raytracer_via_subprocess(lens_stl_path, absorber_stl_path, led_pos, led_dir,
                                   num_rays, max_bounces, led_power, led_model, led_wavelength,
                                   lens_n, lens_abs, absorber_n, absorber_abs) -> Optional[Dict]:
    """Run raytracing via subprocess (fallback when server not available).
    
    Returns:
        dict with exit_positions, exit_intensities, etc. or None on failure
    """
    import tempfile
    import json
    
    global _cached_python_exe
    
    # Get Python executable (use cached if available)
    if _cached_python_exe is None:
        _cached_python_exe = find_python_executable()
    
    python_exe = _cached_python_exe
    if python_exe is None:
        FreeCAD.Console.PrintError("Cannot run raytracer: No Python with PyCUDA found\n")
        return None
    
    # Path to wrapper script
    wrapper_script = os.path.join(RAYTRACER_PATH, "raytracer_wrapper.py")
    if not os.path.exists(wrapper_script):
        FreeCAD.Console.PrintError(f"Raytracer wrapper not found: {wrapper_script}\n")
        return None
    
    # Create temp file for JSON output
    output_json = tempfile.mktemp(suffix='_raytracer_result.json')
    
    try:
        # Convert FreeCAD vectors to lists
        led_pos_list = [led_pos.x, led_pos.y, led_pos.z] if hasattr(led_pos, 'x') else list(led_pos)
        led_dir_list = [led_dir.x, led_dir.y, led_dir.z] if hasattr(led_dir, 'x') else list(led_dir)
        
        # Build command
        cmd = [
            python_exe, wrapper_script,
            '--lens-stl', lens_stl_path,
            '--absorber-stl', absorber_stl_path,
            '--output-json', output_json,
            '--rays', str(num_rays),
            '--bounces', str(max_bounces),
            '--led-pos', str(led_pos_list[0]), str(led_pos_list[1]), str(led_pos_list[2]),
            '--led-dir', str(led_dir_list[0]), str(led_dir_list[1]), str(led_dir_list[2]),
            '--led-power', str(led_power),
            '--led-model', led_model,
            '--wavelength', str(led_wavelength),
            '--refractive-index', str(lens_n),
            '--absorption', str(lens_abs),
            '--absorber-refractive-index', str(absorber_n),
            '--absorber-absorption', str(absorber_abs),
            '--no-display'
        ]
        
        FreeCAD.Console.PrintMessage(f"Running raytracer subprocess ({num_rays} rays)...\n")
        
        # Run subprocess
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            creationflags=creationflags,
            cwd=RAYTRACER_PATH
        )
        
        if result.returncode != 0:
            FreeCAD.Console.PrintError(f"Raytracer subprocess failed (code {result.returncode})\n")
            if result.stderr:
                # Only print first 500 chars of stderr to avoid spam
                FreeCAD.Console.PrintError(f"Error: {result.stderr[:500]}\n")
            return None
        
        # Read JSON result
        if not os.path.exists(output_json):
            FreeCAD.Console.PrintError("Raytracer did not produce output file\n")
            return None
        
        with open(output_json, 'r') as f:
            data = json.load(f)
        
        # Convert to expected format
        exit_positions = []
        exit_intensities = []
        
        for path in data.get('ray_paths', []):
            if path.get('is_absorber_exit', False):
                exit_positions.append(path['end'])
                exit_intensities.append(path.get('intensity', 1.0))
        
        stats = data.get('statistics', {})
        
        return {
            'success': True,
            'exit_positions': np.array(exit_positions) if exit_positions else np.array([]),
            'exit_intensities': np.array(exit_intensities) if exit_intensities else np.array([]),
            'lens_hits': stats.get('initial_rays', num_rays),
            'absorber_exits': stats.get('absorber_exits', len(exit_positions)),
            'total_rays': num_rays
        }
        
    except subprocess.TimeoutExpired:
        FreeCAD.Console.PrintError("Raytracer subprocess timed out (>10 minutes)\n")
        return None
    except Exception as e:
        FreeCAD.Console.PrintError(f"Raytracer subprocess error: {e}\n")
        return None
    finally:
        # Clean up temp file
        try:
            if os.path.exists(output_json):
                os.remove(output_json)
        except:
            pass


def run_raytracer_evaluation(lens_stl_path, absorber_stl_path, led_pos, led_dir,
                             num_rays=1000, max_bounces=50, led_power=None,
                             led_model="U405", led_wavelength=None, led2_config=None,
                             lens_material=None, absorber_material=None) -> Optional[Dict]:
    """Run raytracer and return absorber exit data.
    
    Args:
        lens_stl_path: Path to lens STL file
        absorber_stl_path: Path to absorber STL file
        led_pos: LED 1 position vector
        led_dir: LED 1 direction vector
        num_rays: Total number of rays to trace
        max_bounces: Maximum bounces per ray
        led_power: LED 1 radiant power in mW
        led_model: LED 1 model identifier (U375/U385/U395/U405)
        led_wavelength: LED 1 wavelength in nm
        led2_config: Optional dict for LED 2
        lens_material: Optional dict with refractive_index and absorption_coeff
        absorber_material: Optional dict with refractive_index and absorption_coeff
    
    Returns:
        dict with exit_positions, exit_intensities, uniformity_cv, total_exits
    """
    # Get material properties
    lens_n = lens_material.get('refractive_index', LENS_REFRACTIVE_INDEX) if lens_material else LENS_REFRACTIVE_INDEX
    lens_abs = lens_material.get('absorption_coeff', LENS_ABSORPTION_COEFF) if lens_material else LENS_ABSORPTION_COEFF
    absorber_n = absorber_material.get('refractive_index', ABSORBER_REFRACTIVE_INDEX) if absorber_material else ABSORBER_REFRACTIVE_INDEX
    absorber_abs = absorber_material.get('absorption_coeff', ABSORBER_ABSORPTION_COEFF) if absorber_material else ABSORBER_ABSORPTION_COEFF
    
    # Use default LED power if not specified
    if led_power is None:
        led_power = current_to_radiant_power(DEFAULT_LED_CURRENT, led_model)
    
    # Get wavelength from model if not specified
    if led_wavelength is None:
        led_wavelength = LED_MODELS.get(led_model, {}).get('wavelength', DEFAULT_LED_WAVELENGTH)
    
    # Try server-based raytracing first
    result = _run_raytracer_via_server(
        lens_stl_path, absorber_stl_path,
        led_pos, led_dir,
        num_rays, max_bounces,
        led_power, led_model, led_wavelength,
        lens_n, lens_abs, absorber_n, absorber_abs
    )
    
    if result is not None:
        return result
    
    # No server available - raytracing not possible
    FreeCAD.Console.PrintError("Raytracer not available - cannot evaluate lens design\n")
    return None


def calculate_efficiency(result, num_rays):
    """Calculate raytracing efficiency (rays exiting absorber / total rays)."""
    if result is None or 'exit_positions' not in result:
        return 0.0
    
    total_exits = len(result.get('exit_positions', []))
    return total_exits / num_rays if num_rays > 0 else 0.0


def calculate_lens_entry_rate(result, num_rays):
    """Calculate fraction of rays that hit the lens."""
    if result is None:
        return 0.0
    
    # Get lens hits from result statistics
    lens_hits = result.get('lens_hits', num_rays)
    return lens_hits / num_rays if num_rays > 0 else 0.0


def calculate_absorber_capture_rate(result, num_rays):
    """Calculate what percentage of rays that hit the lens also reach the absorber.
    
    Uses absorber_entries (rays that entered absorber) divided by lens_hits
    (rays that hit the lens). This measures how well the lens directs light
    to the absorber vs light escaping.
    
    Args:
        result: Raytracer result dictionary
        num_rays: Number of initial rays generated
        
    Returns:
        Float 0.0 to 1.0 (1.0 = all rays that hit lens reach absorber)
    """
    if result is None:
        return 0.0
    
    # Get absorber entries (rays that entered the absorber)
    absorber_entries = result.get('absorber_entries', 0)
    lens_hits = result.get('lens_hits', num_rays)
    
    # Return the fraction of lens-hitting rays that reach the absorber
    return absorber_entries / lens_hits if lens_hits > 0 else 0.0
