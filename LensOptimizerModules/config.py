"""
Configuration constants for the Lens Optimizer.

This module contains all configuration parameters, compute budget presets,
LED models, and optimization settings.
"""

import FreeCAD

# ============================================================================
# COMPUTE BUDGET PRESETS
# ============================================================================
# NOTE: absorber_resolution is coarser than mesh_resolution because the absorber
# only needs to catch rays, not accurately model refraction. This dramatically
# improves raytracing performance (10-30x faster for large absorbers).

COMPUTE_BUDGETS = {
    "Quick": {
        "rays": 50000,
        "iterations": 15,
        "mesh_resolution": 0.5,  # mm - lens mesh resolution (smaller = finer)
        "absorber_resolution": 1.0,  # mm - absorber mesh (coarser is fine)
        "convergence_threshold": 0.05,  # 5% improvement threshold
        "description": "Fast iteration for testing (~2-5 min)"
    },
    "Medium": {
        "rays": 100000,
        "iterations": 50,
        "mesh_resolution": 0.25,  # mm - lens mesh resolution
        "absorber_resolution": 1.0,  # mm - absorber mesh
        "convergence_threshold": 0.02,  # 2% improvement threshold
        "description": "Balanced quality/speed (~30-60 min)"
    },
    "Thorough": {
        "rays": 500000,
        "iterations": 200,
        "mesh_resolution": 0.25,  # mm - lens mesh (was 0.1, too fine)
        "absorber_resolution": 1.0,  # mm - absorber mesh (coarse is fine)
        "convergence_threshold": 0.005,  # 0.5% improvement threshold
        "description": "Maximum quality (~10-30 min)"
    }
}

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

DEFAULT_LEARNING_RATE = 0.5  # How much to adjust surface per iteration (increased for visible changes)
DEFAULT_SMOOTHING_FACTOR = 0.2  # Surface smoothing strength (reduced to preserve deformations)
DEFAULT_MIN_ADJUSTMENT = 0.05  # Minimum deformation per vertex in mm (ensures visible changes)
DEFAULT_MAX_BOUNCES = 50
DEFAULT_STARTING_SCALE = 0.7  # Start at 70% (conservative for domed envelopes)
DEFAULT_EFFICIENCY_THRESHOLD = 0.95  # Minimum 95% rays must exit absorber
DEFAULT_SUBDIVISION_ITERATIONS = 2  # Subdivide envelope mesh for better vertex distribution
MAX_VERTEX_DISPLACEMENT = 0.3  # Maximum vertex movement per iteration (mm) - prevents self-intersection
DEFAULT_VOLUME_TOLERANCE = 50.0  # mm³ allowed outside envelope (reduced for stricter validation)

# ============================================================================
# RAYTRACER CONFIGURATION
# ============================================================================

# Raytracer path - now bundled within LensOptimizerModules
import os as _os
_MODULE_DIR = _os.path.dirname(_os.path.abspath(__file__))
RAYTRACER_PATH = _os.path.join(_MODULE_DIR, "raytracer")

# Manual Python path override (set to your Python with PyCUDA if auto-detection fails)
MANUAL_PYTHON_PATH = None  # e.g., r"C:\Users\chris\AppData\Local\Programs\Python\Python312\python.exe"

# Raytracer server configuration
USE_RAYTRACER_SERVER = True  # Set to False to use legacy subprocess mode
RAYTRACER_SERVER_PORT = 5555
RAYTRACER_SERVER_STARTUP_TIMEOUT = 120  # seconds to wait for server startup
RAYTRACER_REQUEST_TIMEOUT = 120  # seconds to wait for raytracing response (was 600s, caused 10min pauses)

# ============================================================================
# LED PARAMETERS
# ============================================================================

# LED parameters (NVSU119CT-U405 defaults)
DEFAULT_LED_HALF_ANGLE = 70.0
DEFAULT_LED_WAVELENGTH = 405
DEFAULT_LED_CURRENT = 700.0  # mA forward current (reference point for efficiency model)
DEFAULT_LED_EMITTER_SIZE = 1.0

# LED Model Database - NVSU119CT series from Nichia datasheet
# Reference current: 700mA, temperature: Ts=25°C
LED_MODELS = {
    "U375": {
        "wavelength": 375,      # nm peak wavelength
        "power_ref": 1160,      # mW radiant flux @ 700mA
        "name": "NVSU119CT-U375",
        "half_angle": 70.0,     # degrees (140° viewing angle)
    },
    "U385": {
        "wavelength": 385,
        "power_ref": 1450,
        "name": "NVSU119CT-U385",
        "half_angle": 70.0,
    },
    "U395": {
        "wavelength": 395,
        "power_ref": 1450,
        "name": "NVSU119CT-U395",
        "half_angle": 70.0,
    },
    "U405": {
        "wavelength": 405,
        "power_ref": 1420,
        "name": "NVSU119CT-U405",
        "half_angle": 70.0,
    },
}


def current_to_radiant_power(current_mA: float, model: str = "U405") -> float:
    """Convert LED forward current (mA) to radiant power (mW) for specified LED model.
    
    Based on Nichia NVSU119CT datasheet:
    - U375: 700mA → 1160mW typical radiant flux
    - U385: 700mA → 1450mW typical radiant flux
    - U395: 700mA → 1450mW typical radiant flux
    - U405: 700mA → 1420mW typical radiant flux
    - Operating range: 70mA (10% min) to 1400mA (absolute max)
    
    Uses a UV LED efficiency droop model where efficiency decreases
    at higher currents due to thermal effects and Auger recombination.
    
    Args:
        current_mA: LED forward current in milliamps
        model: LED model identifier ("U375", "U385", "U395", or "U405")
        
    Returns:
        Radiant power in milliwatts
    """
    # Get model-specific reference power
    if model not in LED_MODELS:
        FreeCAD.Console.PrintWarning(f"Unknown LED model '{model}', defaulting to U405\n")
        model = "U405"
    
    P_ref = LED_MODELS[model]["power_ref"]
    
    # Reference current from datasheet (I_F=700mA)
    I_ref = 700.0   # mA
    
    # Clamp to valid operating range
    current_mA = max(70.0, min(1400.0, current_mA))
    
    # UV LED efficiency droop model
    # Efficiency decreases at higher currents due to thermal effects
    # Typical droop exponent for UV LEDs is 0.10-0.15
    droop_exponent = 0.12
    efficiency_factor = (I_ref / current_mA) ** droop_exponent
    
    # Scale linearly with current, adjusted for efficiency droop
    radiant_power = (current_mA / I_ref) * P_ref * efficiency_factor
    
    return radiant_power


# ============================================================================
# PARAMETRIC LOFT OPTIMIZATION SETTINGS
# ============================================================================

DEFAULT_NUM_PROFILES = 5  # Number of cross-section profiles along centerline
MIN_NUM_PROFILES = 3  # Minimum profiles for valid loft geometry
MAX_NUM_PROFILES = 15  # Maximum profiles to prevent over-complexity
PROFILE_COUNT_MUTATION_RATE = 0.15  # 15% chance to add/remove profile during mutation
DEFAULT_PROFILE_SIDES = 6  # Starting number of polygon sides (hexagon)
MIN_PROFILE_SIDES = 3  # Triangle
MAX_PROFILE_SIDES = 32  # Nearly circular
DEFAULT_PROFILE_LEARNING_RATE = 0.3  # How much to adjust profile parameters per iteration

# ============================================================================
# EVOLUTIONARY OPTIMIZATION SETTINGS
# ============================================================================

DEFAULT_POPULATION_SIZE = 12  # Number of individuals per generation
DEFAULT_GENERATIONS = 10  # Number of generations to evolve
DEFAULT_ELITE_COUNT = 2  # Best individuals to keep unchanged each generation
DEFAULT_MUTATION_RATE = 0.3  # Probability of mutating each profile
DEFAULT_CROSSOVER_RATE = 0.7  # Probability of crossover vs pure mutation

# Conservative mutation ranges (to avoid invalid geometry)
MUTATION_RADIUS_RANGE = 0.5  # mm max change per mutation
MUTATION_SIDES_RANGE = 1  # max sides change per mutation  
MUTATION_ANGLE_RANGE = 15.0  # degrees max change per mutation
