"""
Lens Optimizer Modules Package
==============================

A modular package for lens geometry optimization in FreeCAD.

This package provides functions for:
- Evolutionary optimization of lens geometry
- Centerline extraction and profile creation
- Lofting and sweeping operations
- Raytracer integration for uniformity analysis
- Mesh operations and symmetry
- Visualization and reporting

Usage:
    from LensOptimizerModules import main
    main.main()

Or import specific modules:
    from LensOptimizerModules.config import COMPUTE_BUDGETS
    from LensOptimizerModules.evolutionary_engine import evolve_lens

Modules:
--------
- config: Configuration constants and compute budgets
- materials: Material properties for lens and absorber
- data_classes: ProfileParams, Individual, CenterlinePoint
- sleep_prevention: Windows sleep prevention utilities
- server_management: Persistent raytracer server management
- raytracer: Bundled CUDA raytracer engine (subpackage)
- geometry_validation: Envelope constraint validation
- centerline_extraction: Centerline from envelope/sketch
- profile_creation: Polygon and circle profiles
- lofting: Lofted and swept lens creation
- entry_spheres: Spherical LED entry surfaces
- user_interface: Object selection and direction input
- dialogs: Configuration dialogs
- mesh_operations: Mesh conversion and manipulation
- mesh_symmetry: Y-axis symmetric mesh creation
- geometry_constraints: Envelope clamping
- raytracer_integration: Python detection and raytracing
- uniformity_analysis: Grid-based uniformity analysis
- visualization: FreeCAD visualization objects
- surface_adjustment: Mesh deformation calculations
- optimization_loop: Legacy mesh-based optimization
- evolutionary_core: Mutation, crossover, selection
- evolutionary_engine: Main evolutionary optimization
- analysis: Raytracing analysis of existing geometry
- output_generation: Reports and lens objects
- main: Main entry point

Author: AI Assistant
Version: 2.0.0 (Modular)
"""

__version__ = "2.0.0"
__author__ = "AI Assistant"

# Import main entry point for convenience
from .main import main

# Import commonly used items
from .config import COMPUTE_BUDGETS, LED_MODELS, current_to_radiant_power
from .materials import (
    LENS_MATERIAL_NAME,
    LENS_REFRACTIVE_INDEX,
    ABSORBER_MATERIAL_NAME,
    ABSORBER_REFRACTIVE_INDEX
)
from .data_classes import ProfileParams, Individual, CenterlinePoint

# Import key functions
from .evolutionary_engine import evolve_lens
from .analysis import analyze_lens
from .optimization_loop import optimize_lens
from .user_interface import get_selected_objects, get_led_position
from .dialogs import show_compute_budget_dialog

__all__ = [
    # Main entry point
    'main',
    
    # Configuration
    'COMPUTE_BUDGETS',
    'LED_MODELS',
    'current_to_radiant_power',
    
    # Materials
    'LENS_MATERIAL_NAME',
    'LENS_REFRACTIVE_INDEX',
    'ABSORBER_MATERIAL_NAME',
    'ABSORBER_REFRACTIVE_INDEX',
    
    # Data classes
    'ProfileParams',
    'Individual',
    'CenterlinePoint',
    
    # Key functions
    'evolve_lens',
    'analyze_lens',
    'optimize_lens',
    'get_selected_objects',
    'get_led_position',
    'show_compute_budget_dialog',
]
