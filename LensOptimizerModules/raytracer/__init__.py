"""
CUDA Raytracer Package
======================

This subpackage contains the CUDA-accelerated raytracing engine and related modules.

Modules:
    - raytracer-v2.2.0: Core CUDA raytracer implementation
    - raytracer_with_absorber: Extended raytracer with absorber material support
    - raytracer_client: Client for communicating with the raytracer server
    - raytracer_server: Persistent server that keeps CUDA initialized
    - raytracer_wrapper: Wrapper script for JSON output

Usage:
    The raytracer is typically accessed through the server_management module
    in the parent LensOptimizerModules package, which handles starting/stopping
    the raytracer server and routing requests through the client.
"""

import os

# Path to this raytracer package
RAYTRACER_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ['RAYTRACER_DIR']
