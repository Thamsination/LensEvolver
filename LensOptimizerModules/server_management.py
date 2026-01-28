"""
Raytracer server management.

Handles starting, stopping, and monitoring the persistent raytracer server
for faster raytracing operations.
"""

import os
import importlib.util
import FreeCAD

from .config import (
    RAYTRACER_PATH,
    USE_RAYTRACER_SERVER,
    RAYTRACER_SERVER_PORT,
    RAYTRACER_SERVER_STARTUP_TIMEOUT,
    RAYTRACER_REQUEST_TIMEOUT
)

# Global raytracer client instance (initialized when optimization starts)
_raytracer_client = None
_raytracer_server_started = False


def get_raytracer_client():
    """Get the current raytracer client instance."""
    global _raytracer_client
    return _raytracer_client


def start_raytracer_server(python_exe):
    """Start the persistent raytracer server.
    
    Args:
        python_exe: Path to Python executable with PyCUDA
        
    Returns:
        True if server is running, False otherwise
    """
    global _raytracer_client, _raytracer_server_started
    
    if not USE_RAYTRACER_SERVER:
        FreeCAD.Console.PrintMessage("Raytracer server mode disabled in config\n")
        return False
    
    if python_exe is None:
        FreeCAD.Console.PrintError("Cannot start raytracer server: No Python executable found!\n")
        FreeCAD.Console.PrintError("Please install PyCUDA in your system Python or set MANUAL_PYTHON_PATH in config.py\n")
        return False
    
    try:
        # Import the client module
        client_path = os.path.join(RAYTRACER_PATH, "raytracer_client.py")
        FreeCAD.Console.PrintMessage(f"Looking for raytracer client at: {client_path}\n")
        
        if not os.path.exists(client_path):
            FreeCAD.Console.PrintWarning("Raytracer client not found, using subprocess mode\n")
            return False
        
        # Check if server script exists
        server_path = os.path.join(RAYTRACER_PATH, "raytracer_server.py")
        if not os.path.exists(server_path):
            FreeCAD.Console.PrintWarning(f"Raytracer server script not found: {server_path}\n")
            return False
        
        # Import dynamically
        spec = importlib.util.spec_from_file_location("raytracer_client", client_path)
        client_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client_module)
        
        # Create client instance
        FreeCAD.Console.PrintMessage(f"Using Python: {python_exe}\n")
        FreeCAD.Console.PrintMessage(f"Request timeout: {RAYTRACER_REQUEST_TIMEOUT}s\n")
        _raytracer_client = client_module.RaytracerClient(
            port=RAYTRACER_SERVER_PORT,
            python_exe=python_exe,
            request_timeout=RAYTRACER_REQUEST_TIMEOUT
        )
        
        # Start server
        FreeCAD.Console.PrintMessage(f"Starting raytracer server on port {RAYTRACER_SERVER_PORT}...\n")
        FreeCAD.Console.PrintMessage("(This may take 30-60 seconds for CUDA initialization)\n")
        _raytracer_client.start_server(timeout=RAYTRACER_SERVER_STARTUP_TIMEOUT)
        _raytracer_server_started = True
        FreeCAD.Console.PrintMessage("Raytracer server ready - performance mode enabled\n")
        return True
        
    except Exception as e:
        import traceback
        FreeCAD.Console.PrintError(f"Could not start raytracer server: {e}\n")
        FreeCAD.Console.PrintError(f"Traceback: {traceback.format_exc()}\n")
        FreeCAD.Console.PrintWarning("Falling back to subprocess mode (slower)\n")
        _raytracer_client = None
        _raytracer_server_started = False
        return False


def stop_raytracer_server():
    """Stop the persistent raytracer server."""
    global _raytracer_client, _raytracer_server_started
    
    if _raytracer_client is not None:
        try:
            _raytracer_client.stop_server()
            FreeCAD.Console.PrintMessage("Raytracer server stopped\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Error stopping raytracer server: {e}\n")
        finally:
            _raytracer_client = None
            _raytracer_server_started = False


def is_raytracer_server_running():
    """Check if the raytracer server is running."""
    global _raytracer_client
    if _raytracer_client is None:
        return False
    try:
        return _raytracer_client.is_running()
    except:
        return False
