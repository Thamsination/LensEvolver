"""
Raytracer Server
================
A persistent server that keeps the raytracer loaded and CUDA initialized,
accepting requests over a TCP socket to eliminate subprocess startup overhead.

Usage:
    python raytracer_server.py [--port PORT]

The server accepts JSON requests over TCP and returns JSON responses.
"""

import sys
import os
import json
import socket
import struct
import threading
import time
import traceback
import argparse
import signal

# Set UTF-8 encoding for stdout/stderr
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the directory containing raytracer to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Global state
raytracer_module = None
server_socket = None
running = True
request_count = 0
total_time = 0.0

DEFAULT_PORT = 5555

# ============================================================================
# PERFORMANCE OPTIMIZATION: Caching
# ============================================================================
# Cache absorber mesh (same for all evaluations in an optimization run)
_absorber_cache = {}  # key: (file_path, n, absorption) -> trimesh object

# Cache tracer instance (reuse when absorber hasn't changed)
_cached_tracer = None
_cached_tracer_key = None  # (absorber_stl, absorber_n, absorber_abs)

# Performance tracking
_cache_hits = 0
_cache_misses = 0


def load_raytracer():
    """Load the raytracer module once at startup."""
    global raytracer_module
    
    import importlib.util
    
    # Try to use the raytracer with absorber support
    raytracer_with_absorber_path = os.path.join(script_dir, "raytracer_with_absorber.py")
    raytracer_original_path = os.path.join(script_dir, "raytracer-v2.2.0.py")
    
    if os.path.exists(raytracer_with_absorber_path):
        raytracer_path = raytracer_with_absorber_path
        print("Loading raytracer_with_absorber.py...", file=sys.stderr)
    elif os.path.exists(raytracer_original_path):
        raytracer_path = raytracer_original_path
        print("Loading raytracer-v2.2.0.py...", file=sys.stderr)
    else:
        raise FileNotFoundError(f"No raytracer found. Checked:\n  {raytracer_with_absorber_path}\n  {raytracer_original_path}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("raytracer_v2_2_0", raytracer_path)
    raytracer_module = importlib.util.module_from_spec(spec)
    
    # Execute the module (this will initialize CUDA)
    start_time = time.time()
    spec.loader.exec_module(raytracer_module)
    load_time = time.time() - start_time
    
    print(f"Raytracer module loaded in {load_time:.2f}s", file=sys.stderr)
    
    # Check if CUDA is available
    if hasattr(raytracer_module, 'CUDA_AVAILABLE'):
        print(f"CUDA available: {raytracer_module.CUDA_AVAILABLE}", file=sys.stderr)
    
    return raytracer_module


def cleanup_cuda_context():
    """Clean up CUDA context properly"""
    global raytracer_module
    try:
        import pycuda.driver as cuda
        max_pops = 10
        pop_count = 0
        while pop_count < max_pops:
            try:
                cuda.Context.pop()
                pop_count += 1
            except:
                break
        if pop_count > 0:
            print(f"Cleaned up {pop_count} CUDA context(s)", file=sys.stderr)
    except:
        pass


def get_or_create_tracer(absorber_stl, absorber_n, absorber_abs):
    """Get cached tracer or create new one if absorber changed.
    
    OPTIMIZATION #2: Reuse tracer instance when absorber hasn't changed.
    This saves ~200-500ms per request by avoiding tracer recreation.
    """
    global _cached_tracer, _cached_tracer_key, _cache_hits, _cache_misses, raytracer_module
    
    cache_key = (absorber_stl, absorber_n, absorber_abs)
    
    if _cached_tracer is not None and _cached_tracer_key == cache_key:
        _cache_hits += 1
        return _cached_tracer, True  # Return cached tracer
    
    _cache_misses += 1
    
    # Need to create new tracer
    if hasattr(raytracer_module, 'UVLEDRayTracerWithAbsorber'):
        UVLEDRayTracer = raytracer_module.UVLEDRayTracerWithAbsorber
    else:
        UVLEDRayTracer = raytracer_module.UVLEDRayTracer
    
    import inspect
    sig = inspect.signature(UVLEDRayTracer.__init__)
    
    if 'absorber_stl' in sig.parameters:
        tracer = UVLEDRayTracer(
            step_file_path=None,
            absorber_stl=absorber_stl,
            absorber_refractive_index=absorber_n,
            absorber_absorption=absorber_abs
        )
    else:
        tracer = UVLEDRayTracer()
    
    _cached_tracer = tracer
    _cached_tracer_key = cache_key
    
    return tracer, False  # Return new tracer


def process_request(request):
    """Process a single raytracing request and return results.
    
    OPTIMIZATIONS IMPLEMENTED:
    1. Absorber mesh caching - absorber is cached and reused
    2. Tracer instance reuse - tracer is reused when absorber unchanged
    3. Minimal response mode - only essential data returned when enabled
    """
    global raytracer_module, request_count, total_time, _cache_hits, _cache_misses
    
    import numpy as np
    
    start_time = time.time()
    request_count += 1
    
    try:
        # Extract parameters from request
        lens_stl = request.get('lens_stl')
        absorber_stl = request.get('absorber_stl')
        led_pos = request.get('led_pos', [0, 0, 2])
        led_dir = request.get('led_dir', [0, 0, 1])
        num_rays = request.get('rays', 10000)
        max_bounces = request.get('bounces', 1000)
        max_ray_length = request.get('max_ray_length', 200.0)
        
        # LED parameters
        led_power = request.get('led_power', 1420.0)
        led_angle = request.get('led_angle', 70.0)
        wavelength = request.get('wavelength', 405)
        emitter_size = request.get('emitter_size', 1.0)
        lambertian = request.get('lambertian', True)
        datasheet_directivity = request.get('datasheet_directivity', True)
        led_model = request.get('led_model', 'U405')
        
        # Material parameters
        refractive_index = request.get('refractive_index', 1.535)
        absorption = request.get('absorption', 0.001)
        absorber_refractive_index = request.get('absorber_refractive_index', 1.585)
        absorber_absorption = request.get('absorber_absorption', 0.05)
        
        # OPTIMIZATION #3: Minimal response mode (default: True for performance)
        minimal_response = request.get('minimal_response', True)
        
        # OPTIMIZATION #2: Get or create cached tracer
        tracer, was_cached = get_or_create_tracer(
            absorber_stl, absorber_refractive_index, absorber_absorption
        )
        
        # Set lens material properties (always update in case they changed)
        tracer.material.refractive_index = refractive_index
        tracer.material.absorption_coeff = absorption
        
        # Load lens geometry (this changes every evaluation)
        # The tracer.load_step_model will update the mesh while preserving absorber
        tracer.load_step_model(lens_stl)
        
        # Setup LED source
        tracer.setup_led_source(
            position=led_pos,
            direction=led_dir,
            power=led_power,
            wavelength=wavelength * 1e-9,  # Convert nm to meters
            half_angle=led_angle,
            emitter_size=emitter_size,
            lambertian=lambertian,
            use_datasheet_directivity=datasheet_directivity,
            model=led_model
        )
        
        # Run simulation
        ray_paths, intensities = tracer.simulate(
            num_rays=num_rays,
            max_bounces=max_bounces,
            max_ray_length=max_ray_length
        )
        
        # Get diagnostics if available
        diagnostics = getattr(tracer, 'diagnostics', None)
        
        # OPTIMIZATION #3: Minimal response mode
        # Only process and return what's needed for fitness calculation
        if minimal_response:
            # FAST PATH: Use numpy arrays for vectorized processing (1000x faster)
            total_segments = len(ray_paths)
            
            if total_segments > 0:
                # Pre-extract all data into numpy arrays ONCE (avoid repeated dict access)
                bounces = np.array([p.get('bounce', 0) for p in ray_paths], dtype=np.int32)
                is_exit = np.array([p.get('is_exit', False) for p in ray_paths], dtype=bool)
                is_lens_exit = np.array([p.get('is_lens_exit', False) for p in ray_paths], dtype=bool)
                is_absorber_hit = np.array([p.get('is_absorber_hit', False) for p in ray_paths], dtype=bool)
                is_absorber_entry = np.array([p.get('is_absorber_entry', False) for p in ray_paths], dtype=bool)
                is_absorber_exit = np.array([p.get('is_absorber_exit', False) for p in ray_paths], dtype=bool)
                
                # Vectorized statistics (no Python loops!)
                initial_rays = int(np.sum(bounces == 0))
                lens_entries = int(np.sum(bounces == 1))  # First hit on lens
                max_bounce = int(np.max(bounces)) if len(bounces) > 0 else 0
                exit_segments = int(np.sum(is_exit))
                lens_exits_total = int(np.sum(is_lens_exit))
                absorber_hits_total = int(np.sum(is_absorber_hit))
                absorber_entries_total = int(np.sum(is_absorber_entry))
                absorber_exits_total = int(np.sum(is_absorber_exit))
                
                # Extract absorber exits using boolean indexing (fast!)
                exit_indices = np.where(is_absorber_exit)[0]
                absorber_exits = []
                for idx in exit_indices:
                    path = ray_paths[idx]
                    end = path['end']
                    if isinstance(end, np.ndarray):
                        end = end.tolist()
                    elif not isinstance(end, list):
                        end = list(end)
                    absorber_exits.append({
                        'end': end,
                        'intensity': float(path.get('intensity', 1.0))
                    })
                
                # Extract lens exits for heatmap visualization
                lens_exit_indices = np.where(is_lens_exit)[0]
                lens_exits = []
                for idx in lens_exit_indices:
                    path = ray_paths[idx]
                    end = path['end']
                    if isinstance(end, np.ndarray):
                        end = end.tolist()
                    elif not isinstance(end, list):
                        end = list(end)
                    lens_exits.append({
                        'end': end,
                        'intensity': float(path.get('intensity', 1.0))
                    })
            else:
                initial_rays = 0
                lens_entries = 0
                max_bounce = 0
                exit_segments = 0
                lens_exits_total = 0
                absorber_hits_total = 0
                absorber_entries_total = 0
                absorber_exits_total = 0
                absorber_exits = []
                lens_exits = []
            
            response = {
                'success': True,
                'minimal_response': True,
                'absorber_exits': absorber_exits,
                'lens_exits': lens_exits,
                'ray_paths': [],  # Empty in minimal mode
                'hit_points': [],  # Empty in minimal mode
                'statistics': {
                    'total_segments': total_segments,
                    'initial_rays': initial_rays,
                    'lens_entries': lens_entries,
                    'lens_exits': lens_exits_total,
                    'exit_segments': exit_segments,
                    'hit_points_count': 0,
                    'absorber_hits': absorber_hits_total,
                    'absorber_exits': absorber_exits_total,
                    'absorber_entries': absorber_entries_total,
                    'max_bounce': max_bounce,
                    'cache_hits': _cache_hits,
                    'cache_misses': _cache_misses,
                },
                'diagnostics': diagnostics
            }
        else:
            # Full response mode (for visualization, debugging)
            processed_paths = []
            hit_points = []
            
            for path in ray_paths:
                start = np.array(path['start']) if isinstance(path['start'], (list, np.ndarray)) else path['start']
                end = np.array(path['end']) if isinstance(path['end'], (list, np.ndarray)) else path['end']
                
                is_absorber_hit = path.get('is_absorber_hit', False)
                is_absorber_exit = path.get('is_absorber_exit', False)
                is_absorber_entry = path.get('is_absorber_entry', False)
                
                # Calculate ray direction and length
                direction = end - start
                ray_length = np.linalg.norm(direction)
                
                # Limit very long rays
                if ray_length > max_ray_length * 1.5:
                    if ray_length > 0:
                        direction_normalized = direction / ray_length
                        end = start + direction_normalized * max_ray_length
                
                is_hit_point = path.get('bounce', 0) > 0 and not is_absorber_hit
                
                if is_hit_point:
                    hit_points.append({
                        'position': end.tolist() if isinstance(end, np.ndarray) else list(end),
                        'bounce': int(path.get('bounce', 0)),
                        'is_exit': bool(path.get('is_exit', False)),
                        'intensity': float(path.get('intensity', 1.0))
                    })
                
                processed_paths.append({
                    'start': start.tolist() if isinstance(start, np.ndarray) else list(start),
                    'end': end.tolist() if isinstance(end, np.ndarray) else list(end),
                    'intensity': float(path['intensity']),
                    'bounce': int(path.get('bounce', 0)),
                    'is_exit': bool(path.get('is_exit', False)),
                    'is_hit_point': bool(is_hit_point),
                    'is_absorber_hit': bool(is_absorber_hit),
                    'is_absorber_exit': bool(is_absorber_exit),
                    'is_absorber_entry': bool(is_absorber_entry)
                })
            
            # Build full response
            absorber_hits_total = sum(1 for p in processed_paths if p.get('is_absorber_hit', False))
            absorber_exits_total = sum(1 for p in processed_paths if p.get('is_absorber_exit', False))
            absorber_entries_total = sum(1 for p in processed_paths if p.get('is_absorber_entry', False))
            
            response = {
                'success': True,
                'minimal_response': False,
                'ray_paths': processed_paths,
                'hit_points': hit_points,
                'statistics': {
                    'total_segments': len(processed_paths),
                    'initial_rays': sum(1 for p in processed_paths if p.get('bounce', 0) == 0),
                    'lens_entries': sum(1 for p in processed_paths if p.get('bounce', 0) == 1),
                    'exit_segments': sum(1 for p in processed_paths if p.get('is_exit', False)),
                    'hit_points_count': len(hit_points),
                    'absorber_hits': absorber_hits_total,
                    'absorber_exits': absorber_exits_total,
                    'absorber_entries': absorber_entries_total,
                    'max_bounce': max((p.get('bounce', 0) for p in processed_paths), default=0),
                    'cache_hits': _cache_hits,
                    'cache_misses': _cache_misses,
                },
                'diagnostics': diagnostics
            }
        
        elapsed = time.time() - start_time
        total_time += elapsed
        avg_time = total_time / request_count
        cache_status = "CACHED" if was_cached else "NEW"
        mode_status = "minimal" if minimal_response else "full"
        print(f"Request #{request_count}: {num_rays} rays in {elapsed:.2f}s (avg: {avg_time:.2f}s) [{cache_status}, {mode_status}]", file=sys.stderr)
        
        return response
        
    except Exception as e:
        elapsed = time.time() - start_time
        total_time += elapsed
        error_msg = f"Error processing request: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def send_message(sock, data):
    """Send a JSON message with length prefix."""
    json_data = json.dumps(data).encode('utf-8')
    length = len(json_data)
    # Send 4-byte length prefix (big-endian)
    sock.sendall(struct.pack('>I', length))
    sock.sendall(json_data)


def recv_message(sock):
    """Receive a JSON message with length prefix."""
    # Receive 4-byte length prefix
    length_data = recv_exactly(sock, 4)
    if not length_data:
        return None
    length = struct.unpack('>I', length_data)[0]
    
    # Receive the JSON data
    json_data = recv_exactly(sock, length)
    if not json_data:
        return None
    
    return json.loads(json_data.decode('utf-8'))


def recv_exactly(sock, n):
    """Receive exactly n bytes."""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def handle_client(client_socket, client_addr):
    """Handle a single client connection."""
    global running
    
    print(f"Client connected: {client_addr}", file=sys.stderr)
    
    try:
        while running:
            # Receive request
            request = recv_message(client_socket)
            if request is None:
                break
            
            # Check for shutdown command
            if request.get('command') == 'shutdown':
                print("Shutdown command received", file=sys.stderr)
                response = {'success': True, 'message': 'Server shutting down'}
                send_message(client_socket, response)
                running = False
                break
            
            # Check for ping command
            if request.get('command') == 'ping':
                response = {'success': True, 'message': 'pong', 'requests_processed': request_count}
                send_message(client_socket, response)
                continue
            
            # Process raytracing request
            response = process_request(request)
            send_message(client_socket, response)
            
    except Exception as e:
        print(f"Error handling client: {e}", file=sys.stderr)
    finally:
        client_socket.close()
        print(f"Client disconnected: {client_addr}", file=sys.stderr)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global running
    print(f"\nReceived signal {signum}, shutting down...", file=sys.stderr)
    running = False
    if server_socket:
        try:
            server_socket.close()
        except:
            pass


def run_server(port=DEFAULT_PORT):
    """Run the raytracer server."""
    global server_socket, running
    
    # Load raytracer module
    print("=" * 60, file=sys.stderr)
    print("RAYTRACER SERVER", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    load_raytracer()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('127.0.0.1', port))
        server_socket.listen(1)
        server_socket.settimeout(1.0)  # Allow periodic check for shutdown
        
        print(f"Server listening on port {port}", file=sys.stderr)
        print("Ready to accept connections...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # Write port to a file for client to discover
        port_file = os.path.join(script_dir, '.raytracer_server_port')
        with open(port_file, 'w') as f:
            f.write(str(port))
        
        while running:
            try:
                client_socket, client_addr = server_socket.accept()
                handle_client(client_socket, client_addr)
            except socket.timeout:
                continue
            except OSError as e:
                if running:
                    print(f"Socket error: {e}", file=sys.stderr)
                break
                
    finally:
        print("Shutting down server...", file=sys.stderr)
        try:
            server_socket.close()
        except:
            pass
        
        # Clean up port file
        try:
            os.remove(os.path.join(script_dir, '.raytracer_server_port'))
        except:
            pass
        
        # Clean up CUDA
        cleanup_cuda_context()
        
        print(f"Server stopped. Processed {request_count} requests.", file=sys.stderr)
        if request_count > 0:
            print(f"Average time per request: {total_time / request_count:.2f}s", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Raytracer Server')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to listen on (default: {DEFAULT_PORT})')
    args = parser.parse_args()
    
    run_server(args.port)


if __name__ == "__main__":
    main()

