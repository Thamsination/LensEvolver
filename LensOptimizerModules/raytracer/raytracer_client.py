"""
Raytracer Client
================
Client module for communicating with the persistent raytracer server.
This module is designed to be imported by FreeCAD macros.

Usage:
    from raytracer_client import RaytracerClient
    
    client = RaytracerClient()
    client.start_server()
    
    result = client.trace(
        lens_stl='path/to/lens.stl',
        absorber_stl='path/to/absorber.stl',
        led_pos=[0, 0, 2],
        led_dir=[0, 0, 1],
        ...
    )
    
    client.stop_server()
"""

import os
import sys
import json
import socket
import struct
import subprocess
import time
import threading

# Default settings
DEFAULT_PORT = 5555
DEFAULT_REQUEST_TIMEOUT = 120  # seconds (was 600s which caused 10-minute pauses)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class RaytracerClient:
    """Client for the persistent raytracer server."""
    
    def __init__(self, port=DEFAULT_PORT, python_exe=None, request_timeout=None):
        """
        Initialize the raytracer client.
        
        Args:
            port: Port to connect to (default: 5555)
            python_exe: Path to Python executable with PyCUDA (required for starting server)
            request_timeout: Socket timeout in seconds for raytracing requests (default: 120s)
        """
        self.port = port
        self.python_exe = python_exe
        self.request_timeout = request_timeout if request_timeout is not None else DEFAULT_REQUEST_TIMEOUT
        self.server_process = None
        self.socket = None
        self.connected = False
        self._lock = threading.Lock()
    
    def start_server(self, timeout=60):
        """
        Start the raytracer server if not already running.
        
        Args:
            timeout: Maximum seconds to wait for server startup
            
        Returns:
            True if server is running, False otherwise
        """
        # First check if server is already running
        if self._ping():
            print("Raytracer server already running")
            return True
        
        if self.python_exe is None:
            raise ValueError("python_exe must be specified to start server")
        
        server_script = os.path.join(SCRIPT_DIR, 'raytracer_server.py')
        if not os.path.exists(server_script):
            raise FileNotFoundError(f"Server script not found: {server_script}")
        
        # Start server process
        print(f"Starting raytracer server on port {self.port}...")
        
        # Build command with proper flags for Windows
        cmd = [self.python_exe, server_script, '--port', str(self.port)]
        
        # Start as background process with flags to survive session changes
        creationflags = 0
        if sys.platform == 'win32':
            # CREATE_NO_WINDOW: Hide console window
            # DETACHED_PROCESS: Detach from parent console (survives logout better)
            # CREATE_NEW_PROCESS_GROUP: New process group (better signal handling)
            # CREATE_BREAKAWAY_FROM_JOB: Break away from job object if any
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_BREAKAWAY_FROM_JOB = 0x01000000
            creationflags = CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            cwd=SCRIPT_DIR,
            # Don't inherit handles - helps with independence from parent
            close_fds=(sys.platform != 'win32')
        )
        
        # Wait for server to become ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._ping():
                print("Raytracer server started successfully")
                return True
            time.sleep(0.5)
            
            # Check if process died
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                error_msg = stderr.decode('utf-8', errors='replace') if stderr else "Unknown error"
                raise RuntimeError(f"Server process died during startup:\n{error_msg}")
        
        raise TimeoutError(f"Server did not start within {timeout} seconds")
    
    def stop_server(self):
        """Stop the raytracer server."""
        try:
            # Try graceful shutdown via command
            if self._connect():
                try:
                    self._send_message({'command': 'shutdown'})
                    self._recv_message()  # Wait for acknowledgment
                except:
                    pass
                self._disconnect()
        except:
            pass
        
        # Force kill if process is still running
        if self.server_process is not None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                try:
                    self.server_process.kill()
                except:
                    pass
            self.server_process = None
        
        print("Raytracer server stopped")
    
    def is_running(self):
        """Check if the server is running."""
        return self._ping()
    
    def ensure_running(self, timeout=60):
        """Ensure the server is running, restarting if necessary.
        
        Args:
            timeout: Maximum seconds to wait for server restart
            
        Returns:
            True if server is running, False if restart failed
        """
        if self._ping():
            return True
        
        # Server not responding - check if process died
        if self.server_process is not None:
            poll_result = self.server_process.poll()
            if poll_result is not None:
                print(f"Raytracer server process died (exit code: {poll_result})")
                self.server_process = None
        
        # Try to restart
        print("Attempting to restart raytracer server...")
        try:
            return self.start_server(timeout=timeout)
        except Exception as e:
            print(f"Failed to restart server: {e}")
            return False
    
    def trace(self, lens_stl, absorber_stl, led_pos, led_dir, 
              rays=10000, bounces=1000, max_ray_length=200.0,
              led_power=1420.0, led_angle=70.0, wavelength=405,
              emitter_size=1.0, lambertian=True, datasheet_directivity=True,
              led_model='U405', refractive_index=1.535, absorption=0.001,
              absorber_refractive_index=1.585, absorber_absorption=0.05,
              minimal_response=True):
        """
        Run a raytracing simulation.
        
        Args:
            lens_stl: Path to lens STL file
            absorber_stl: Path to absorber STL file
            led_pos: LED position [x, y, z]
            led_dir: LED direction [dx, dy, dz]
            rays: Number of rays to trace
            bounces: Maximum bounces per ray
            max_ray_length: Maximum ray length in mm
            led_power: LED power in mW
            led_angle: LED half-angle in degrees
            wavelength: LED wavelength in nm
            emitter_size: LED emitter size in mm
            lambertian: Use Lambertian emission
            datasheet_directivity: Use datasheet directivity pattern
            led_model: LED model identifier
            refractive_index: Lens refractive index
            absorption: Lens absorption coefficient
            absorber_refractive_index: Absorber refractive index
            absorber_absorption: Absorber absorption coefficient
            minimal_response: If True, return only absorber exit data (faster)
            
        Returns:
            dict with ray_paths, hit_points, and statistics
            (or absorber_exits in minimal mode)
        """
        request = {
            'lens_stl': lens_stl,
            'absorber_stl': absorber_stl,
            'led_pos': list(led_pos) if hasattr(led_pos, '__iter__') else led_pos,
            'led_dir': list(led_dir) if hasattr(led_dir, '__iter__') else led_dir,
            'rays': rays,
            'bounces': bounces,
            'max_ray_length': max_ray_length,
            'led_power': led_power,
            'led_angle': led_angle,
            'wavelength': wavelength,
            'emitter_size': emitter_size,
            'lambertian': lambertian,
            'datasheet_directivity': datasheet_directivity,
            'led_model': led_model,
            'refractive_index': refractive_index,
            'absorption': absorption,
            'absorber_refractive_index': absorber_refractive_index,
            'absorber_absorption': absorber_absorption,
            'minimal_response': minimal_response
        }
        
        # Auto-restart if server died (e.g., after Windows logout/lock)
        # IMPORTANT: Server restart is done OUTSIDE the lock to avoid blocking UI
        max_retries = 2
        last_error = None
        need_restart = False
        
        for attempt in range(max_retries):
            # If previous attempt indicated we need a restart, do it OUTSIDE the lock
            # This prevents the UI from freezing during the 60+ second server startup
            if need_restart:
                need_restart = False
                if self.python_exe:
                    print(f"Attempting server restart (attempt {attempt}/{max_retries})...")
                    try:
                        if not self.ensure_running():
                            raise ConnectionError("Failed to restart raytracer server")
                    except Exception as e:
                        print(f"Server restart failed: {e}")
                        raise ConnectionError(f"Server restart failed: {e}")
            
            # Now try the actual trace operation with the lock held briefly
            with self._lock:
                if not self._connect():
                    # Connection failed - mark for restart on next attempt
                    self._disconnect()
                    if attempt < max_retries - 1 and self.python_exe:
                        print(f"Connection failed (attempt {attempt + 1}/{max_retries})")
                        need_restart = True
                        continue  # Will restart outside lock on next iteration
                    raise ConnectionError("Cannot connect to raytracer server")
                
                try:
                    self._send_message(request)
                    response = self._recv_message()
                    
                    if response is None:
                        # Lost connection mid-request - server may have died
                        self._disconnect()
                        last_error = ConnectionError("Lost connection to server")
                        if attempt < max_retries - 1 and self.python_exe:
                            print(f"Lost connection (attempt {attempt + 1}/{max_retries})")
                            need_restart = True
                            continue  # Will restart outside lock on next iteration
                        raise last_error
                    
                    if not response.get('success', False):
                        error = response.get('error', 'Unknown error')
                        tb = response.get('traceback', '')
                        raise RuntimeError(f"Raytracer error: {error}\n{tb}")
                    
                    return response
                    
                except ConnectionError:
                    self._disconnect()
                    raise
                except Exception as e:
                    self._disconnect()
                    last_error = e
                    if attempt < max_retries - 1:
                        need_restart = True
                        continue
                    raise
        
        if last_error:
            raise last_error
        raise ConnectionError("Failed to communicate with raytracer server")
    
    def _connect(self):
        """Connect to the server."""
        if self.connected and self.socket:
            return True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.request_timeout)  # Configurable timeout (default 120s)
            self.socket.connect(('127.0.0.1', self.port))
            self.connected = True
            return True
        except Exception as e:
            self.socket = None
            self.connected = False
            return False
    
    def _disconnect(self):
        """Disconnect from the server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
    
    def _ping(self):
        """Check if server is responding."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(('127.0.0.1', self.port))
            
            # Send ping command
            request = {'command': 'ping'}
            json_data = json.dumps(request).encode('utf-8')
            sock.sendall(struct.pack('>I', len(json_data)))
            sock.sendall(json_data)
            
            # Receive response
            length_data = self._recv_exactly_sock(sock, 4)
            if length_data:
                length = struct.unpack('>I', length_data)[0]
                json_data = self._recv_exactly_sock(sock, length)
                if json_data:
                    response = json.loads(json_data.decode('utf-8'))
                    sock.close()
                    return response.get('success', False)
            
            sock.close()
            return False
        except:
            return False
    
    def _send_message(self, data):
        """Send a JSON message with length prefix."""
        json_data = json.dumps(data).encode('utf-8')
        length = len(json_data)
        self.socket.sendall(struct.pack('>I', length))
        self.socket.sendall(json_data)
    
    def _recv_message(self):
        """Receive a JSON message with length prefix."""
        length_data = self._recv_exactly(4)
        if not length_data:
            return None
        length = struct.unpack('>I', length_data)[0]
        
        json_data = self._recv_exactly(length)
        if not json_data:
            return None
        
        return json.loads(json_data.decode('utf-8'))
    
    def _recv_exactly(self, n):
        """Receive exactly n bytes from self.socket."""
        return self._recv_exactly_sock(self.socket, n)
    
    def _recv_exactly_sock(self, sock, n):
        """Receive exactly n bytes from a socket."""
        data = b''
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except:
                return None
        return data
    
    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
        return False


# Convenience function for simple usage
def run_raytracer_server_mode(python_exe, **kwargs):
    """
    Run a single raytracing simulation using the server.
    Starts server if not running, runs simulation, returns results.
    Does NOT stop the server (call client.stop_server() when done).
    
    Returns:
        (client, result) tuple - keep client reference to stop server later
    """
    client = RaytracerClient(python_exe=python_exe)
    client.start_server()
    result = client.trace(**kwargs)
    return client, result

