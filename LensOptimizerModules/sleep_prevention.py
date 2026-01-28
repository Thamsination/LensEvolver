"""
Sleep prevention utilities for Windows.

Prevents Windows from sleeping during long simulations while allowing
the display to turn off to save power.
"""

import ctypes
import FreeCAD

# Windows SetThreadExecutionState flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
# Note: ES_DISPLAY_REQUIRED (0x00000002) is NOT included to allow screen off

_sleep_prevented = False


def prevent_sleep():
    """Prevent Windows from sleeping during simulation.
    
    The system will stay awake but the display can still turn off.
    Call allow_sleep() when simulation is complete.
    """
    global _sleep_prevented
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        _sleep_prevented = True
        FreeCAD.Console.PrintMessage("Sleep prevention enabled (screen can still turn off)\n")
    except Exception as e:
        FreeCAD.Console.PrintWarning(f"Could not prevent sleep: {e}\n")


def allow_sleep():
    """Allow Windows to sleep again after simulation completes."""
    global _sleep_prevented
    if _sleep_prevented:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            _sleep_prevented = False
            FreeCAD.Console.PrintMessage("Sleep prevention disabled\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not restore sleep: {e}\n")


def keep_ui_responsive():
    """Process Qt events to keep FreeCAD UI responsive during long operations.
    
    Call this periodically during long-running operations to prevent
    FreeCAD from appearing frozen. This allows the UI to update and
    respond to user interactions (like cancel buttons).
    """
    try:
        from PySide2 import QtWidgets
    except ImportError:
        from PySide import QtWidgets
    
    # Process pending events to keep UI responsive
    QtWidgets.QApplication.processEvents()
