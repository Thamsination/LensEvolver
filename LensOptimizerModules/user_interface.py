"""
User interface functions for the Lens Optimizer.

This module provides functions for object selection, LED position/direction
input, and basic UI utilities.
"""

import math
from typing import Optional, Tuple
import FreeCAD
import FreeCADGui


def get_selected_objects():
    """Get currently selected objects in FreeCAD.
    
    Selection order:
    1. LED Source (for position)
    2. Envelope Solid (max volume constraint)
    3. Absorber Geometry (target surface)
    4. Centerline Sketch (containing BSpline curve) - OPTIONAL
    
    If only 3 objects are selected, centerline will be auto-extracted from envelope.
    
    Returns:
        Tuple of (led_obj, envelope_obj, absorber_obj, centerline_obj)
        or (None, None, None, None) on error
    """
    selection = FreeCADGui.Selection.getSelection()
    
    if len(selection) < 3:
        FreeCAD.Console.PrintError("Please select 3 or 4 objects in this order:\n")
        FreeCAD.Console.PrintMessage("  1. LED Source (point/sphere for position)\n")
        FreeCAD.Console.PrintMessage("  2. Envelope Solid (maximum lens volume)\n")
        FreeCAD.Console.PrintMessage("  3. Absorber Geometry (target surface)\n")
        FreeCAD.Console.PrintMessage("  4. Centerline Sketch (OPTIONAL - auto-extracted if not provided)\n")
        return None, None, None, None
    
    led_obj = selection[0]
    envelope_obj = selection[1]
    absorber_obj = selection[2]
    centerline_obj = selection[3] if len(selection) >= 4 else None
    
    # Validate envelope has a solid shape
    if not hasattr(envelope_obj, 'Shape') or not envelope_obj.Shape.Solids:
        FreeCAD.Console.PrintError(f"Envelope '{envelope_obj.Label}' must be a solid object!\n")
        return None, None, None, None
    
    # Validate absorber has a shape
    if not hasattr(absorber_obj, 'Shape'):
        FreeCAD.Console.PrintError(f"Absorber '{absorber_obj.Label}' must have geometry!\n")
        return None, None, None, None
    
    # Validate centerline if provided
    if centerline_obj is not None:
        if not hasattr(centerline_obj, 'Shape') or not centerline_obj.Shape.Edges:
            FreeCAD.Console.PrintError(f"Centerline '{centerline_obj.Label}' must be a Sketch with edges!\n")
            return None, None, None, None
    
    FreeCAD.Console.PrintMessage(f"Selected objects:\n")
    FreeCAD.Console.PrintMessage(f"  LED:        {led_obj.Label}\n")
    FreeCAD.Console.PrintMessage(f"  Envelope:   {envelope_obj.Label}\n")
    FreeCAD.Console.PrintMessage(f"  Absorber:   {absorber_obj.Label}\n")
    if centerline_obj:
        FreeCAD.Console.PrintMessage(f"  Centerline: {centerline_obj.Label}\n")
    else:
        FreeCAD.Console.PrintMessage(f"  Centerline: (auto-extract from envelope)\n")
    
    return led_obj, envelope_obj, absorber_obj, centerline_obj


def get_led_position(led_obj) -> FreeCAD.Vector:
    """Extract LED position from object.
    
    Args:
        led_obj: FreeCAD object representing the LED
        
    Returns:
        FreeCAD.Vector position of the LED
    """
    # Try CenterOfMass for spheres/solids
    if hasattr(led_obj, 'Shape'):
        if hasattr(led_obj.Shape, 'CenterOfMass'):
            return led_obj.Shape.CenterOfMass
        if led_obj.Shape.Vertexes:
            return led_obj.Shape.Vertexes[0].Point
    
    # Try bounding box center
    if hasattr(led_obj, 'Shape') and hasattr(led_obj.Shape, 'BoundBox'):
        bb = led_obj.Shape.BoundBox
        return FreeCAD.Vector(
            (bb.XMin + bb.XMax) / 2,
            (bb.YMin + bb.YMax) / 2,
            (bb.ZMin + bb.ZMax) / 2
        )
    
    FreeCAD.Console.PrintWarning("Could not determine LED position, using origin\n")
    return FreeCAD.Vector(0, 0, 0)


def get_led_direction(led_obj, envelope_obj) -> FreeCAD.Vector:
    """Determine LED emission direction (toward envelope center).
    
    DEPRECATED: This auto-calculation is kept for backward compatibility.
    Use get_led_direction_from_user() for manual input instead.
    
    Args:
        led_obj: FreeCAD object representing the LED
        envelope_obj: FreeCAD envelope solid object
        
    Returns:
        Normalized FreeCAD.Vector direction from LED to envelope center
    """
    led_pos = get_led_position(led_obj)
    
    # Direction toward envelope center
    if hasattr(envelope_obj, 'Shape') and hasattr(envelope_obj.Shape, 'CenterOfMass'):
        envelope_center = envelope_obj.Shape.CenterOfMass
    else:
        bb = envelope_obj.Shape.BoundBox
        envelope_center = FreeCAD.Vector(
            (bb.XMin + bb.XMax) / 2,
            (bb.YMin + bb.YMax) / 2,
            (bb.ZMin + bb.ZMax) / 2
        )
    
    direction = envelope_center - led_pos
    if direction.Length > 0:
        direction.normalize()
    else:
        direction = FreeCAD.Vector(0, 0, 1)  # Default to +Z
    
    return direction


def get_led_direction_from_user(default_x: float = 0.0, 
                                 default_y: float = 1.0, 
                                 default_z: float = 0.0) -> Optional[FreeCAD.Vector]:
    """Show dialog for user to manually input LED direction vector.
    
    Args:
        default_x: Default X component (default: 0)
        default_y: Default Y component (default: 1 for +Y axis)
        default_z: Default Z component (default: 0)
        
    Returns:
        Normalized FreeCAD.Vector representing LED direction, or None if cancelled
    """
    try:
        from PySide2 import QtWidgets, QtCore
    except ImportError:
        from PySide import QtWidgets, QtCore
    
    # Create dialog
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("LED Direction")
    dialog.setMinimumWidth(300)
    
    layout = QtWidgets.QVBoxLayout(dialog)
    
    # Title
    title_label = QtWidgets.QLabel("Set LED Emission Direction")
    title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
    layout.addWidget(title_label)
    
    # Info text
    info_label = QtWidgets.QLabel(
        "Enter the direction vector for LED emission.\n"
        "The vector will be normalized automatically."
    )
    info_label.setStyleSheet("color: gray;")
    layout.addWidget(info_label)
    
    layout.addSpacing(10)
    
    # Grid layout for X, Y, Z inputs
    grid = QtWidgets.QGridLayout()
    
    # X component
    grid.addWidget(QtWidgets.QLabel("X:"), 0, 0)
    x_spin = QtWidgets.QDoubleSpinBox()
    x_spin.setRange(-1.0, 1.0)
    x_spin.setDecimals(3)
    x_spin.setSingleStep(0.1)
    x_spin.setValue(default_x)
    grid.addWidget(x_spin, 0, 1)
    
    # Y component
    grid.addWidget(QtWidgets.QLabel("Y:"), 1, 0)
    y_spin = QtWidgets.QDoubleSpinBox()
    y_spin.setRange(-1.0, 1.0)
    y_spin.setDecimals(3)
    y_spin.setSingleStep(0.1)
    y_spin.setValue(default_y)
    grid.addWidget(y_spin, 1, 1)
    
    # Z component
    grid.addWidget(QtWidgets.QLabel("Z:"), 2, 0)
    z_spin = QtWidgets.QDoubleSpinBox()
    z_spin.setRange(-1.0, 1.0)
    z_spin.setDecimals(3)
    z_spin.setSingleStep(0.1)
    z_spin.setValue(default_z)
    grid.addWidget(z_spin, 2, 1)
    
    layout.addLayout(grid)
    
    layout.addSpacing(10)
    
    # Preview label showing normalized vector
    preview_label = QtWidgets.QLabel()
    def update_preview():
        x, y, z = x_spin.value(), y_spin.value(), z_spin.value()
        length = math.sqrt(x*x + y*y + z*z)
        if length > 0.001:
            nx, ny, nz = x/length, y/length, z/length
            preview_label.setText(f"Normalized: ({nx:.3f}, {ny:.3f}, {nz:.3f})")
            preview_label.setStyleSheet("color: green;")
        else:
            preview_label.setText("Warning: Zero-length vector!")
            preview_label.setStyleSheet("color: red;")
    
    x_spin.valueChanged.connect(update_preview)
    y_spin.valueChanged.connect(update_preview)
    z_spin.valueChanged.connect(update_preview)
    update_preview()
    layout.addWidget(preview_label)
    
    layout.addSpacing(10)
    
    # Preset buttons
    preset_layout = QtWidgets.QHBoxLayout()
    preset_label = QtWidgets.QLabel("Presets:")
    preset_layout.addWidget(preset_label)
    
    def set_preset(x, y, z):
        x_spin.setValue(x)
        y_spin.setValue(y)
        z_spin.setValue(z)
    
    btn_pos_x = QtWidgets.QPushButton("+X")
    btn_pos_x.clicked.connect(lambda: set_preset(1, 0, 0))
    preset_layout.addWidget(btn_pos_x)
    
    btn_pos_y = QtWidgets.QPushButton("+Y")
    btn_pos_y.clicked.connect(lambda: set_preset(0, 1, 0))
    preset_layout.addWidget(btn_pos_y)
    
    btn_pos_z = QtWidgets.QPushButton("+Z")
    btn_pos_z.clicked.connect(lambda: set_preset(0, 0, 1))
    preset_layout.addWidget(btn_pos_z)
    
    btn_neg_x = QtWidgets.QPushButton("-X")
    btn_neg_x.clicked.connect(lambda: set_preset(-1, 0, 0))
    preset_layout.addWidget(btn_neg_x)
    
    btn_neg_y = QtWidgets.QPushButton("-Y")
    btn_neg_y.clicked.connect(lambda: set_preset(0, -1, 0))
    preset_layout.addWidget(btn_neg_y)
    
    btn_neg_z = QtWidgets.QPushButton("-Z")
    btn_neg_z.clicked.connect(lambda: set_preset(0, 0, -1))
    preset_layout.addWidget(btn_neg_z)
    
    layout.addLayout(preset_layout)
    
    layout.addSpacing(15)
    
    # Buttons
    button_layout = QtWidgets.QHBoxLayout()
    ok_button = QtWidgets.QPushButton("OK")
    cancel_button = QtWidgets.QPushButton("Cancel")
    
    ok_button.clicked.connect(dialog.accept)
    cancel_button.clicked.connect(dialog.reject)
    
    button_layout.addWidget(ok_button)
    button_layout.addWidget(cancel_button)
    layout.addLayout(button_layout)
    
    # Show dialog
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        x, y, z = x_spin.value(), y_spin.value(), z_spin.value()
        direction = FreeCAD.Vector(x, y, z)
        
        # Normalize
        if direction.Length > 0.001:
            direction.normalize()
        else:
            FreeCAD.Console.PrintWarning("Zero-length direction, defaulting to +Y\n")
            direction = FreeCAD.Vector(0, 1, 0)
        
        return direction
    else:
        return None


def show_progress_dialog(title="Optimizing Lens"):
    """Create a progress dialog (non-blocking).
    
    Args:
        title: Dialog title text
        
    Returns:
        QProgressDialog instance
    """
    try:
        from PySide2 import QtWidgets, QtCore
    except ImportError:
        from PySide import QtWidgets, QtCore
    
    progress = QtWidgets.QProgressDialog(title, "Cancel", 0, 100)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.setMinimumDuration(0)
    
    return progress
