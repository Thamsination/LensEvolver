"""
Dialog functions for the Lens Optimizer.

This module provides the main configuration dialog for the optimizer,
including LED settings (dual LED support), material properties, and
optimization parameters.
"""

from typing import Optional, Dict
import FreeCAD

from .config import (
    COMPUTE_BUDGETS,
    DEFAULT_NUM_PROFILES,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_LED_CURRENT,
    LED_MODELS,
    current_to_radiant_power
)
from .materials import (
    LENS_MATERIAL_NAME,
    LENS_REFRACTIVE_INDEX,
    LENS_ABSORPTION_COEFF,
    ABSORBER_MATERIAL_NAME,
    ABSORBER_REFRACTIVE_INDEX,
    ABSORBER_ABSORPTION_COEFF
)

# Material presets for quick selection
MATERIAL_PRESETS = {
    'lens': {
        'ZEONEX K26R (COP)': {'n': 1.535, 'abs': 0.001, 'desc': 'Cyclo Olefin Polymer - excellent UV transmission'},
        'TOPAS 5013S-04 (COC)': {'n': 1.53, 'abs': 0.002, 'desc': 'TOPAS COC medical grade - n=1.53 (brochure), excellent optical, biocompatible'},
        'Fused Silica': {'n': 1.458, 'abs': 0.0001, 'desc': 'Quartz glass - superior UV transmission'},
        'PMMA (Acrylic)': {'n': 1.49, 'abs': 0.015, 'desc': 'Standard acrylic - good UV transmission'},
        'Polycarbonate': {'n': 1.585, 'abs': 0.05, 'desc': 'PC - high impact, moderate UV absorption'},
        'TRIREX 3020MD (PC)': {'n': 1.585, 'abs': 0.63, 'desc': 'Samyang TRIREX - polycarbonate, standard absorber'},
        'ETFE (Fluoropolymer)': {'n': 1.38, 'abs': 0.002, 'desc': 'Low refractive index - promotes side exit'},
        'Silicone (Optical)': {'n': 1.41, 'abs': 0.003, 'desc': 'Flexible, low n - good for diffuse exit'},
        'Custom': {'n': 1.5, 'abs': 0.01, 'desc': 'User-defined values'},
    },
    'absorber': {
        'TRIREX 3020MD (PC)': {'n': 1.585, 'abs': 0.63, 'desc': 'Polycarbonate - standard absorber'},
        'Black PMMA': {'n': 1.49, 'abs': 0.1, 'desc': 'Black acrylic - high absorption'},
        'Absorbing Silicone': {'n': 1.41, 'abs': 0.08, 'desc': 'Flexible absorber'},
        'Custom': {'n': 1.5, 'abs': 0.05, 'desc': 'User-defined values'},
    }
}


def show_compute_budget_dialog() -> Optional[Dict]:
    """Show dialog for selecting optimization settings.
    
    This creates a comprehensive dialog for configuring:
    - Operation mode (Evolution vs Analysis)
    - Raytracing quality (Quick/Medium/Thorough)
    - Evolution settings (population, generations, profiles)
    - LED 1 settings (model, current, direction)
    - LED 2 settings (optional secondary LED)
    - Material properties (lens and absorber with presets)
    - Lens geometry options (entry spheres, profile shape)
    
    Returns:
        Dict with settings or None if cancelled
    """
    try:
        from PySide2 import QtWidgets, QtCore
    except ImportError:
        from PySide import QtWidgets, QtCore
    
    # Create dialog
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Evolutionary Lens Optimizer")
    dialog.setMinimumWidth(500)
    
    # Create scroll area for the dialog content
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    
    # Container widget for scroll content
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # Title
    title_label = QtWidgets.QLabel("Lens Optimizer / Raytracing Analysis")
    title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
    layout.addWidget(title_label)
    
    layout.addSpacing(10)
    
    # =========== Mode Selection ===========
    mode_label = QtWidgets.QLabel("Operation Mode:")
    mode_label.setStyleSheet("font-weight: bold;")
    layout.addWidget(mode_label)
    
    mode_combo = QtWidgets.QComboBox()
    mode_combo.addItem("Lens Evolution (generate optimized lens)", "evolution")
    mode_combo.addItem("Raytracing Analysis (analyze existing geometry)", "analysis")
    layout.addWidget(mode_combo)
    
    layout.addSpacing(10)
    
    # =========== Budget Selection ===========
    budget_label = QtWidgets.QLabel("Raytracing Quality:")
    layout.addWidget(budget_label)
    
    budget_combo = QtWidgets.QComboBox()
    for name, settings in COMPUTE_BUDGETS.items():
        budget_combo.addItem(f"{name}: {settings['description']}", name)
    layout.addWidget(budget_combo)
    
    # Multi-phase refinement checkbox (only for evolution mode)
    multi_phase_checkbox = QtWidgets.QCheckBox("Enable multi-phase refinement (Quick → Medium → Thorough)")
    multi_phase_checkbox.setChecked(False)
    multi_phase_checkbox.setToolTip(
        "Automatically run 3 phases:\n"
        "• Phase 1 (Quick): Explore design space, vary profile count\n"
        "• Phase 2 (Medium): Refine promising designs\n"
        "• Phase 3 (Thorough): Final polish with high accuracy\n"
        "Estimated total time: 20-40 minutes"
    )
    layout.addWidget(multi_phase_checkbox)
    
    layout.addSpacing(10)
    
    # =========== Evolution Settings (wrapped for show/hide) ===========
    evo_settings_widget = QtWidgets.QWidget()
    evo_settings_layout = QtWidgets.QVBoxLayout(evo_settings_widget)
    evo_settings_layout.setContentsMargins(0, 0, 0, 0)
    
    evo_label = QtWidgets.QLabel("Evolution Settings:")
    evo_label.setStyleSheet("font-weight: bold;")
    evo_settings_layout.addWidget(evo_label)
    
    evo_grid = QtWidgets.QGridLayout()
    
    evo_grid.addWidget(QtWidgets.QLabel("Population Size:"), 0, 0)
    pop_spin = QtWidgets.QSpinBox()
    pop_spin.setMinimum(4)
    pop_spin.setMaximum(30)
    pop_spin.setValue(DEFAULT_POPULATION_SIZE)
    evo_grid.addWidget(pop_spin, 0, 1)
    
    evo_grid.addWidget(QtWidgets.QLabel("Generations:"), 0, 2)
    gen_spin = QtWidgets.QSpinBox()
    gen_spin.setMinimum(1)
    gen_spin.setMaximum(100)
    gen_spin.setValue(DEFAULT_GENERATIONS)
    evo_grid.addWidget(gen_spin, 0, 3)
    
    evo_grid.addWidget(QtWidgets.QLabel("Number of Profiles:"), 1, 0)
    profile_spin = QtWidgets.QSpinBox()
    profile_spin.setMinimum(3)
    profile_spin.setMaximum(15)
    profile_spin.setValue(DEFAULT_NUM_PROFILES)
    evo_grid.addWidget(profile_spin, 1, 1)
    
    evo_settings_layout.addLayout(evo_grid)
    
    layout.addWidget(evo_settings_widget)
    
    # Separator
    line1 = QtWidgets.QFrame()
    line1.setFrameShape(QtWidgets.QFrame.HLine)
    layout.addWidget(line1)
    
    # =========== LED 1 Settings ===========
    led1_label = QtWidgets.QLabel("LED 1 (Primary):")
    led1_label.setStyleSheet("font-weight: bold;")
    layout.addWidget(led1_label)
    
    led1_grid = QtWidgets.QGridLayout()
    
    # LED 1 Model
    led1_grid.addWidget(QtWidgets.QLabel("Model:"), 0, 0)
    led1_model_combo = QtWidgets.QComboBox()
    for model_id, model_data in LED_MODELS.items():
        led1_model_combo.addItem(f"{model_data['name']} ({model_data['wavelength']}nm)", model_id)
    led1_model_combo.setCurrentIndex(3)  # Default to U405
    led1_grid.addWidget(led1_model_combo, 0, 1)
    
    # LED 1 Current
    led1_grid.addWidget(QtWidgets.QLabel("Current (mA):"), 1, 0)
    led1_current_spin = QtWidgets.QSpinBox()
    led1_current_spin.setMinimum(70)
    led1_current_spin.setMaximum(1400)
    led1_current_spin.setValue(int(DEFAULT_LED_CURRENT))
    led1_current_spin.setSingleStep(50)
    led1_grid.addWidget(led1_current_spin, 1, 1)
    
    # LED 1 Power display
    led1_power_label = QtWidgets.QLabel()
    led1_grid.addWidget(led1_power_label, 1, 2)
    
    layout.addLayout(led1_grid)
    
    # LED 1 Direction
    led1_dir_layout = QtWidgets.QHBoxLayout()
    led1_dir_layout.addWidget(QtWidgets.QLabel("Direction:"))
    
    led1_dir_x = QtWidgets.QDoubleSpinBox()
    led1_dir_x.setRange(-1, 1)
    led1_dir_x.setValue(0)
    led1_dir_x.setDecimals(3)
    led1_dir_x.setSingleStep(0.1)
    led1_dir_layout.addWidget(QtWidgets.QLabel("X:"))
    led1_dir_layout.addWidget(led1_dir_x)
    
    led1_dir_y = QtWidgets.QDoubleSpinBox()
    led1_dir_y.setRange(-1, 1)
    led1_dir_y.setValue(1)  # Default pointing up in Y
    led1_dir_y.setDecimals(3)
    led1_dir_y.setSingleStep(0.1)
    led1_dir_layout.addWidget(QtWidgets.QLabel("Y:"))
    led1_dir_layout.addWidget(led1_dir_y)
    
    led1_dir_z = QtWidgets.QDoubleSpinBox()
    led1_dir_z.setRange(-1, 1)
    led1_dir_z.setValue(0)
    led1_dir_z.setDecimals(3)
    led1_dir_z.setSingleStep(0.1)
    led1_dir_layout.addWidget(QtWidgets.QLabel("Z:"))
    led1_dir_layout.addWidget(led1_dir_z)
    
    layout.addLayout(led1_dir_layout)
    
    # Update LED 1 power display
    def update_led1_power():
        model = led1_model_combo.currentData()
        current = led1_current_spin.value()
        power = current_to_radiant_power(current, model)
        led1_power_label.setText(f"→ {power:.0f} mW")
    
    led1_current_spin.valueChanged.connect(update_led1_power)
    led1_model_combo.currentIndexChanged.connect(update_led1_power)
    update_led1_power()
    
    layout.addSpacing(10)
    
    # =========== LED 2 Settings ===========
    led2_label = QtWidgets.QLabel("LED 2 (Secondary):")
    led2_label.setStyleSheet("font-weight: bold;")
    layout.addWidget(led2_label)
    
    led2_grid = QtWidgets.QGridLayout()
    
    # LED 2 Model (includes Disabled option)
    led2_grid.addWidget(QtWidgets.QLabel("Model:"), 0, 0)
    led2_model_combo = QtWidgets.QComboBox()
    led2_model_combo.addItem("Disabled", None)
    for model_id, model_data in LED_MODELS.items():
        led2_model_combo.addItem(f"{model_data['name']} ({model_data['wavelength']}nm)", model_id)
    led2_model_combo.setCurrentIndex(0)  # Default to Disabled
    led2_grid.addWidget(led2_model_combo, 0, 1)
    
    # LED 2 Current
    led2_grid.addWidget(QtWidgets.QLabel("Current (mA):"), 1, 0)
    led2_current_spin = QtWidgets.QSpinBox()
    led2_current_spin.setMinimum(70)
    led2_current_spin.setMaximum(1400)
    led2_current_spin.setValue(int(DEFAULT_LED_CURRENT))
    led2_current_spin.setSingleStep(50)
    led2_current_spin.setEnabled(False)  # Disabled by default
    led2_grid.addWidget(led2_current_spin, 1, 1)
    
    # LED 2 Power display
    led2_power_label = QtWidgets.QLabel("→ 0 mW")
    led2_grid.addWidget(led2_power_label, 1, 2)
    
    layout.addLayout(led2_grid)
    
    # LED 2 Position and Direction (only shown when enabled)
    led2_pos_widget = QtWidgets.QWidget()
    led2_pos_layout = QtWidgets.QGridLayout(led2_pos_widget)
    led2_pos_layout.setContentsMargins(0, 0, 0, 0)
    
    # LED 2 Position object selector
    led2_pos_layout.addWidget(QtWidgets.QLabel("Position Object:"), 0, 0)
    led2_obj_combo = QtWidgets.QComboBox()
    led2_obj_combo.addItem("Select LED 2 object...", None)
    
    # Populate with objects from the document
    doc = FreeCAD.ActiveDocument
    if doc is not None:
        for obj in doc.Objects:
            try:
                if hasattr(obj, 'Shape') and obj.Shape is not None:
                    shape_type = obj.Shape.ShapeType if hasattr(obj.Shape, 'ShapeType') else ""
                    if shape_type in ['Solid', 'Shell', 'Compound', 'Vertex'] or hasattr(obj.Shape, 'CenterOfMass'):
                        led2_obj_combo.addItem(f"{obj.Label} ({shape_type})", obj.Name)
            except:
                pass
    
    led2_obj_combo.setToolTip("Select a geometry object (sphere) to use as LED 2 position")
    led2_pos_layout.addWidget(led2_obj_combo, 0, 1, 1, 6)
    
    # LED 2 Direction
    led2_pos_layout.addWidget(QtWidgets.QLabel("Direction:"), 1, 0)
    led2_dir_x = QtWidgets.QDoubleSpinBox()
    led2_dir_x.setRange(-1, 1)
    led2_dir_x.setValue(0)
    led2_dir_x.setDecimals(3)
    led2_dir_x.setSingleStep(0.1)
    led2_pos_layout.addWidget(QtWidgets.QLabel("X:"), 1, 1)
    led2_pos_layout.addWidget(led2_dir_x, 1, 2)
    
    led2_dir_y = QtWidgets.QDoubleSpinBox()
    led2_dir_y.setRange(-1, 1)
    led2_dir_y.setValue(1)  # Default pointing up in Y
    led2_dir_y.setDecimals(3)
    led2_dir_y.setSingleStep(0.1)
    led2_pos_layout.addWidget(QtWidgets.QLabel("Y:"), 1, 3)
    led2_pos_layout.addWidget(led2_dir_y, 1, 4)
    
    led2_dir_z = QtWidgets.QDoubleSpinBox()
    led2_dir_z.setRange(-1, 1)
    led2_dir_z.setValue(0)
    led2_dir_z.setDecimals(3)
    led2_dir_z.setSingleStep(0.1)
    led2_pos_layout.addWidget(QtWidgets.QLabel("Z:"), 1, 5)
    led2_pos_layout.addWidget(led2_dir_z, 1, 6)
    
    led2_pos_widget.setVisible(False)  # Hidden by default
    layout.addWidget(led2_pos_widget)
    
    # Update LED 2 UI when model changes
    def update_led2_ui():
        enabled = led2_model_combo.currentData() is not None
        led2_current_spin.setEnabled(enabled)
        led2_pos_widget.setVisible(enabled)
        if enabled:
            model = led2_model_combo.currentData()
            current = led2_current_spin.value()
            power = current_to_radiant_power(current, model)
            led2_power_label.setText(f"→ {power:.0f} mW")
        else:
            led2_power_label.setText("→ 0 mW")
    
    def update_led2_power():
        if led2_model_combo.currentData() is not None:
            model = led2_model_combo.currentData()
            current = led2_current_spin.value()
            power = current_to_radiant_power(current, model)
            led2_power_label.setText(f"→ {power:.0f} mW")
    
    led2_model_combo.currentIndexChanged.connect(update_led2_ui)
    led2_current_spin.valueChanged.connect(update_led2_power)
    update_led2_ui()
    
    layout.addSpacing(10)
    
    # Separator
    line2 = QtWidgets.QFrame()
    line2.setFrameShape(QtWidgets.QFrame.HLine)
    layout.addWidget(line2)
    
    # =========== Material Properties ===========
    mat_label = QtWidgets.QLabel("Material Properties:")
    mat_label.setStyleSheet("font-weight: bold;")
    layout.addWidget(mat_label)
    
    mat_grid = QtWidgets.QGridLayout()
    
    # --- Lens Material ---
    mat_grid.addWidget(QtWidgets.QLabel("Lens Material:"), 0, 0)
    lens_mat_combo = QtWidgets.QComboBox()
    for name in MATERIAL_PRESETS['lens'].keys():
        lens_mat_combo.addItem(name, name)
    lens_mat_combo.setCurrentIndex(0)  # Default to ZEONEX K26R
    lens_mat_combo.setToolTip("Select lens material preset or choose Custom")
    mat_grid.addWidget(lens_mat_combo, 0, 1, 1, 3)
    
    mat_grid.addWidget(QtWidgets.QLabel("n:"), 1, 0)
    lens_n_spin = QtWidgets.QDoubleSpinBox()
    lens_n_spin.setRange(1.0, 2.5)
    lens_n_spin.setValue(LENS_REFRACTIVE_INDEX)
    lens_n_spin.setDecimals(3)
    lens_n_spin.setSingleStep(0.001)
    lens_n_spin.setToolTip("Lens refractive index")
    mat_grid.addWidget(lens_n_spin, 1, 1)
    
    mat_grid.addWidget(QtWidgets.QLabel("α (/mm):"), 1, 2)
    lens_abs_spin = QtWidgets.QDoubleSpinBox()
    lens_abs_spin.setRange(0.0, 1.0)
    lens_abs_spin.setValue(LENS_ABSORPTION_COEFF)
    lens_abs_spin.setDecimals(4)
    lens_abs_spin.setSingleStep(0.001)
    lens_abs_spin.setToolTip("Lens absorption coefficient per mm")
    mat_grid.addWidget(lens_abs_spin, 1, 3)
    
    # --- Absorber Material --- (same presets as lens for consistency)
    mat_grid.addWidget(QtWidgets.QLabel("Absorber Material:"), 2, 0)
    absorber_mat_combo = QtWidgets.QComboBox()
    for name in MATERIAL_PRESETS['lens'].keys():
        absorber_mat_combo.addItem(name, name)
    absorber_mat_combo.setCurrentIndex(0)  # Default to ZEONEX K26R (same as lens)
    absorber_mat_combo.setToolTip("Select absorber material preset (same options as lens) or choose Custom")
    mat_grid.addWidget(absorber_mat_combo, 2, 1, 1, 3)
    
    mat_grid.addWidget(QtWidgets.QLabel("n:"), 3, 0)
    absorber_n_spin = QtWidgets.QDoubleSpinBox()
    absorber_n_spin.setRange(1.0, 2.5)
    absorber_n_spin.setValue(ABSORBER_REFRACTIVE_INDEX)
    absorber_n_spin.setDecimals(3)
    absorber_n_spin.setSingleStep(0.001)
    absorber_n_spin.setToolTip("Absorber refractive index")
    mat_grid.addWidget(absorber_n_spin, 3, 1)
    
    mat_grid.addWidget(QtWidgets.QLabel("α (/mm):"), 3, 2)
    absorber_abs_spin = QtWidgets.QDoubleSpinBox()
    absorber_abs_spin.setRange(0.0, 1.0)
    absorber_abs_spin.setValue(ABSORBER_ABSORPTION_COEFF)
    absorber_abs_spin.setDecimals(4)
    absorber_abs_spin.setSingleStep(0.01)
    absorber_abs_spin.setToolTip("Absorber absorption coefficient per mm")
    mat_grid.addWidget(absorber_abs_spin, 3, 3)
    
    layout.addLayout(mat_grid)
    
    # Update material values when preset changes
    def update_lens_material():
        mat_name = lens_mat_combo.currentData()
        if mat_name and mat_name != 'Custom':
            props = MATERIAL_PRESETS['lens'][mat_name]
            lens_n_spin.setValue(props['n'])
            lens_abs_spin.setValue(props['abs'])
    
    def update_absorber_material():
        mat_name = absorber_mat_combo.currentData()
        if mat_name and mat_name != 'Custom':
            props = MATERIAL_PRESETS['lens'][mat_name]
            absorber_n_spin.setValue(props['n'])
            absorber_abs_spin.setValue(props['abs'])
    
    lens_mat_combo.currentIndexChanged.connect(update_lens_material)
    absorber_mat_combo.currentIndexChanged.connect(update_absorber_material)
    
    # Material info
    mat_info = QtWidgets.QLabel("n = refractive index, α = absorption coefficient (/mm)")
    mat_info.setStyleSheet("color: gray; font-size: 9px;")
    layout.addWidget(mat_info)
    
    layout.addSpacing(10)
    
    # =========== Geometry Options (wrapped for show/hide) ===========
    geo_options_widget = QtWidgets.QWidget()
    geo_options_layout = QtWidgets.QVBoxLayout(geo_options_widget)
    geo_options_layout.setContentsMargins(0, 0, 0, 0)
    
    # Separator
    line3 = QtWidgets.QFrame()
    line3.setFrameShape(QtWidgets.QFrame.HLine)
    geo_options_layout.addWidget(line3)
    
    geo_label = QtWidgets.QLabel("Geometry Options:")
    geo_label.setStyleSheet("font-weight: bold;")
    geo_options_layout.addWidget(geo_label)
    
    entry_spheres_cb = QtWidgets.QCheckBox("Enable spherical LED entry surfaces")
    entry_spheres_cb.setChecked(True)
    entry_spheres_cb.setToolTip(
        "Cut spherical surfaces at LED positions to create proper lens entry.\n"
        "This replaces flat bottom surfaces with curved refractive surfaces."
    )
    geo_options_layout.addWidget(entry_spheres_cb)
    
    circle_profile_cb = QtWidgets.QCheckBox("Use circular profiles (vs polygon)")
    circle_profile_cb.setChecked(False)
    geo_options_layout.addWidget(circle_profile_cb)
    
    vary_profile_cb = QtWidgets.QCheckBox("Allow variable profile count during evolution")
    vary_profile_cb.setChecked(False)
    geo_options_layout.addWidget(vary_profile_cb)
    
    layout.addWidget(geo_options_widget)
    
    layout.addStretch()
    
    # Set scroll content
    scroll.setWidget(container)
    
    # Main dialog layout
    main_layout = QtWidgets.QVBoxLayout(dialog)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.addWidget(scroll)
    
    # Buttons at bottom (outside scroll area)
    button_widget = QtWidgets.QWidget()
    button_layout = QtWidgets.QHBoxLayout(button_widget)
    button_layout.setContentsMargins(10, 5, 10, 10)
    
    ok_button = QtWidgets.QPushButton("Start")
    cancel_button = QtWidgets.QPushButton("Cancel")
    
    ok_button.clicked.connect(dialog.accept)
    cancel_button.clicked.connect(dialog.reject)
    
    button_layout.addStretch()
    button_layout.addWidget(ok_button)
    button_layout.addWidget(cancel_button)
    
    main_layout.addWidget(button_widget)
    
    # Set dialog size
    dialog.resize(520, 650)
    
    # =========== Mode Change Handler ===========
    # Hide evolution-only fields when Analysis mode is selected
    def update_mode_ui():
        is_evolution = mode_combo.currentData() == 'evolution'
        multi_phase_checkbox.setVisible(is_evolution)
        evo_settings_widget.setVisible(is_evolution)
        geo_options_widget.setVisible(is_evolution)
    
    mode_combo.currentIndexChanged.connect(update_mode_ui)
    update_mode_ui()  # Set initial state based on default selection
    
    # Show dialog and return results
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        led1_model = led1_model_combo.currentData()
        led1_current = led1_current_spin.value()
        
        # LED 2 config (None if disabled)
        led2_model = led2_model_combo.currentData()
        led2_config = None
        if led2_model is not None:
            led2_current = led2_current_spin.value()
            led2_power = current_to_radiant_power(led2_current, led2_model)
            led2_obj_name = led2_obj_combo.currentData()
            if led2_obj_name is None:
                FreeCAD.Console.PrintWarning("LED 2 enabled but no position object selected. LED 2 will be disabled.\n")
            else:
                led2_config = {
                    'model': led2_model,
                    'current_mA': led2_current,
                    'power_mW': led2_power,
                    'wavelength': LED_MODELS[led2_model]['wavelength'],
                    'position_object': led2_obj_name,
                    'direction': (led2_dir_x.value(), led2_dir_y.value(), led2_dir_z.value()),
                }
        
        return {
            'mode': mode_combo.currentData(),
            'budget_name': budget_combo.currentData(),
            'population_size': pop_spin.value(),
            'generations': gen_spin.value(),
            'num_profiles': profile_spin.value(),
            # LED 1
            'led_model': led1_model,
            'led_current_mA': led1_current,
            'led_power_mW': current_to_radiant_power(led1_current, led1_model),
            'led_wavelength': LED_MODELS[led1_model]['wavelength'],
            'led1_direction': (led1_dir_x.value(), led1_dir_y.value(), led1_dir_z.value()),
            # LED 2
            'led2_config': led2_config,
            # Materials
            'lens_material': {
                'name': lens_mat_combo.currentData(),
                'refractive_index': lens_n_spin.value(),
                'absorption_coeff': lens_abs_spin.value(),
            },
            'absorber_material': {
                'name': absorber_mat_combo.currentData(),
                'refractive_index': absorber_n_spin.value(),
                'absorption_coeff': absorber_abs_spin.value(),
            },
            # Geometry options
            'enable_entry_spheres': entry_spheres_cb.isChecked(),
            'entry_sphere_depth': 0.3,
            'use_circle_profile': circle_profile_cb.isChecked(),
            'vary_profile_count': vary_profile_cb.isChecked(),
            'multi_phase_refinement': multi_phase_checkbox.isChecked(),
            'distribution_weight': 0.0,
            'envelope_reduction': 0.0
        }
    else:
        return None
