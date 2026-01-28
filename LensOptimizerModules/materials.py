"""
Material properties for the Lens Optimizer.

This module defines material properties for lenses and absorbers,
including refractive indices and absorption coefficients.
"""

# ============================================================================
# LENS MATERIAL PROPERTIES
# ============================================================================

# Lens Material: COP-Zeon ZEONEX® K26R (Cyclo Olefin Polymer)
# Source: ZEON Corporation datasheet 200323391-1.pdf
# Properties:
#   - Refractive index: 1.535 (ASTM D542)
#   - Glass transition temp: 143°C
#   - Excellent UV transmission (>90% at 350nm for 3mm plate)
#   - Very low water absorption: 0.014%
#   - Specific gravity: 1.01
LENS_MATERIAL_NAME = "ZEONEX K26R (COP)"
LENS_REFRACTIVE_INDEX = 1.535
LENS_ABSORPTION_COEFF = 0.001  # Very low UV absorption for COP

# ============================================================================
# ABSORBER MATERIAL PROPERTIES
# ============================================================================

# Absorber Material: Samyang TRIREX® 3020MD (Polycarbonate)
# Source: TRIREX_3020MD.pdf datasheet
# Properties:
#   - Refractive index: ~1.585 (typical for PC)
#   - Specific gravity: 1.20
#   - Higher UV absorption than COP
#   - Good mechanical properties
ABSORBER_MATERIAL_NAME = "TRIREX 3020MD (PC)"
ABSORBER_REFRACTIVE_INDEX = 1.585
ABSORBER_ABSORPTION_COEFF = 0.05  # Moderate absorption for polycarbonate

# ============================================================================
# LEGACY DEFAULTS (for backward compatibility)
# ============================================================================

DEFAULT_REFRACTIVE_INDEX = LENS_REFRACTIVE_INDEX
DEFAULT_ABSORPTION_COEFF = LENS_ABSORPTION_COEFF
