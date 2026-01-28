"""
Data classes for the Lens Optimizer.

This module defines the core data structures used throughout the optimizer:
- ProfileParams: Parameters for cross-section profiles
- Individual: Evolutionary algorithm individual (lens design)
- CenterlinePoint: Point along a curved centerline
"""

from dataclasses import dataclass, field
from typing import List, Optional
import FreeCAD

from .config import (
    MIN_PROFILE_SIDES,
    MAX_PROFILE_SIDES,
    DEFAULT_PROFILE_SIDES
)


@dataclass
class ProfileParams:
    """Parameters for a single cross-section profile in the lofted lens.
    
    Attributes:
        sides: Number of polygon sides (3-32)
        radius: Profile radius in mm (bounded by envelope)
        angle: Rotation angle in degrees (0-360)
        z_position: Position along centerline (0.0 = start, 1.0 = end)
        max_radius: Maximum allowed radius at this Z position (envelope constraint)
    """
    sides: int = 6
    radius: float = 5.0
    angle: float = 0.0
    z_position: float = 0.0
    max_radius: float = 10.0
    
    def __post_init__(self):
        """Validate and clamp parameters to valid ranges."""
        self.sides = max(MIN_PROFILE_SIDES, min(MAX_PROFILE_SIDES, self.sides))
        self.radius = max(0.5, min(self.max_radius, self.radius))
        self.angle = self.angle % 360.0
    
    def clamp_to_envelope(self):
        """Ensure radius doesn't exceed envelope constraint."""
        self.radius = min(self.radius, self.max_radius * 0.95)  # 5% margin
    
    def copy(self) -> 'ProfileParams':
        """Create a copy of this profile."""
        return ProfileParams(
            sides=self.sides,
            radius=self.radius,
            angle=self.angle,
            z_position=self.z_position,
            max_radius=self.max_radius
        )


@dataclass
class Individual:
    """An individual in the evolutionary population.
    
    Represents one lens design with its profile parameters and fitness scores.
    All profiles in an individual share the same number of polygon sides and
    rotation angle to ensure compatible geometry for sweep operations without twisting.
    """
    profiles: List[ProfileParams] = field(default_factory=list)
    shared_sides: int = DEFAULT_PROFILE_SIDES  # All profiles use this side count
    shared_angle: float = 0.0  # All profiles use this rotation angle (prevents twist)
    fitness: float = 0.0
    uniformity: float = 0.0
    efficiency: float = 0.0
    lens_entry_rate: float = 0.0  # Fraction of rays that hit the lens
    absorber_capture_rate: float = 0.0  # Fraction of lens rays that reach absorber
    generation: int = 0
    is_valid: bool = False
    validation_error: str = ""
    
    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        return Individual(
            profiles=[p.copy() for p in self.profiles],
            shared_sides=self.shared_sides,
            shared_angle=self.shared_angle,
            fitness=self.fitness,
            uniformity=self.uniformity,
            efficiency=self.efficiency,
            lens_entry_rate=self.lens_entry_rate,
            absorber_capture_rate=self.absorber_capture_rate,
            generation=self.generation,
            is_valid=self.is_valid,
            validation_error=self.validation_error
        )
    
    def apply_shared_sides(self):
        """Apply the shared_sides value to all profiles."""
        for profile in self.profiles:
            profile.sides = self.shared_sides
    
    def apply_shared_angle(self):
        """Apply the shared_angle value to all profiles (prevents twist in sweeps)."""
        for profile in self.profiles:
            profile.angle = self.shared_angle
    
    def apply_shared_params(self):
        """Apply all shared parameters to profiles."""
        self.apply_shared_sides()
        self.apply_shared_angle()


@dataclass
class CenterlinePoint:
    """A point along the curved centerline with associated data.
    
    Attributes:
        position: 3D position of centerline point (FreeCAD.Vector)
        t: Normalized position along centerline (0.0 = start, 1.0 = end)
        max_radius: Maximum inscribed radius at this point
        tangent: Tangent direction at this point (computed later)
    """
    position: FreeCAD.Vector  # 3D position of centerline point
    t: float  # Normalized position (0.0 to 1.0)
    max_radius: float  # Maximum inscribed radius at this point
    tangent: Optional[FreeCAD.Vector] = None  # Tangent direction (computed later)
