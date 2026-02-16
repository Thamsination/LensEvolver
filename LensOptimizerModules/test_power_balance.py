"""
Unit tests for power balance verification in LensEvolver.

Run with: python -m pytest test_power_balance.py -v
Or standalone: python test_power_balance.py
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uniformity_analysis import analyze_uniformity_grid, calculate_probe_irradiance


def test_power_balance_uniform_distribution():
    """Test that power budget is consistent for uniform distribution."""
    # Simulate exit rays with known properties
    num_rays = 10000
    num_exits = 8000  # 80% efficiency
    led_power_mW = 1000.0  # Easy number for calculations
    
    # Create uniform distribution of exit positions in 10x10 mm area
    np.random.seed(42)
    exit_positions = np.random.uniform(-5, 5, (num_exits, 3))
    exit_positions[:, 2] = 0  # All on same Z plane
    
    # Each ray carries led_power (per raytracer convention)
    exit_intensities = np.ones(num_exits) * led_power_mW
    
    # Analyze
    result = analyze_uniformity_grid(
        exit_positions, exit_intensities,
        grid_size=10, num_rays=num_rays
    )
    
    # Check power budget
    integrated_power_raw = result['integrated_power_raw']
    implied_power_mW = integrated_power_raw / num_rays
    
    # Expected: efficiency * led_power = 0.8 * 1000 = 800 mW
    efficiency = num_exits / num_rays
    expected_power_mW = efficiency * led_power_mW
    
    print(f"Implied power: {implied_power_mW:.2f} mW")
    print(f"Expected power: {expected_power_mW:.2f} mW")
    print(f"Efficiency: {efficiency:.2%}")
    
    # They should match exactly for this test case (no absorption)
    assert abs(implied_power_mW - expected_power_mW) < 0.01, \
        f"Power budget mismatch: {implied_power_mW:.2f} vs {expected_power_mW:.2f}"
    
    print("PASS: Power budget is consistent")


def test_power_balance_with_absorption():
    """Test power budget with simulated absorption losses."""
    num_rays = 10000
    num_exits = 9000
    led_power_mW = 1420.0  # Typical U405 power
    
    # Create exit positions
    np.random.seed(123)
    exit_positions = np.random.uniform(-5, 5, (num_exits, 3))
    exit_positions[:, 2] = 0
    
    # Simulate absorption: rays lose some power (random 0.8-1.0 factor)
    attenuation = np.random.uniform(0.8, 1.0, num_exits)
    exit_intensities = np.ones(num_exits) * led_power_mW * attenuation
    
    result = analyze_uniformity_grid(
        exit_positions, exit_intensities,
        grid_size=10, num_rays=num_rays
    )
    
    integrated_power_raw = result['integrated_power_raw']
    implied_power_mW = integrated_power_raw / num_rays
    
    # Expected: sum of attenuated intensities / num_rays
    expected_power_mW = np.sum(exit_intensities) / num_rays
    
    print(f"Implied power: {implied_power_mW:.2f} mW")
    print(f"Expected power: {expected_power_mW:.2f} mW")
    
    assert abs(implied_power_mW - expected_power_mW) < 0.01, \
        f"Power budget mismatch: {implied_power_mW:.2f} vs {expected_power_mW:.2f}"
    
    print("PASS: Power budget with absorption is consistent")


def test_probe_irradiance_calculation():
    """Test probe-equivalent irradiance calculation."""
    num_rays = 10000
    led_power_mW = 1000.0
    
    # Create rays concentrated at center (0, 0)
    np.random.seed(456)
    num_exits = 5000
    # Most rays within 0.5mm of center
    exit_positions = np.random.normal(0, 0.3, (num_exits, 3))
    exit_positions[:, 2] = 0
    exit_intensities = np.ones(num_exits) * led_power_mW
    
    # Calculate probe irradiance at center with 0.4mm diameter probe
    probe_result = calculate_probe_irradiance(
        exit_positions, exit_intensities,
        center_x=0, center_y=0,
        probe_diameter_mm=0.4,
        num_rays=num_rays
    )
    
    print(f"Probe irradiance: {probe_result['irradiance_mW_cm2']:.2f} mW/cm²")
    print(f"Rays in probe: {probe_result['num_rays_in_probe']}")
    print(f"Probe area: {probe_result['probe_area_cm2']:.6f} cm²")
    
    # Verify probe area calculation (pi * 0.2^2 / 100 = 0.001257 cm²)
    expected_area = np.pi * 0.2**2 / 100.0
    assert abs(probe_result['probe_area_cm2'] - expected_area) < 1e-6, \
        f"Probe area mismatch: {probe_result['probe_area_cm2']:.6f} vs {expected_area:.6f}"
    
    # Irradiance should be positive and reasonable
    assert probe_result['irradiance_mW_cm2'] > 0, "Irradiance should be positive"
    assert probe_result['num_rays_in_probe'] > 0, "Should have rays in probe"
    
    print("PASS: Probe irradiance calculation works")


def test_empty_input():
    """Test handling of empty input."""
    result = analyze_uniformity_grid(
        np.array([]).reshape(0, 3),
        np.array([]),
        grid_size=10,
        num_rays=1000
    )
    
    assert result['mean_intensity'] == 0
    assert result['integrated_power_raw'] == 0
    assert result['num_exits'] == 0
    
    probe_result = calculate_probe_irradiance(
        np.array([]).reshape(0, 3),
        np.array([]),
        center_x=0, center_y=0
    )
    
    assert probe_result['irradiance_mW_cm2'] == 0
    assert probe_result['num_rays_in_probe'] == 0
    
    print("PASS: Empty input handled correctly")


def test_irradiance_units():
    """Verify irradiance units are correct (mW/cm²)."""
    num_rays = 1000
    led_power_mW = 1000.0
    
    # Create a 10x10 mm area with 1000 rays, uniform distribution
    # Each ray carries 1000 mW, so total "raw power" = 1000 * 1000 = 1e6
    # After normalization by num_rays: total power = 1000 mW
    # Area = 100 mm² = 1 cm²
    # Mean irradiance should be ~1000 mW/cm² (approximately, with grid effects)
    
    np.random.seed(789)
    num_exits = 1000  # 100% efficiency
    exit_positions = np.random.uniform(-5, 5, (num_exits, 3))
    exit_positions[:, 2] = 0
    exit_intensities = np.ones(num_exits) * led_power_mW
    
    result = analyze_uniformity_grid(
        exit_positions, exit_intensities,
        grid_size=10, num_rays=num_rays
    )
    
    total_area_cm2 = result['total_area_cm2']
    mean_irradiance = result['mean_intensity']
    implied_power = result['integrated_power_raw'] / num_rays
    
    print(f"Total area: {total_area_cm2:.4f} cm²")
    print(f"Mean irradiance: {mean_irradiance:.2f} mW/cm²")
    print(f"Implied power: {implied_power:.2f} mW")
    
    # Check that mean_irradiance * area ≈ implied_power
    # Note: mean is over non-empty cells, so this is approximate
    # The relationship should be within the right order of magnitude
    estimated_power_from_irradiance = mean_irradiance * total_area_cm2
    
    print(f"Estimated power from irradiance: {estimated_power_from_irradiance:.2f} mW")
    print(f"Ratio: {estimated_power_from_irradiance / implied_power:.2f}")
    
    # Should be in the same order of magnitude (within 2x for grid effects)
    assert 0.5 < estimated_power_from_irradiance / implied_power < 2.0, \
        "Irradiance units seem incorrect"
    
    print("PASS: Irradiance units are reasonable")


if __name__ == "__main__":
    print("=" * 60)
    print("LensEvolver Power Balance Tests")
    print("=" * 60)
    
    test_power_balance_uniform_distribution()
    print()
    
    test_power_balance_with_absorption()
    print()
    
    test_probe_irradiance_calculation()
    print()
    
    test_empty_input()
    print()
    
    test_irradiance_units()
    print()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
