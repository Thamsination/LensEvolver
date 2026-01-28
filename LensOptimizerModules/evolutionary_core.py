"""
Evolutionary optimization core functions for the Lens Optimizer.

This module provides mutation, crossover, and selection functions
for the evolutionary lens optimization algorithm.
"""

import random
from typing import List
import FreeCAD

from .config import (
    MIN_NUM_PROFILES,
    MAX_NUM_PROFILES,
    MIN_PROFILE_SIDES,
    MAX_PROFILE_SIDES,
    DEFAULT_PROFILE_SIDES,
    DEFAULT_MUTATION_RATE,
    DEFAULT_CROSSOVER_RATE,
    MUTATION_RADIUS_RANGE,
    MUTATION_SIDES_RANGE,
    MUTATION_ANGLE_RANGE,
    PROFILE_COUNT_MUTATION_RATE
)
from .data_classes import ProfileParams, Individual
from .geometry_validation import clamp_profile_to_envelope


def mutate_profile(profile: ProfileParams, max_radii: List[float]) -> ProfileParams:
    """Mutate a single profile's parameters.
    
    Args:
        profile: The profile to mutate
        max_radii: Pre-computed list of max radii along centerline
        
    Returns:
        New mutated ProfileParams
    """
    new_profile = profile.copy()
    
    # Mutate radius (most impactful on light distribution)
    if random.random() < 0.7:  # 70% chance
        delta_r = random.uniform(-MUTATION_RADIUS_RANGE, MUTATION_RADIUS_RANGE)
        new_profile.radius = max(1.0, new_profile.radius + delta_r)
    
    # Mutate sides (less frequent)
    if random.random() < 0.3:  # 30% chance
        delta_s = random.randint(-MUTATION_SIDES_RANGE, MUTATION_SIDES_RANGE)
        new_profile.sides = max(MIN_PROFILE_SIDES, 
                                min(MAX_PROFILE_SIDES, new_profile.sides + delta_s))
    
    # Mutate angle (rotation)
    if random.random() < 0.4:  # 40% chance
        delta_a = random.uniform(-MUTATION_ANGLE_RANGE, MUTATION_ANGLE_RANGE)
        new_profile.angle = (new_profile.angle + delta_a) % 360.0
    
    # Clamp to envelope constraints
    new_profile = clamp_profile_to_envelope(new_profile, max_radii)
    
    return new_profile


def add_profile_to_individual(individual: Individual, max_radii: List[float]) -> bool:
    """Add a new profile to an individual at a random position.
    
    Args:
        individual: The individual to modify
        max_radii: Pre-computed list of max radii along centerline
        
    Returns:
        True if profile was added, False if at maximum
    """
    if len(individual.profiles) >= MAX_NUM_PROFILES:
        return False
    
    # Find a gap to insert the new profile
    profiles = sorted(individual.profiles, key=lambda p: p.z_position)
    
    # Find largest gap
    best_pos = 0.5
    best_gap = 0.0
    
    for i in range(len(profiles) - 1):
        gap = profiles[i + 1].z_position - profiles[i].z_position
        if gap > best_gap:
            best_gap = gap
            best_pos = (profiles[i].z_position + profiles[i + 1].z_position) / 2
    
    # Also check gaps at start and end
    if len(profiles) > 0:
        if profiles[0].z_position > best_gap:
            best_gap = profiles[0].z_position
            best_pos = profiles[0].z_position / 2
        if (1.0 - profiles[-1].z_position) > best_gap:
            best_pos = (1.0 + profiles[-1].z_position) / 2
    
    # Create new profile at this position
    from .geometry_validation import get_max_radius_at_position
    max_r = get_max_radius_at_position(best_pos, max_radii)
    
    new_profile = ProfileParams(
        sides=individual.shared_sides,
        radius=max_r * 0.7,  # Start at 70% of max
        angle=individual.shared_angle,
        z_position=best_pos,
        max_radius=max_r
    )
    
    individual.profiles.append(new_profile)
    return True


def remove_profile_from_individual(individual: Individual) -> bool:
    """Remove a random profile from an individual.
    
    Args:
        individual: The individual to modify
        
    Returns:
        True if profile was removed, False if at minimum
    """
    if len(individual.profiles) <= MIN_NUM_PROFILES:
        return False
    
    # Don't remove first or last profile (they define the ends)
    profiles = sorted(individual.profiles, key=lambda p: p.z_position)
    
    if len(profiles) <= 2:
        return False
    
    # Remove a random middle profile
    remove_idx = random.randint(1, len(profiles) - 2)
    individual.profiles.remove(profiles[remove_idx])
    return True


def mutate_profile_count(individual: Individual, max_radii: List[float]) -> bool:
    """Randomly add or remove a profile from an individual.
    
    Args:
        individual: The individual to modify
        max_radii: Pre-computed list of max radii along centerline
        
    Returns:
        True if profile count was changed
    """
    if random.random() < 0.5:
        return add_profile_to_individual(individual, max_radii)
    else:
        return remove_profile_from_individual(individual)


def mutate_individual_sides(individual: Individual) -> None:
    """Mutate the shared polygon sides for an individual.
    
    Args:
        individual: The individual to modify
    """
    delta_s = random.randint(-MUTATION_SIDES_RANGE, MUTATION_SIDES_RANGE)
    individual.shared_sides = max(MIN_PROFILE_SIDES, 
                                   min(MAX_PROFILE_SIDES, individual.shared_sides + delta_s))
    individual.apply_shared_sides()


def mutate_individual(individual: Individual, max_radii: List[float],
                      mutation_rate: float = DEFAULT_MUTATION_RATE) -> Individual:
    """Mutate an individual's profiles.
    
    Args:
        individual: The individual to mutate
        max_radii: Pre-computed list of max radii along centerline
        mutation_rate: Probability of mutating each profile
        
    Returns:
        New mutated Individual
    """
    mutated = individual.copy()
    
    # Mutate shared sides occasionally
    if random.random() < 0.2:
        mutate_individual_sides(mutated)
    
    # Mutate profile count occasionally
    if random.random() < PROFILE_COUNT_MUTATION_RATE:
        mutate_profile_count(mutated, max_radii)
    
    # Mutate individual profiles
    for i, profile in enumerate(mutated.profiles):
        if random.random() < mutation_rate:
            mutated.profiles[i] = mutate_profile(profile, max_radii)
    
    # Reset fitness (needs re-evaluation)
    mutated.fitness = 0.0
    mutated.is_valid = False
    
    return mutated


def crossover(parent1: Individual, parent2: Individual, 
              max_radii: List[float]) -> Individual:
    """Create offspring by combining two parents.
    
    Uses uniform crossover where each profile position randomly
    selects from either parent.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        max_radii: Pre-computed list of max radii along centerline
        
    Returns:
        New offspring Individual
    """
    # Inherit shared parameters from random parent
    if random.random() < 0.5:
        shared_sides = parent1.shared_sides
        shared_angle = parent1.shared_angle
    else:
        shared_sides = parent2.shared_sides
        shared_angle = parent2.shared_angle
    
    # Get profiles from both parents sorted by z_position
    p1_profiles = sorted(parent1.profiles, key=lambda p: p.z_position)
    p2_profiles = sorted(parent2.profiles, key=lambda p: p.z_position)
    
    # Create child profiles by interpolating/selecting from parents
    child_profiles = []
    
    # Take average number of profiles
    n_profiles = (len(p1_profiles) + len(p2_profiles)) // 2
    n_profiles = max(MIN_NUM_PROFILES, min(MAX_NUM_PROFILES, n_profiles))
    
    for i in range(n_profiles):
        t = i / (n_profiles - 1) if n_profiles > 1 else 0.5
        
        # Find closest profile in each parent
        def find_closest(profiles, t_target):
            if not profiles:
                return None
            closest = min(profiles, key=lambda p: abs(p.z_position - t_target))
            return closest
        
        p1_closest = find_closest(p1_profiles, t)
        p2_closest = find_closest(p2_profiles, t)
        
        # Select or interpolate
        if p1_closest is None:
            selected = p2_closest.copy() if p2_closest else None
        elif p2_closest is None:
            selected = p1_closest.copy()
        elif random.random() < 0.5:
            selected = p1_closest.copy()
        else:
            selected = p2_closest.copy()
        
        if selected:
            selected.z_position = t
            selected.sides = shared_sides
            selected.angle = shared_angle
            selected = clamp_profile_to_envelope(selected, max_radii)
            child_profiles.append(selected)
    
    return Individual(
        profiles=child_profiles,
        shared_sides=shared_sides,
        shared_angle=shared_angle,
        fitness=0.0,
        is_valid=False
    )


def create_random_individual(num_profiles: int, max_radii: List[float]) -> Individual:
    """Create a random individual with specified number of profiles.
    
    Args:
        num_profiles: Number of profiles to create
        max_radii: Pre-computed list of max radii along centerline
        
    Returns:
        New random Individual
    """
    from .geometry_validation import get_max_radius_at_position
    
    shared_sides = random.randint(MIN_PROFILE_SIDES, MAX_PROFILE_SIDES)
    shared_angle = random.uniform(0, 360)
    
    profiles = []
    for i in range(num_profiles):
        t = i / (num_profiles - 1) if num_profiles > 1 else 0.5
        max_r = get_max_radius_at_position(t, max_radii)
        
        profile = ProfileParams(
            sides=shared_sides,
            radius=max_r * random.uniform(0.5, 0.9),
            angle=shared_angle,
            z_position=t,
            max_radius=max_r
        )
        profiles.append(profile)
    
    return Individual(
        profiles=profiles,
        shared_sides=shared_sides,
        shared_angle=shared_angle,
        fitness=0.0,
        is_valid=False
    )


def tournament_select(population: List[Individual], tournament_size: int = 3) -> Individual:
    """Select an individual using tournament selection.
    
    Args:
        population: List of individuals to select from
        tournament_size: Number of individuals in each tournament
        
    Returns:
        Selected individual
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda ind: ind.fitness)


def evolve_generation(population: List[Individual], max_radii: List[float],
                      elite_count: int = 2, 
                      crossover_rate: float = DEFAULT_CROSSOVER_RATE) -> List[Individual]:
    """Evolve population to create next generation.
    
    Args:
        population: Current generation
        max_radii: Pre-computed list of max radii along centerline
        elite_count: Number of best individuals to keep unchanged
        crossover_rate: Probability of using crossover vs mutation
        
    Returns:
        New generation population
    """
    # Sort by fitness
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    
    # Keep elites
    new_generation = [ind.copy() for ind in sorted_pop[:elite_count]]
    
    # Fill rest with offspring
    while len(new_generation) < len(population):
        if random.random() < crossover_rate:
            # Crossover
            parent1 = tournament_select(sorted_pop)
            parent2 = tournament_select(sorted_pop)
            child = crossover(parent1, parent2, max_radii)
        else:
            # Mutation only
            parent = tournament_select(sorted_pop)
            child = mutate_individual(parent, max_radii)
        
        new_generation.append(child)
    
    return new_generation
