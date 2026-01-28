"""
Main entry point for the Lens Optimizer.

This module provides the main() function that serves as the entry point
for the lens optimization macro.
"""

import FreeCAD
import FreeCADGui
from datetime import datetime

from .user_interface import get_selected_objects, get_led_position
from .dialogs import show_compute_budget_dialog
from .evolutionary_engine import evolve_lens, run_multi_phase_optimization
from .analysis import analyze_lens
from .output_generation import create_optimized_lens_object, create_evolution_report, create_analysis_report, create_best_lens_geometry
from .materials import LENS_REFRACTIVE_INDEX, ABSORBER_REFRACTIVE_INDEX


def main():
    """Main entry point for lens optimizer / raytracing analysis."""
    
    FreeCAD.Console.PrintMessage("\n" + "="*70 + "\n")
    FreeCAD.Console.PrintMessage("LENS OPTIMIZER / RAYTRACING ANALYSIS\n")
    FreeCAD.Console.PrintMessage("="*70 + "\n\n")
    
    # Get selected objects (LED, Envelope/Lens, Absorber, Centerline optional)
    led_obj, envelope_obj, absorber_obj, centerline_obj = get_selected_objects()
    
    # Only LED, Envelope/Lens, and Absorber are required; Centerline is optional
    if led_obj is None or envelope_obj is None or absorber_obj is None:
        return
    
    # Get optimization settings
    settings = show_compute_budget_dialog()
    if settings is None:
        FreeCAD.Console.PrintMessage("Operation cancelled.\n")
        return
    
    # Check if we're running analysis mode
    if settings.get('mode') == 'analysis':
        FreeCAD.Console.PrintMessage("Mode: RAYTRACING ANALYSIS\n")
        FreeCAD.Console.PrintMessage(f"Analyzing lens geometry: {envelope_obj.Label}\n\n")
        
        # Run analysis on existing geometry
        result = analyze_lens(
            led_obj, envelope_obj, absorber_obj,
            budget_name=settings['budget_name'],
            led_power_mW=settings['led_power_mW'],
            led_current_mA=settings['led_current_mA'],
            led_model=settings['led_model'],
            led_wavelength=settings['led_wavelength'],
            led1_direction=settings.get('led1_direction'),
            led2_config=settings.get('led2_config'),
            lens_material=settings.get('lens_material'),
            absorber_material=settings.get('absorber_material')
        )
        
        if result is None:
            FreeCAD.Console.PrintError("Analysis failed!\n")
            return
        
        analysis_result, stats = result
        
        # Create result group for visualization
        doc = FreeCAD.ActiveDocument
        timestamp = datetime.now().strftime("%H%M%S")
        result_group = doc.addObject("App::DocumentObjectGroup", f"Analysis_{envelope_obj.Label}_{timestamp}")
        
        # Create heatmap visualization
        grid_analysis = stats.get('grid_analysis')
        if analysis_result is not None and grid_analysis is not None:
            try:
                from .visualization import create_wavelength_heatmap
                
                FreeCAD.Console.PrintMessage("\nCreating heatmap visualizations...\n")
                
                # Create heatmap for LED 1
                led1_exits = analysis_result.get('led1_exits', {})
                if len(led1_exits.get('positions', [])) > 0:
                    heatmap1 = create_wavelength_heatmap(
                        doc, result_group,
                        led1_exits,
                        "LED1",
                        stats['num_rays']
                    )
                    if heatmap1:
                        FreeCAD.Console.PrintMessage(f"  LED 1 heatmap created: {heatmap1.Label}\n")
                
                # Create heatmap for LED 2
                led2_exits = analysis_result.get('led2_exits', {})
                if len(led2_exits.get('positions', [])) > 0:
                    heatmap2 = create_wavelength_heatmap(
                        doc, result_group,
                        led2_exits,
                        "LED2",
                        stats['num_rays']
                    )
                    if heatmap2:
                        FreeCAD.Console.PrintMessage(f"  LED 2 heatmap created: {heatmap2.Label}\n")
                        
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"Could not create heatmap: {e}\n")
                import traceback
                traceback.print_exc()
        
        # Create analysis report document
        try:
            report_doc = create_analysis_report(doc, stats, lens_name=envelope_obj.Label)
            result_group.addObject(report_doc)
            FreeCAD.Console.PrintMessage(f"Analysis report created: {report_doc.Label}\n")
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not create analysis report: {e}\n")
        
        doc.recompute()
        FreeCADGui.updateGui()
        
        FreeCAD.Console.PrintMessage(f"\nAnalysis complete! Results saved to: {result_group.Label}\n")
        return
    
    # Evolution mode continues below
    FreeCAD.Console.PrintMessage("Mode: LENS EVOLUTION\n\n")
    
    # Log LED power conversion
    FreeCAD.Console.PrintMessage(f"LED 1: {settings['led_model']} ({settings['led_wavelength']}nm) - ")
    FreeCAD.Console.PrintMessage(f"{settings['led_current_mA']}mA -> {settings['led_power_mW']:.0f}mW radiant power\n")
    led1_dir = settings.get('led1_direction', (0, 1, 0))
    FreeCAD.Console.PrintMessage(f"LED 1 Direction: ({led1_dir[0]:.2f}, {led1_dir[1]:.2f}, {led1_dir[2]:.2f})\n")
    
    # Log material properties
    lens_mat = settings.get('lens_material')
    absorber_mat = settings.get('absorber_material')
    if lens_mat:
        FreeCAD.Console.PrintMessage(f"Lens Material: {lens_mat.get('name', 'Custom')} ")
        FreeCAD.Console.PrintMessage(f"(n={lens_mat.get('refractive_index', LENS_REFRACTIVE_INDEX):.3f})\n")
    if absorber_mat:
        FreeCAD.Console.PrintMessage(f"Absorber Material: {absorber_mat.get('name', 'Custom')} ")
        FreeCAD.Console.PrintMessage(f"(n={absorber_mat.get('refractive_index', ABSORBER_REFRACTIVE_INDEX):.3f})\n")
    
    # Log lens geometry settings
    profile_shape = "Circle" if settings.get('use_circle_profile', False) else "Polygon"
    FreeCAD.Console.PrintMessage(f"Profile Shape: {profile_shape}\n")
    
    if settings.get('enable_entry_spheres', True):
        FreeCAD.Console.PrintMessage(f"Spherical Entry: Enabled (depth: {settings.get('entry_sphere_depth', 0.3)*100:.0f}%)\n")
    else:
        FreeCAD.Console.PrintMessage("Spherical Entry: Disabled (flat entry surface)\n")
    
    # Check for multi-phase refinement mode
    if settings.get('multi_phase_refinement', False):
        FreeCAD.Console.PrintMessage("Multi-Phase Refinement: ENABLED (Quick → Medium → Thorough)\n")
        
        # Run multi-phase optimization
        result = run_multi_phase_optimization(
            led_obj, envelope_obj, absorber_obj, centerline_obj,
            population_size=settings['population_size'],
            generations=settings['generations'],
            num_profiles=settings['num_profiles'],
            led_power_mW=settings['led_power_mW'],
            led_current_mA=settings['led_current_mA'],
            led_model=settings['led_model'],
            led_wavelength=settings['led_wavelength'],
            led1_direction=settings.get('led1_direction'),
            led2_config=settings.get('led2_config'),
            lens_material=settings.get('lens_material'),
            absorber_material=settings.get('absorber_material'),
            enable_entry_spheres=settings.get('enable_entry_spheres', True),
            entry_sphere_depth=settings.get('entry_sphere_depth', 0.3),
            use_circle=settings.get('use_circle_profile', False),
            distribution_weight=settings.get('distribution_weight', 0.0),
            envelope_reduction=settings.get('envelope_reduction', 0.0)
        )
    else:
        # Run single-phase evolutionary optimization
        result = evolve_lens(
            led_obj, envelope_obj, absorber_obj, centerline_obj,
            budget_name=settings['budget_name'],
            population_size=settings['population_size'],
            generations=settings['generations'],
            num_profiles=settings['num_profiles'],
            vary_profile_count=settings.get('vary_profile_count', False),
            led_power_mW=settings['led_power_mW'],
            led_current_mA=settings['led_current_mA'],
            led_model=settings['led_model'],
            led_wavelength=settings['led_wavelength'],
            led1_direction=settings.get('led1_direction'),
            led2_config=settings.get('led2_config'),
            lens_material=settings.get('lens_material'),
            absorber_material=settings.get('absorber_material'),
            enable_entry_spheres=settings.get('enable_entry_spheres', True),
            entry_sphere_depth=settings.get('entry_sphere_depth', 0.3),
            use_circle=settings.get('use_circle_profile', False),
            distribution_weight=settings.get('distribution_weight', 0.0),
            envelope_reduction=settings.get('envelope_reduction', 0.0)
        )
    
    if result is None:
        FreeCAD.Console.PrintError("Optimization failed!\n")
        return
    
    optimized_solid, stats = result
    
    # Add to document
    doc = FreeCAD.ActiveDocument
    result_group = stats.get('result_group')
    
    if result_group is None:
        timestamp = datetime.now().strftime("%H%M%S")
        group_name = f"LensEvolution_{timestamp}"
        result_group = doc.addObject("App::DocumentObjectGroup", group_name)
    
    group_name = result_group.Label
    
    # Create evolution report and add to result group
    report_doc = create_evolution_report(doc, stats, "EvolutionReport")
    result_group.addObject(report_doc)
    
    # Add optimized lens (make visible)
    lens_obj = create_optimized_lens_object(doc, optimized_solid, "BestLens")
    lens_obj.ViewObject.Visibility = True
    result_group.addObject(lens_obj)
    
    # Create toggleable centerline/profile geometry for best lens
    if stats.get('centerline_points') and stats.get('best_individual'):
        create_best_lens_geometry(
            doc, result_group,
            stats['centerline_points'],
            stats['best_individual'],
            use_circle=settings.get('use_circle_profile', False)
        )
    
    # Hide all intermediate/debug geometry, only show BestLens
    FreeCAD.Console.PrintMessage("\nCleaning up visualization (hiding intermediate geometry)...\n")
    for obj in result_group.Group:
        if obj.Name == "BestLens" or obj.Label == "BestLens":
            obj.ViewObject.Visibility = True
        else:
            obj.ViewObject.Visibility = False
    
    doc.recompute()
    FreeCADGui.updateGui()
    
    FreeCAD.Console.PrintMessage("\nEvolution complete!\n")
    FreeCAD.Console.PrintMessage(f"  Best lens added to: {group_name}\n")
    FreeCAD.Console.PrintMessage(f"  Best Fitness Score: {stats['best_fitness']:.4f}\n")
    FreeCAD.Console.PrintMessage(f"  Best Uniformity: {stats['best_uniformity']:.4f}\n")
    FreeCAD.Console.PrintMessage(f"  Best Efficiency: {stats['best_efficiency']*100:.1f}%\n")
    FreeCAD.Console.PrintMessage(f"  Best Lens Entry: {stats.get('best_lens_entry', 1.0)*100:.1f}%\n")
    FreeCAD.Console.PrintMessage(f"  Best Absorber Capture: {stats.get('best_absorber_capture', 1.0)*100:.1f}%\n")
    FreeCAD.Console.PrintMessage(f"  Generations: {stats['generations']}\n")
    FreeCAD.Console.PrintMessage(f"  Population size: {stats['population_size']}\n")
    
    # Print irradiance distribution (if available)
    grid_analysis = stats.get('best_grid_analysis')
    if grid_analysis:
        FreeCAD.Console.PrintMessage("\nIrradiance Distribution:\n")
        FreeCAD.Console.PrintMessage(f"  Mean: {grid_analysis.get('mean_intensity', 0):.4f} mW/cm2\n")
        FreeCAD.Console.PrintMessage(f"  Max:  {grid_analysis.get('max_intensity', 0):.4f} mW/cm2\n")
        FreeCAD.Console.PrintMessage(f"  Min:  {grid_analysis.get('min_intensity', 0):.4f} mW/cm2\n")
        hot_zones = len(grid_analysis.get('hot_zones', []))
        cold_zones = len(grid_analysis.get('cold_zones', []))
        FreeCAD.Console.PrintMessage(f"  Hot Zones: {hot_zones}\n")
        FreeCAD.Console.PrintMessage(f"  Cold Zones: {cold_zones}\n")
    
    # Create heatmap visualization for best result (separate per LED)
    best_result = stats.get('best_result')
    if best_result is not None and grid_analysis is not None:
        try:
            from .visualization import create_wavelength_heatmap
            import numpy as np
            num_rays = stats.get('num_rays', 1000)
            
            FreeCAD.Console.PrintMessage("\nCreating heatmap visualizations...\n")
            
            # Create heatmap for LED 1
            led1_exits = best_result.get('led1_exits', {})
            if len(led1_exits.get('positions', [])) > 0:
                heatmap1 = create_wavelength_heatmap(
                    doc, result_group,
                    led1_exits,
                    "LED1",
                    num_rays
                )
                if heatmap1:
                    FreeCAD.Console.PrintMessage(f"  LED 1 heatmap created: {heatmap1.Label}\n")
            
            # Create heatmap for LED 2
            led2_exits = best_result.get('led2_exits', {})
            if len(led2_exits.get('positions', [])) > 0:
                heatmap2 = create_wavelength_heatmap(
                    doc, result_group,
                    led2_exits,
                    "LED2",
                    num_rays
                )
                if heatmap2:
                    FreeCAD.Console.PrintMessage(f"  LED 2 heatmap created: {heatmap2.Label}\n")
                    
        except Exception as e:
            FreeCAD.Console.PrintWarning(f"Could not create heatmap: {e}\n")
            import traceback
            traceback.print_exc()


# Run the macro when executed directly
if __name__ == "__main__":
    main()
