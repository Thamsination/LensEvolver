# LensEvolver

A modular lens optimizer and raytracing analysis tool for FreeCAD.

## Features

- **Evolutionary Lens Optimization**: Generate optimized lens geometries using evolutionary algorithms
- **Raytracing Analysis**: Analyze existing lens geometries with GPU-accelerated raytracing
- **Dual LED Support**: Configure primary and secondary LED sources
- **Material Properties**: Customizable lens and absorber materials with refractive indices
- **Heatmap Visualization**: Visual representation of irradiance distribution
- **Detailed Reports**: Comprehensive analysis reports with uniformity, efficiency, and fitness metrics

## Requirements

- FreeCAD 0.19 or later
- Python with PyCUDA for GPU-accelerated raytracing
- CUDA-capable NVIDIA GPU

## Installation

1. Clone this repository or download the files
2. Place `LensEvolver.FCMacro` and `LensOptimizerModules/` in your FreeCAD Macro folder:
   - Windows: `%APPDATA%\FreeCAD\Macro\`
   - Linux: `~/.FreeCAD/Macro/`
   - macOS: `~/Library/Preferences/FreeCAD/Macro/`

## Usage

1. Open FreeCAD and create/load your model with:
   - LED Source (point/sphere for position)
   - Envelope Solid (maximum lens volume) or existing lens geometry
   - Absorber Geometry (target surface)
   
2. Select objects in this order: LED, Envelope/Lens, Absorber

3. Run the LensEvolver macro from Macro menu

4. Choose operation mode:
   - **Lens Evolution**: Generate optimized lens from envelope
   - **Raytracing Analysis**: Analyze existing lens geometry

5. Configure settings and click Start

## Module Structure

```
LensOptimizerModules/
├── __init__.py           # Package initialization
├── main.py               # Main entry point
├── config.py             # Configuration and compute budgets
├── dialogs.py            # GUI dialogs
├── evolutionary_engine.py # Evolutionary optimization
├── analysis.py           # Raytracing analysis
├── raytracer/            # GPU raytracer (PyCUDA)
│   ├── raytracer_server.py
│   ├── raytracer_client.py
│   └── ...
└── ...                   # Additional modules
```

## License

MIT License
