# Gaussian Splat Deformation

A research implementation for testing various graphics deformation methods on point clouds and Gaussian splats. This library provides a collection of deformation techniques including rigid transformations, affine transformations, and non-rigid deformations.

## Overview

This project implements:
- **Core data structures** for point clouds and 3D Gaussian splats
- **Rigid deformations**: Translation, Rotation, Scaling
- **Affine transformations**: General 3x3 matrix transformations
- **Non-rigid deformations**: Radial Basis Function (RBF) based deformations
- **Utility functions** for generating and manipulating point clouds

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from gaussian_splat import PointCloud, GaussianSplatCollection
from deformation import Translation, Rotation, NonRigidDeformation
from utils import generate_sphere_point_cloud

# Create a point cloud
pc = generate_sphere_point_cloud(num_points=1000, radius=1.0)

# Apply translation
translation = Translation(np.array([1.0, 0.0, 0.0]))
pc_translated = translation.apply_to_point_cloud(pc)

# Create Gaussian splats from point cloud
splats = GaussianSplatCollection.from_point_cloud(pc, scale=0.1)

# Apply rotation to splats
rotation = Rotation(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi/4)
splats_rotated = rotation.apply_to_splat_collection(splats)
```

## Examples

Run the example script to see all deformation methods in action:

```bash
python example.py
```

This will demonstrate:
- Translation deformation
- Rotation deformation
- Scaling deformation
- Affine transformations
- Non-rigid deformations
- Combined deformations

## Deformation Methods

### Translation
Translates all points by a fixed offset vector.

```python
from deformation import Translation
translation = Translation(offset=np.array([1.0, 2.0, 0.5]))
```

### Rotation
Rotates points around an arbitrary axis using Rodrigues' formula.

```python
from deformation import Rotation
rotation = Rotation(axis=np.array([0.0, 1.0, 0.0]), angle=np.pi/4)
```

### Scaling
Uniformly scales points around a center point.

```python
from deformation import Scaling
scaling = Scaling(scale=1.5, center=np.array([0.0, 0.0, 0.0]))
```

### Affine Transform
Applies a general 3x3 matrix transformation with optional translation.

```python
from deformation import AffineTransform
matrix = np.array([[1.0, 0.2, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
affine = AffineTransform(matrix=matrix)
```

### Non-Rigid Deformation
Uses Radial Basis Functions (RBF) for smooth, non-rigid deformations based on control points.

```python
from deformation import NonRigidDeformation
control_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
displacements = np.array([[0.5, 0.0, 0.0], [0.0, 0.3, 0.0]])
non_rigid = NonRigidDeformation(control_points, displacements, kernel_width=1.0)
```

## Point Cloud Utilities

Generate various point cloud shapes:

```python
from utils import (generate_sphere_point_cloud, 
                  generate_cube_point_cloud,
                  generate_torus_point_cloud)

sphere = generate_sphere_point_cloud(num_points=1000, radius=1.0)
cube = generate_cube_point_cloud(num_points=1000, size=2.0)
torus = generate_torus_point_cloud(num_points=1000, major_radius=1.0, minor_radius=0.3)
```

## Architecture

The library is organized into several modules:

- **`gaussian_splat.py`**: Core data structures for point clouds and Gaussian splats
- **`deformation.py`**: Deformation method implementations
- **`utils.py`**: Utility functions for point cloud generation and manipulation
- **`example.py`**: Example usage demonstrating all deformation methods

## Research Applications

This library is designed for research in:
- Graphics and rendering techniques
- Point cloud processing
- 3D scene representation
- Deformation analysis
- Gaussian splatting methods

## License

This is a research project for testing graphics deformation methods.