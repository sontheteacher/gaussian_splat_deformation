"""
Example script demonstrating various deformation methods on point clouds and Gaussian splats.
"""

import numpy as np
from gaussian_splat import PointCloud, GaussianSplatCollection
from deformation import (Translation, Rotation, Scaling, 
                        AffineTransform, NonRigidDeformation)
from utils import (generate_sphere_point_cloud, generate_torus_point_cloud,
                  compute_centroid, visualize_point_cloud_3d)


def example_translation():
    """Demonstrate translation deformation."""
    print("\n=== Translation Deformation ===")
    
    # Create a sphere point cloud
    pc = generate_sphere_point_cloud(num_points=500, radius=1.0)
    print(f"Created sphere with {len(pc)} points")
    
    # Apply translation
    translation = Translation(np.array([2.0, 1.0, 0.5]))
    pc_translated = translation.apply_to_point_cloud(pc)
    
    print(f"Original centroid: {compute_centroid(pc)}")
    print(f"Translated centroid: {compute_centroid(pc_translated)}")
    
    # Create Gaussian splats and apply translation
    splats = GaussianSplatCollection.from_point_cloud(pc, scale=0.1)
    splats_translated = translation.apply_to_splat_collection(splats)
    print(f"Translated {len(splats)} Gaussian splats")
    
    return pc_translated


def example_rotation():
    """Demonstrate rotation deformation."""
    print("\n=== Rotation Deformation ===")
    
    # Create a torus point cloud
    pc = generate_torus_point_cloud(num_points=800)
    print(f"Created torus with {len(pc)} points")
    
    # Rotate around Z axis by 45 degrees
    rotation = Rotation(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi/4)
    pc_rotated = rotation.apply_to_point_cloud(pc)
    
    print(f"Rotated around Z axis by 45 degrees")
    
    # Apply to Gaussian splats
    splats = GaussianSplatCollection.from_point_cloud(pc, scale=0.05)
    splats_rotated = rotation.apply_to_splat_collection(splats)
    print(f"Rotated {len(splats)} Gaussian splats")
    
    return pc_rotated


def example_scaling():
    """Demonstrate scaling deformation."""
    print("\n=== Scaling Deformation ===")
    
    # Create a sphere point cloud
    pc = generate_sphere_point_cloud(num_points=500, radius=1.0)
    print(f"Created sphere with {len(pc)} points")
    
    # Scale by 1.5x
    scaling = Scaling(scale=1.5)
    pc_scaled = scaling.apply_to_point_cloud(pc)
    
    print(f"Scaled by factor 1.5")
    
    # Apply to Gaussian splats
    splats = GaussianSplatCollection.from_point_cloud(pc, scale=0.1)
    splats_scaled = scaling.apply_to_splat_collection(splats)
    print(f"Scaled {len(splats)} Gaussian splats")
    
    return pc_scaled


def example_affine():
    """Demonstrate affine transformation."""
    print("\n=== Affine Transformation ===")
    
    # Create a sphere point cloud
    pc = generate_sphere_point_cloud(num_points=500, radius=1.0)
    print(f"Created sphere with {len(pc)} points")
    
    # Create a shearing transformation
    matrix = np.array([
        [1.0, 0.3, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    affine = AffineTransform(matrix=matrix)
    pc_affine = affine.apply_to_point_cloud(pc)
    
    print("Applied shearing transformation")
    
    # Apply to Gaussian splats
    splats = GaussianSplatCollection.from_point_cloud(pc, scale=0.1)
    splats_affine = affine.apply_to_splat_collection(splats)
    print(f"Transformed {len(splats)} Gaussian splats")
    
    return pc_affine


def example_non_rigid():
    """Demonstrate non-rigid deformation."""
    print("\n=== Non-Rigid Deformation ===")
    
    # Create a sphere point cloud
    pc = generate_sphere_point_cloud(num_points=500, radius=1.0)
    print(f"Created sphere with {len(pc)} points")
    
    # Define control points and displacements for a "bump"
    control_points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    displacements = np.array([
        [0.5, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0],
        [-0.2, 0.0, 0.0]
    ], dtype=np.float32)
    
    non_rigid = NonRigidDeformation(control_points, displacements, kernel_width=1.5)
    pc_deformed = non_rigid.apply_to_point_cloud(pc)
    
    print("Applied non-rigid deformation with 4 control points")
    
    # Apply to Gaussian splats
    splats = GaussianSplatCollection.from_point_cloud(pc, scale=0.1)
    splats_deformed = non_rigid.apply_to_splat_collection(splats)
    print(f"Deformed {len(splats)} Gaussian splats")
    
    return pc_deformed


def example_combined():
    """Demonstrate combining multiple deformations."""
    print("\n=== Combined Deformations ===")
    
    # Create a sphere point cloud
    pc = generate_sphere_point_cloud(num_points=500, radius=1.0)
    print(f"Created sphere with {len(pc)} points")
    
    # Apply multiple deformations in sequence
    # 1. Scale
    scaling = Scaling(scale=1.2)
    pc = scaling.apply_to_point_cloud(pc)
    print("1. Applied scaling (1.2x)")
    
    # 2. Rotate
    rotation = Rotation(axis=np.array([0.0, 1.0, 0.0]), angle=np.pi/6)
    pc = rotation.apply_to_point_cloud(pc)
    print("2. Applied rotation (30 degrees around Y)")
    
    # 3. Translate
    translation = Translation(np.array([1.0, 0.5, 0.0]))
    pc = translation.apply_to_point_cloud(pc)
    print("3. Applied translation")
    
    print(f"Final centroid: {compute_centroid(pc)}")
    
    return pc


def main():
    """Run all examples."""
    print("=" * 60)
    print("Gaussian Splat Deformation Examples")
    print("=" * 60)
    
    # Run each example
    example_translation()
    example_rotation()
    example_scaling()
    example_affine()
    example_non_rigid()
    example_combined()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
