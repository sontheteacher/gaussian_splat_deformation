"""
Tests for Gaussian splat deformation methods.
"""

import numpy as np
import sys

from gaussian_splat import PointCloud, GaussianSplat, GaussianSplatCollection
from deformation import (Translation, Rotation, Scaling, 
                        AffineTransform, NonRigidDeformation)
from utils import (generate_sphere_point_cloud, compute_centroid,
                  generate_cube_point_cloud, generate_torus_point_cloud)


def test_point_cloud_creation():
    """Test basic point cloud creation."""
    print("Testing PointCloud creation...", end=" ")
    
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    pc = PointCloud(points)
    
    assert len(pc) == 3
    assert pc.points.shape == (3, 3)
    assert pc.colors.shape == (3, 3)
    
    # Test with colors
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    pc2 = PointCloud(points, colors)
    assert np.allclose(pc2.colors, colors)
    
    print("✓")


def test_gaussian_splat_creation():
    """Test Gaussian splat creation."""
    print("Testing GaussianSplat creation...", end=" ")
    
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.eye(3, dtype=np.float32) * 0.5
    color = np.array([1.0, 0.5, 0.0])
    
    splat = GaussianSplat(mean, cov, color, opacity=0.8)
    
    assert splat.mean.shape == (3,)
    assert splat.covariance.shape == (3, 3)
    assert splat.color.shape == (3,)
    assert splat.opacity == 0.8
    
    print("✓")


def test_gaussian_splat_from_point_cloud():
    """Test creating Gaussian splats from point cloud."""
    print("Testing GaussianSplatCollection.from_point_cloud...", end=" ")
    
    points = np.random.randn(10, 3).astype(np.float32)
    pc = PointCloud(points)
    
    splats = GaussianSplatCollection.from_point_cloud(pc, scale=0.1)
    
    assert len(splats) == len(pc)
    for i in range(len(splats)):
        assert np.allclose(splats[i].mean, pc.points[i])
        assert np.allclose(splats[i].color, pc.colors[i])
    
    print("✓")


def test_translation():
    """Test translation deformation."""
    print("Testing Translation...", end=" ")
    
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    pc = PointCloud(points)
    
    offset = np.array([1.0, 2.0, 3.0])
    translation = Translation(offset)
    
    pc_trans = translation.apply_to_point_cloud(pc)
    
    expected = points + offset
    assert np.allclose(pc_trans.points, expected)
    
    # Test on Gaussian splat
    splat = GaussianSplat(np.array([0, 0, 0]), np.eye(3), np.array([1, 1, 1]))
    splat_trans = translation.apply_to_splat(splat)
    assert np.allclose(splat_trans.mean, offset)
    
    print("✓")


def test_rotation():
    """Test rotation deformation."""
    print("Testing Rotation...", end=" ")
    
    # Rotate 90 degrees around Z axis
    rotation = Rotation(axis=np.array([0, 0, 1]), angle=np.pi/2)
    
    # Point on X axis should go to Y axis
    points = np.array([[1, 0, 0]], dtype=np.float32)
    pc = PointCloud(points)
    
    pc_rot = rotation.apply_to_point_cloud(pc)
    
    # Should be approximately [0, 1, 0]
    expected = np.array([[0, 1, 0]], dtype=np.float32)
    assert np.allclose(pc_rot.points, expected, atol=1e-6)
    
    print("✓")


def test_scaling():
    """Test scaling deformation."""
    print("Testing Scaling...", end=" ")
    
    points = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
    pc = PointCloud(points)
    
    scaling = Scaling(scale=2.0)
    pc_scaled = scaling.apply_to_point_cloud(pc)
    
    expected = points * 2.0
    assert np.allclose(pc_scaled.points, expected)
    
    # Test scaling with center
    center = np.array([1, 1, 1], dtype=np.float32)
    scaling_centered = Scaling(scale=2.0, center=center)
    pc_scaled2 = scaling_centered.apply_to_point_cloud(pc)
    
    expected2 = (points - center) * 2.0 + center
    assert np.allclose(pc_scaled2.points, expected2)
    
    print("✓")


def test_affine_transform():
    """Test affine transformation."""
    print("Testing AffineTransform...", end=" ")
    
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    pc = PointCloud(points)
    
    # Identity transform
    matrix = np.eye(3, dtype=np.float32)
    affine = AffineTransform(matrix)
    
    pc_trans = affine.apply_to_point_cloud(pc)
    assert np.allclose(pc_trans.points, points)
    
    # Scaling transform
    matrix = np.eye(3, dtype=np.float32) * 2.0
    affine = AffineTransform(matrix)
    pc_trans = affine.apply_to_point_cloud(pc)
    assert np.allclose(pc_trans.points, points * 2.0)
    
    print("✓")


def test_non_rigid_deformation():
    """Test non-rigid deformation."""
    print("Testing NonRigidDeformation...", end=" ")
    
    # Create a simple point cloud
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    pc = PointCloud(points)
    
    # Control points and displacements
    control_points = np.array([[1, 0, 0]], dtype=np.float32)
    displacements = np.array([[0, 1, 0]], dtype=np.float32)
    
    non_rigid = NonRigidDeformation(control_points, displacements, kernel_width=1.0)
    pc_deformed = non_rigid.apply_to_point_cloud(pc)
    
    # Point at [1, 0, 0] should be affected most
    # Points farther away should be affected less
    assert pc_deformed.points.shape == pc.points.shape
    assert not np.allclose(pc_deformed.points, pc.points)  # Should be different
    
    print("✓")


def test_generate_sphere():
    """Test sphere generation."""
    print("Testing generate_sphere_point_cloud...", end=" ")
    
    pc = generate_sphere_point_cloud(num_points=100, radius=2.0)
    
    assert len(pc) == 100
    
    # All points should be approximately on sphere surface
    distances = np.linalg.norm(pc.points, axis=1)
    assert np.allclose(distances, 2.0, atol=0.1)
    
    print("✓")


def test_generate_cube():
    """Test cube generation."""
    print("Testing generate_cube_point_cloud...", end=" ")
    
    pc = generate_cube_point_cloud(num_points=100, size=2.0)
    
    assert len(pc) == 100
    
    # All points should be within cube bounds
    assert np.all(pc.points >= -1.0)
    assert np.all(pc.points <= 1.0)
    
    print("✓")


def test_generate_torus():
    """Test torus generation."""
    print("Testing generate_torus_point_cloud...", end=" ")
    
    pc = generate_torus_point_cloud(num_points=100, major_radius=1.0, minor_radius=0.3)
    
    assert len(pc) == 100
    assert pc.points.shape == (100, 3)
    
    print("✓")


def test_compute_centroid():
    """Test centroid computation."""
    print("Testing compute_centroid...", end=" ")
    
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
    pc = PointCloud(points)
    
    centroid = compute_centroid(pc)
    expected = np.array([0.5, 0.5, 0.5])
    
    assert np.allclose(centroid, expected)
    
    print("✓")


def test_combined_deformations():
    """Test applying multiple deformations in sequence."""
    print("Testing combined deformations...", end=" ")
    
    pc = generate_sphere_point_cloud(num_points=50, radius=1.0)
    
    # Apply multiple transforms
    translation = Translation(np.array([1.0, 0.0, 0.0]))
    rotation = Rotation(axis=np.array([0, 0, 1]), angle=np.pi/4)
    scaling = Scaling(scale=2.0)
    
    pc = translation.apply_to_point_cloud(pc)
    pc = rotation.apply_to_point_cloud(pc)
    pc = scaling.apply_to_point_cloud(pc)
    
    assert len(pc) == 50
    
    print("✓")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Gaussian Splat Deformation Tests")
    print("=" * 60 + "\n")
    
    test_functions = [
        test_point_cloud_creation,
        test_gaussian_splat_creation,
        test_gaussian_splat_from_point_cloud,
        test_translation,
        test_rotation,
        test_scaling,
        test_affine_transform,
        test_non_rigid_deformation,
        test_generate_sphere,
        test_generate_cube,
        test_generate_torus,
        test_compute_centroid,
        test_combined_deformations,
    ]
    
    failed = []
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"✗ (Error: {e})")
            failed.append((test_func.__name__, e))
    
    print("\n" + "=" * 60)
    if not failed:
        print("All tests passed! ✓")
    else:
        print(f"{len(failed)} test(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
