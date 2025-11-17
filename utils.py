"""
Utility functions for point cloud operations.
"""

import numpy as np
from gaussian_splat import PointCloud


def generate_sphere_point_cloud(num_points: int = 1000, radius: float = 1.0) -> PointCloud:
    """
    Generate a point cloud in the shape of a sphere.
    
    Args:
        num_points: Number of points to generate
        radius: Radius of the sphere
        
    Returns:
        PointCloud representing a sphere
    """
    # Use Fibonacci sphere algorithm for uniform distribution
    indices = np.arange(0, num_points, dtype=np.float32)
    phi = np.arccos(1 - 2 * (indices + 0.5) / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    
    points = np.stack([x, y, z], axis=1)
    
    # Color based on position (normalized to [0, 1])
    colors = (points + radius) / (2 * radius)
    
    return PointCloud(points, colors)


def generate_cube_point_cloud(num_points: int = 1000, size: float = 1.0) -> PointCloud:
    """
    Generate a point cloud in the shape of a cube.
    
    Args:
        num_points: Number of points to generate
        size: Size of the cube
        
    Returns:
        PointCloud representing a cube
    """
    points = np.random.uniform(-size/2, size/2, (num_points, 3)).astype(np.float32)
    
    # Color based on position (normalized to [0, 1])
    colors = (points + size/2) / size
    
    return PointCloud(points, colors)


def generate_torus_point_cloud(num_points: int = 1000, 
                               major_radius: float = 1.0,
                               minor_radius: float = 0.3) -> PointCloud:
    """
    Generate a point cloud in the shape of a torus.
    
    Args:
        num_points: Number of points to generate
        major_radius: Major radius (from center to tube center)
        minor_radius: Minor radius (tube radius)
        
    Returns:
        PointCloud representing a torus
    """
    u = np.random.uniform(0, 2*np.pi, num_points).astype(np.float32)
    v = np.random.uniform(0, 2*np.pi, num_points).astype(np.float32)
    
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    
    points = np.stack([x, y, z], axis=1)
    
    # Color based on angles
    colors = np.stack([
        (np.cos(u) + 1) / 2,
        (np.cos(v) + 1) / 2,
        (np.sin(u) + 1) / 2
    ], axis=1)
    
    return PointCloud(points, colors)


def compute_centroid(pc: PointCloud) -> np.ndarray:
    """
    Compute the centroid of a point cloud.
    
    Args:
        pc: Input point cloud
        
    Returns:
        3D centroid position
    """
    return np.mean(pc.points, axis=0)


def compute_bounding_box(pc: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the axis-aligned bounding box of a point cloud.
    
    Args:
        pc: Input point cloud
        
    Returns:
        Tuple of (min_corner, max_corner)
    """
    min_corner = np.min(pc.points, axis=0)
    max_corner = np.max(pc.points, axis=0)
    return min_corner, max_corner


def visualize_point_cloud_3d(pc: PointCloud, title: str = "Point Cloud", 
                             save_path: str = None):
    """
    Visualize a point cloud using matplotlib (requires matplotlib).
    
    Args:
        pc: Point cloud to visualize
        title: Plot title
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Warning: matplotlib not available for visualization")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pc.points[:, 0], pc.points[:, 1], pc.points[:, 2],
              c=pc.colors, s=20, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.max(np.ptp(pc.points, axis=0)) / 2
    mid = compute_centroid(pc)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
