"""
Core data structures for Gaussian splats and point clouds.
"""

import numpy as np
from typing import Tuple, Optional


class PointCloud:
    """Represents a 3D point cloud."""
    
    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Initialize a point cloud.
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (optional, values in [0, 1])
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")
        
        self.points = points.astype(np.float32)
        
        if colors is not None:
            if colors.shape != points.shape:
                raise ValueError("Colors must have same shape as points")
            self.colors = colors.astype(np.float32)
        else:
            # Default to white
            self.colors = np.ones_like(points)
    
    def __len__(self) -> int:
        return len(self.points)
    
    def copy(self) -> 'PointCloud':
        """Create a deep copy of the point cloud."""
        return PointCloud(self.points.copy(), self.colors.copy())


class GaussianSplat:
    """Represents a 3D Gaussian splat for rendering."""
    
    def __init__(self, 
                 mean: np.ndarray,
                 covariance: np.ndarray,
                 color: np.ndarray,
                 opacity: float = 1.0):
        """
        Initialize a Gaussian splat.
        
        Args:
            mean: 3D position (center) of the Gaussian
            covariance: 3x3 covariance matrix
            color: RGB color (values in [0, 1])
            opacity: Opacity value in [0, 1]
        """
        if mean.shape != (3,):
            raise ValueError("Mean must be 3D vector")
        if covariance.shape != (3, 3):
            raise ValueError("Covariance must be 3x3 matrix")
        if color.shape != (3,):
            raise ValueError("Color must be RGB (3D)")
        
        self.mean = mean.astype(np.float32)
        self.covariance = covariance.astype(np.float32)
        self.color = color.astype(np.float32)
        self.opacity = float(opacity)
    
    def copy(self) -> 'GaussianSplat':
        """Create a deep copy of the Gaussian splat."""
        return GaussianSplat(
            self.mean.copy(),
            self.covariance.copy(),
            self.color.copy(),
            self.opacity
        )


class GaussianSplatCollection:
    """Collection of Gaussian splats."""
    
    def __init__(self, splats: list[GaussianSplat]):
        """Initialize a collection of Gaussian splats."""
        self.splats = splats
    
    def __len__(self) -> int:
        return len(self.splats)
    
    def __getitem__(self, idx: int) -> GaussianSplat:
        return self.splats[idx]
    
    @classmethod
    def from_point_cloud(cls, point_cloud: PointCloud, 
                        scale: float = 0.1) -> 'GaussianSplatCollection':
        """
        Create Gaussian splats from a point cloud.
        
        Args:
            point_cloud: Input point cloud
            scale: Scale factor for the Gaussian covariances
            
        Returns:
            GaussianSplatCollection with one splat per point
        """
        splats = []
        for i in range(len(point_cloud)):
            mean = point_cloud.points[i]
            color = point_cloud.colors[i]
            # Isotropic covariance
            covariance = np.eye(3, dtype=np.float32) * (scale ** 2)
            splats.append(GaussianSplat(mean, covariance, color))
        
        return cls(splats)
    
    def copy(self) -> 'GaussianSplatCollection':
        """Create a deep copy of the collection."""
        return GaussianSplatCollection([s.copy() for s in self.splats])
