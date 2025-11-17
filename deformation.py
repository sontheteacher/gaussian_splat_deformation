"""
Deformation methods for point clouds and Gaussian splats.
"""

import numpy as np
from typing import Optional
from gaussian_splat import PointCloud, GaussianSplat, GaussianSplatCollection


class Deformation:
    """Base class for deformation operations."""
    
    def apply_to_point_cloud(self, pc: PointCloud) -> PointCloud:
        """Apply deformation to a point cloud."""
        raise NotImplementedError
    
    def apply_to_splat(self, splat: GaussianSplat) -> GaussianSplat:
        """Apply deformation to a single Gaussian splat."""
        raise NotImplementedError
    
    def apply_to_splat_collection(self, collection: GaussianSplatCollection) -> GaussianSplatCollection:
        """Apply deformation to a collection of Gaussian splats."""
        new_splats = [self.apply_to_splat(splat) for splat in collection.splats]
        return GaussianSplatCollection(new_splats)


class Translation(Deformation):
    """Translation deformation."""
    
    def __init__(self, offset: np.ndarray):
        """
        Initialize translation.
        
        Args:
            offset: 3D translation vector
        """
        if offset.shape != (3,):
            raise ValueError("Offset must be 3D vector")
        self.offset = offset.astype(np.float32)
    
    def apply_to_point_cloud(self, pc: PointCloud) -> PointCloud:
        """Translate all points."""
        new_points = pc.points + self.offset
        return PointCloud(new_points, pc.colors.copy())
    
    def apply_to_splat(self, splat: GaussianSplat) -> GaussianSplat:
        """Translate Gaussian splat mean."""
        new_mean = splat.mean + self.offset
        return GaussianSplat(new_mean, splat.covariance.copy(), 
                           splat.color.copy(), splat.opacity)


class Rotation(Deformation):
    """Rotation deformation."""
    
    def __init__(self, axis: np.ndarray, angle: float):
        """
        Initialize rotation around an axis.
        
        Args:
            axis: 3D rotation axis (will be normalized)
            angle: Rotation angle in radians
        """
        if axis.shape != (3,):
            raise ValueError("Axis must be 3D vector")
        
        # Normalize axis
        self.axis = axis / np.linalg.norm(axis)
        self.angle = float(angle)
        self.rotation_matrix = self._compute_rotation_matrix()
    
    def _compute_rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix using Rodrigues' formula."""
        K = np.array([
            [0, -self.axis[2], self.axis[1]],
            [self.axis[2], 0, -self.axis[0]],
            [-self.axis[1], self.axis[0], 0]
        ], dtype=np.float32)
        
        I = np.eye(3, dtype=np.float32)
        R = I + np.sin(self.angle) * K + (1 - np.cos(self.angle)) * (K @ K)
        return R
    
    def apply_to_point_cloud(self, pc: PointCloud) -> PointCloud:
        """Rotate all points."""
        new_points = (self.rotation_matrix @ pc.points.T).T
        return PointCloud(new_points, pc.colors.copy())
    
    def apply_to_splat(self, splat: GaussianSplat) -> GaussianSplat:
        """Rotate Gaussian splat (mean and covariance)."""
        new_mean = self.rotation_matrix @ splat.mean
        # Transform covariance: R * Cov * R^T
        new_cov = self.rotation_matrix @ splat.covariance @ self.rotation_matrix.T
        return GaussianSplat(new_mean, new_cov, splat.color.copy(), splat.opacity)


class Scaling(Deformation):
    """Scaling deformation."""
    
    def __init__(self, scale: float, center: Optional[np.ndarray] = None):
        """
        Initialize uniform scaling.
        
        Args:
            scale: Scaling factor
            center: Center of scaling (default: origin)
        """
        self.scale = float(scale)
        self.center = center if center is not None else np.zeros(3, dtype=np.float32)
    
    def apply_to_point_cloud(self, pc: PointCloud) -> PointCloud:
        """Scale all points around center."""
        new_points = (pc.points - self.center) * self.scale + self.center
        return PointCloud(new_points, pc.colors.copy())
    
    def apply_to_splat(self, splat: GaussianSplat) -> GaussianSplat:
        """Scale Gaussian splat (mean and covariance)."""
        new_mean = (splat.mean - self.center) * self.scale + self.center
        # Scale covariance by scale^2
        new_cov = splat.covariance * (self.scale ** 2)
        return GaussianSplat(new_mean, new_cov, splat.color.copy(), splat.opacity)


class AffineTransform(Deformation):
    """General affine transformation."""
    
    def __init__(self, matrix: np.ndarray, translation: Optional[np.ndarray] = None):
        """
        Initialize affine transform.
        
        Args:
            matrix: 3x3 transformation matrix
            translation: 3D translation vector (optional)
        """
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")
        
        self.matrix = matrix.astype(np.float32)
        self.translation = translation if translation is not None else np.zeros(3, dtype=np.float32)
    
    def apply_to_point_cloud(self, pc: PointCloud) -> PointCloud:
        """Apply affine transform to points."""
        new_points = (self.matrix @ pc.points.T).T + self.translation
        return PointCloud(new_points, pc.colors.copy())
    
    def apply_to_splat(self, splat: GaussianSplat) -> GaussianSplat:
        """Apply affine transform to Gaussian splat."""
        new_mean = self.matrix @ splat.mean + self.translation
        # Transform covariance: A * Cov * A^T
        new_cov = self.matrix @ splat.covariance @ self.matrix.T
        return GaussianSplat(new_mean, new_cov, splat.color.copy(), splat.opacity)


class NonRigidDeformation(Deformation):
    """Non-rigid deformation using radial basis functions."""
    
    def __init__(self, control_points: np.ndarray, displacements: np.ndarray, 
                 kernel_width: float = 1.0):
        """
        Initialize non-rigid deformation.
        
        Args:
            control_points: Nx3 array of control point positions
            displacements: Nx3 array of displacements at control points
            kernel_width: Width of the radial basis function kernel
        """
        if control_points.shape[1] != 3 or displacements.shape != control_points.shape:
            raise ValueError("Control points and displacements must be Nx3 arrays")
        
        self.control_points = control_points.astype(np.float32)
        self.displacements = displacements.astype(np.float32)
        self.kernel_width = float(kernel_width)
    
    def _compute_rbf_weights(self, point: np.ndarray) -> np.ndarray:
        """Compute RBF weights for a point."""
        distances = np.linalg.norm(self.control_points - point, axis=1)
        weights = np.exp(-(distances ** 2) / (2 * self.kernel_width ** 2))
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        return weights
    
    def _deform_point(self, point: np.ndarray) -> np.ndarray:
        """Deform a single point."""
        weights = self._compute_rbf_weights(point)
        displacement = np.sum(weights[:, np.newaxis] * self.displacements, axis=0)
        return point + displacement
    
    def apply_to_point_cloud(self, pc: PointCloud) -> PointCloud:
        """Apply non-rigid deformation to point cloud."""
        new_points = np.array([self._deform_point(p) for p in pc.points])
        return PointCloud(new_points, pc.colors.copy())
    
    def apply_to_splat(self, splat: GaussianSplat) -> GaussianSplat:
        """Apply non-rigid deformation to Gaussian splat."""
        new_mean = self._deform_point(splat.mean)
        # For simplicity, keep covariance unchanged
        # (More sophisticated approach would compute local Jacobian)
        return GaussianSplat(new_mean, splat.covariance.copy(), 
                           splat.color.copy(), splat.opacity)
