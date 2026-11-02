from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


@dataclass
class OBB:
    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.center) @ self.axes

    def from_local(self, local: np.ndarray) -> np.ndarray:
        return np.asarray(local) @ self.axes.T + self.center


def pass_through(
    points: np.ndarray,
    minimum: list[float] | tuple[float, ...],
    maximum: list[float] | tuple[float, ...],
) -> np.ndarray:
    pts = np.asarray(points)
    lo = np.asarray(minimum, dtype=np.float64)
    hi = np.asarray(maximum, dtype=np.float64)
    if np.any(lo >= hi):
        raise ValueError("ROI minimum must be smaller than maximum")
    return np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(pts) == 0:
        return pts.copy()
    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    output = np.column_stack(
        [np.bincount(inverse, weights=pts[:, axis]) / counts for axis in range(3)]
    )
    return output


def statistical_outlier_mask(
    points: np.ndarray, k: int = 20, std_ratio: float = 2.5
) -> np.ndarray:
    pts = np.asarray(points)
    if len(pts) <= k:
        return np.ones(len(pts), dtype=bool)
    distances, _ = cKDTree(pts).query(pts, k=k + 1)
    mean_distance = np.mean(distances[:, 1:], axis=1)
    cutoff = np.mean(mean_distance) + std_ratio * np.std(mean_distance)
    return mean_distance <= cutoff


def radius_outlier_mask(
    points: np.ndarray, radius: float, min_neighbors: int
) -> np.ndarray:
    pts = np.asarray(points)
    counts = cKDTree(pts).query_ball_point(pts, radius, return_length=True)
    return counts - 1 >= min_neighbors


def estimate_normals(
    points: np.ndarray, k: int = 24
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("At least three points are required")
    k = min(max(3, k), len(pts))
    _, indices = cKDTree(pts).query(pts, k=k)
    neighborhoods = pts[indices] - pts[:, None, :]
    covariance = np.einsum("nki,nkj->nij", neighborhoods, neighborhoods)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    total = np.maximum(np.sum(eigenvalues, axis=1), 1e-15)
    variation = eigenvalues[:, 0] / total
    return normals, eigenvalues, variation


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int = 500,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("RANSAC needs at least three points")
    rng = np.random.default_rng(seed)
    best_indices = np.empty(0, dtype=np.int64)
    for _ in range(iterations):
        sample = pts[rng.choice(len(pts), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        d = -np.dot(normal, sample[0])
        inliers = np.flatnonzero(np.abs(pts @ normal + d) <= distance_threshold)
        if len(inliers) > len(best_indices):
            best_indices = inliers
    if len(best_indices) < 3:
        raise RuntimeError("RANSAC did not find a valid plane")
    # Least-squares plane refinement on the winning inliers.
    centroid = pts[best_indices].mean(axis=0)
    _, _, vh = np.linalg.svd(pts[best_indices] - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    return normal, d


def segment_largest_plane(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int = 500,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal, d = fit_plane_ransac(points, distance_threshold, iterations, seed)
    inliers = np.flatnonzero(np.abs(np.asarray(points) @ normal + d) <= distance_threshold)
    return normal, d, inliers


def cluster_dbscan(
    points: np.ndarray, eps: float, min_samples: int
) -> np.ndarray:
    if eps <= 0 or min_samples < 1:
        raise ValueError("DBSCAN eps must be positive and min_samples >= 1")
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)


def aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points)
    return np.min(pts, axis=0), np.max(pts, axis=0)


def oriented_obb(points: np.ndarray) -> OBB:
    """PCA OBB; stable and fast, but not guaranteed minimum-volume for every shape."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("At least three points are required for an OBB")
    center = pts.mean(axis=0)
    covariance = np.cov(pts - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (pts - center) @ axes
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    corrected_center = center + axes @ ((local_min + local_max) / 2)
    return OBB(
        center=corrected_center,
        axes=axes,
        extents=local_max - local_min,
    )


def crop_obb(points: np.ndarray, box: OBB, margin: float = 0.0) -> np.ndarray:
    local = box.to_local(points)
    keep = np.all(np.abs(local) <= box.extents / 2 + margin, axis=1)
    return np.flatnonzero(keep)


def crop_aabb(
    points: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    pts = np.asarray(points)
    keep = np.all(
        (pts >= np.asarray(minimum) - margin) & (pts <= np.asarray(maximum) + margin),
        axis=1,
    )
    return np.flatnonzero(keep)


def boundary_mask(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 24,
    angle_threshold_deg: float = 120.0,
) -> np.ndarray:
    """Detect scan/surface boundaries from a large angular gap in tangent neighbors."""
    pts = np.asarray(points)
    normals = np.asarray(normals)
    if len(pts) < 3:
        return np.zeros(len(pts), dtype=bool)
    k = min(max(3, k), len(pts))
    _, indices = cKDTree(pts).query(pts, k=k)
    flags = np.zeros(len(pts), dtype=bool)
    threshold = np.deg2rad(angle_threshold_deg)
    for i, neighbor_ids in enumerate(indices):
        difference = pts[neighbor_ids] - pts[i]
        tangent = difference - np.outer(difference @ normals[i], normals[i])
        lengths = np.linalg.norm(tangent, axis=1)
        tangent = tangent[lengths > 1e-10]
        if len(tangent) < 3:
            flags[i] = True
            continue
        reference = tangent[0]
        reference /= np.linalg.norm(reference)
        binormal = np.cross(normals[i], reference)
        angles = np.sort(np.mod(np.arctan2(tangent @ binormal, tangent @ reference), 2 * np.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        flags[i] = gaps.max() > threshold
    return flags


def extract_edges(
    points: np.ndarray,
    k: int = 24,
    variation_quantile: float = 0.9,
    boundary_angle_deg: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals, eigenvalues, variation = estimate_normals(points, k)
    corner_like = variation >= np.quantile(variation, variation_quantile)
    boundary = boundary_mask(points, normals, k, boundary_angle_deg)
    return corner_like | boundary, normals, variation


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Correspondence arrays must both have shape (N >= 3, 3)")
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    covariance = (src - src_center).T @ (dst - dst_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = dst_center - rotation @ src_center
    return rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(rotation).T + np.asarray(translation)


def transform_to_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix

