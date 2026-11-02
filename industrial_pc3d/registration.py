from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    apply_transform,
    estimate_normals,
    rigid_transform,
    transform_to_matrix,
    voxel_downsample,
)


@dataclass
class RegistrationResult:
    rotation: np.ndarray
    translation: np.ndarray
    fitness: float
    inlier_rmse: float
    feature_matches: int

    @property
    def transformation(self) -> np.ndarray:
        return transform_to_matrix(self.rotation, self.translation)


def compute_fpfh(
    points: np.ndarray,
    normals: np.ndarray,
    k: int = 32,
    bins: int = 11,
) -> np.ndarray:
    """A compact FPFH-style 33-D local descriptor implementation."""
    pts = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if len(pts) != len(normals):
        raise ValueError("points and normals must have the same length")
    k = min(k, len(pts))
    _, neighbors = cKDTree(pts).query(pts, k=k)
    spfh = np.zeros((len(pts), 3, bins), dtype=np.float64)
    distances = np.zeros((len(pts), k), dtype=np.float64)
    for i in range(len(pts)):
        ids = np.atleast_1d(neighbors[i])
        offset = pts[ids] - pts[i]
        distance = np.linalg.norm(offset, axis=1)
        distances[i] = distance
        valid = distance > 1e-12
        if not np.any(valid):
            continue
        unit = offset[valid] / distance[valid, None]
        u = normals[i]
        v = np.cross(unit, u)
        v_norm = np.linalg.norm(v, axis=1)
        good_v = v_norm > 1e-12
        alpha = np.einsum("ij,ij->i", v[good_v] / v_norm[good_v, None], normals[ids][valid][good_v])
        phi = unit[good_v] @ u
        w = np.cross(u, v[good_v] / v_norm[good_v, None])
        theta = np.arctan2(
            np.einsum("ij,ij->i", w, normals[ids][valid][good_v]),
            normals[ids][valid][good_v] @ u,
        )
        alpha_idx = np.clip(((alpha + 1.0) * 0.5 * bins).astype(int), 0, bins - 1)
        phi_idx = np.clip(((phi + 1.0) * 0.5 * bins).astype(int), 0, bins - 1)
        theta_idx = np.clip(
            ((theta + np.pi) / (2 * np.pi) * bins).astype(int), 0, bins - 1
        )
        np.add.at(spfh[i, 0], alpha_idx, 1.0)
        np.add.at(spfh[i, 1], phi_idx, 1.0)
        np.add.at(spfh[i, 2], theta_idx, 1.0)

    fpfh = np.empty_like(spfh)
    valid_radius = np.median(distances[:, 1:]) * 2.5 if k > 1 else 0.0
    for i in range(len(pts)):
        ids = np.atleast_1d(neighbors[i])[1:]
        d = distances[i, 1:]
        valid = d <= max(valid_radius, 1e-9)
        if not np.any(valid):
            fpfh[i] = spfh[i]
            continue
        weights = 1.0 / np.maximum(d[valid], 1e-9)
        fpfh[i] = spfh[i] + np.einsum("n,nij->ij", weights, spfh[ids[valid]])
    fpfh = fpfh.reshape(len(pts), -1)
    norms = np.linalg.norm(fpfh, axis=1, keepdims=True)
    return fpfh / np.maximum(norms, 1e-12)


def _prepare(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    down = voxel_downsample(points, voxel_size)
    normals, _, _ = estimate_normals(down, k=min(30, len(down)))
    return down, normals


def global_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float = 0.015,
    ransac_iterations: int = 500,
    seed: int = 7,
) -> RegistrationResult:
    source_down, source_normals = _prepare(source, voxel_size)
    target_down, target_normals = _prepare(target, voxel_size)
    source_feature = compute_fpfh(source_down, source_normals)
    target_feature = compute_fpfh(target_down, target_normals)
    target_tree = cKDTree(target_down)
    feature_tree = cKDTree(target_feature)
    _, nearest = feature_tree.query(source_feature, k=1)
    source_to_target = feature_tree.query(target_feature, k=1)[1]
    mutual = source_to_target[nearest] == np.arange(len(source_down))
    candidate_ids = np.flatnonzero(mutual)
    if len(candidate_ids) < 3:
        candidate_ids = np.arange(len(source_down))

    rng = np.random.default_rng(seed)
    threshold = voxel_size * 1.8
    best_rotation = np.eye(3)
    best_translation = np.zeros(3)
    best_count = -1
    best_rmse = np.inf
    for _ in range(ransac_iterations):
        ids = rng.choice(candidate_ids, size=3, replace=False)
        correspondences = nearest[ids]
        if len(np.unique(correspondences)) < 3:
            continue
        rotation, translation = rigid_transform(source_down[ids], target_down[correspondences])
        transformed = apply_transform(source_down, rotation, translation)
        distance, _ = target_tree.query(transformed, k=1, workers=-1)
        inliers = distance <= threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_rotation = rotation
            best_translation = translation
            best_rmse = float(np.sqrt(np.mean(distance[inliers] ** 2)))

    fitness = best_count / len(source_down) if len(source_down) else 0.0
    return RegistrationResult(
        rotation=best_rotation,
        translation=best_translation,
        fitness=fitness,
        inlier_rmse=best_rmse,
        feature_matches=int(len(candidate_ids)),
    )


def refine_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: RegistrationResult | None = None,
    max_iterations: int = 35,
    distance_threshold: float = 0.03,
) -> RegistrationResult:
    rotation = initial.rotation.copy() if initial else np.eye(3)
    translation = initial.translation.copy() if initial else np.zeros(3)
    target_tree = cKDTree(target)
    for _ in range(max_iterations):
        transformed = apply_transform(source, rotation, translation)
        distances, indices = target_tree.query(transformed, k=1, workers=-1)
        keep = distances <= distance_threshold
        if keep.sum() < 3:
            break
        step_rotation, step_translation = rigid_transform(
            transformed[keep], np.asarray(target)[indices[keep]]
        )
        rotation = step_rotation @ rotation
        translation = step_rotation @ translation + step_translation

    transformed = apply_transform(source, rotation, translation)
    distances, _ = target_tree.query(transformed, k=1, workers=-1)
    keep = distances <= distance_threshold
    fitness = float(keep.mean())
    rmse = float(np.sqrt(np.mean(distances[keep] ** 2))) if keep.any() else np.inf
    return RegistrationResult(
        rotation=rotation,
        translation=translation,
        fitness=fitness,
        inlier_rmse=rmse,
        feature_matches=initial.feature_matches if initial else 0,
    )

