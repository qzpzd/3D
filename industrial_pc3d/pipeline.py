from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    OBB,
    aabb,
    cluster_dbscan,
    crop_obb,
    estimate_normals,
    extract_edges,
    oriented_obb,
    pass_through,
    radius_outlier_mask,
    segment_largest_plane,
    statistical_outlier_mask,
    voxel_downsample,
)


@dataclass
class TraditionalConfig:
    roi_min: tuple[float, float, float] = (-2.0, -2.0, -0.1)
    roi_max: tuple[float, float, float] = (2.0, 2.0, 1.0)
    voxel_size: float = 0.012
    sor_k: int = 20
    sor_std_ratio: float = 2.5
    radius_nb_points: int = 0
    radius: float = 0.04
    ransac_distance: float = 0.006
    ransac_iterations: int = 500
    cluster_eps: float = 0.035
    cluster_min_samples: int = 8
    cluster_min_points: int = 40
    target_xy: tuple[float, float] | None = (0.42, 0.0)
    normal_k: int = 24
    edge_quantile: float = 0.9
    boundary_angle_deg: float = 120.0
    obb_margin: float = 0.015
    random_seed: int = 7

    @classmethod
    def from_json(cls, path: str | Path) -> "TraditionalConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("roi_min", "roi_max", "target_xy"):
            if key in data and data[key] is not None:
                data[key] = tuple(float(v) for v in data[key])
        return cls(**data)


@dataclass
class PipelineResult:
    raw: np.ndarray
    roi: np.ndarray
    denoised: np.ndarray
    without_plane: np.ndarray
    target: np.ndarray
    target_edges: np.ndarray
    normals: np.ndarray
    variation: np.ndarray
    cluster_labels: np.ndarray
    plane_model: np.ndarray
    target_cluster_id: int
    obb: OBB
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)


class TraditionalPipeline:
    def __init__(self, config: TraditionalConfig):
        self.config = config

    def process(self, raw_points: np.ndarray) -> PipelineResult:
        cfg = self.config
        raw = np.asarray(raw_points, dtype=np.float64)
        raw = raw[np.all(np.isfinite(raw), axis=1)]
        timings: dict[str, float] = {}
        counts: dict[str, int] = {"raw": len(raw)}
        t0 = time.perf_counter()

        roi_indices = pass_through(raw, cfg.roi_min, cfg.roi_max)
        roi = voxel_downsample(raw[roi_indices], cfg.voxel_size)
        timings["roi_and_voxel"] = time.perf_counter() - t0
        counts["roi_voxel"] = len(roi)
        if len(roi) < 20:
            raise RuntimeError("Too few points remain after ROI/downsampling")

        t0 = time.perf_counter()
        keep = statistical_outlier_mask(roi, cfg.sor_k, cfg.sor_std_ratio)
        if cfg.radius_nb_points > 0:
            keep &= radius_outlier_mask(roi, cfg.radius, cfg.radius_nb_points)
        denoised = roi[keep]
        timings["denoise"] = time.perf_counter() - t0
        counts["denoised"] = len(denoised)

        t0 = time.perf_counter()
        plane_normal, plane_d, plane_indices = segment_largest_plane(
            denoised,
            cfg.ransac_distance,
            cfg.ransac_iterations,
            cfg.random_seed,
        )
        plane_mask = np.zeros(len(denoised), dtype=bool)
        plane_mask[plane_indices] = True
        without_plane = denoised[~plane_mask]
        timings["ransac_plane"] = time.perf_counter() - t0
        counts["plane_inliers"] = len(plane_indices)
        counts["without_plane"] = len(without_plane)
        if len(without_plane) < 10:
            raise RuntimeError("Plane removal left no object points")

        t0 = time.perf_counter()
        labels = cluster_dbscan(without_plane, cfg.cluster_eps, cfg.cluster_min_samples)
        raw_cluster_ids = [x for x in np.unique(labels) if x >= 0]
        cluster_ids = [
            x for x in raw_cluster_ids if np.sum(labels == x) >= cfg.cluster_min_points
        ]
        if not cluster_ids and raw_cluster_ids:
            cluster_ids = [
                max(raw_cluster_ids, key=lambda x: np.sum(labels == x))
            ]
        if not cluster_ids:
            raise RuntimeError("No valid workpiece cluster found")
        cluster_centroids = {
            cluster_id: without_plane[labels == cluster_id].mean(axis=0)
            for cluster_id in cluster_ids
        }
        if cfg.target_xy is None:
            target_cluster_id = max(cluster_ids, key=lambda x: np.sum(labels == x))
        else:
            target_xy = np.asarray(cfg.target_xy)
            target_cluster_id = min(
                cluster_ids,
                key=lambda x: np.linalg.norm(cluster_centroids[x][:2] - target_xy),
            )
        target = without_plane[labels == target_cluster_id]
        timings["dbscan_and_select"] = time.perf_counter() - t0
        counts["clusters"] = len(cluster_ids)
        counts["ignored_small_clusters"] = len(raw_cluster_ids) - len(cluster_ids)
        counts["target"] = len(target)
        cluster_sizes = {
            str(cid): int(np.sum(labels == cid)) for cid in cluster_ids
        }

        t0 = time.perf_counter()
        edge_mask, normals, variation = extract_edges(
            target,
            cfg.normal_k,
            cfg.edge_quantile,
            cfg.boundary_angle_deg,
        )
        target_edges = target[edge_mask]
        timings["normals_and_edges"] = time.perf_counter() - t0
        counts["target_edges"] = len(target_edges)

        t0 = time.perf_counter()
        obb = oriented_obb(target)
        crop_indices = crop_obb(target, obb, cfg.obb_margin)
        target = target[crop_indices]
        edge_mask = edge_mask[crop_indices]
        normals = normals[crop_indices]
        variation = variation[crop_indices]
        target_edges = target[edge_mask]
        aabb_min, aabb_max = aabb(target)
        timings["obb_crop"] = time.perf_counter() - t0
        counts["target_after_obb"] = len(target)

        total = sum(timings.values())
        summary = {
            "counts": counts,
            "timings_seconds": {k: round(v, 6) for k, v in timings.items()},
            "total_seconds": round(total, 6),
            "plane": {
                "normal": plane_normal.tolist(),
                "d": float(plane_d),
                "inlier_ratio": round(len(plane_indices) / len(denoised), 4),
            },
            "clusters": cluster_sizes,
            "selected_cluster": int(target_cluster_id),
            "obb": {
                "center": obb.center.tolist(),
                "axes": obb.axes.tolist(),
                "extents": obb.extents.tolist(),
                "note": "PCA OBB approximation; validate scale-specific fixtures",
            },
            "aabb": {"min": aabb_min.tolist(), "max": aabb_max.tolist()},
            "config": asdict(cfg),
        }
        return PipelineResult(
            raw=raw,
            roi=roi,
            denoised=denoised,
            without_plane=without_plane,
            target=target,
            target_edges=target_edges,
            normals=normals,
            variation=variation,
            cluster_labels=labels,
            plane_model=np.r_[plane_normal, plane_d],
            target_cluster_id=int(target_cluster_id),
            obb=obb,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            summary=summary,
        )

