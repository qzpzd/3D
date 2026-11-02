"""Optional Open3D implementations of the traditional processing pipeline."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .geometry import extract_edges
from .pipeline import TraditionalConfig


def _open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is not installed. Run: python -m pip install -r requirements-open3d.txt"
        ) from exc
    return o3d


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item(),
        ),
        encoding="utf-8",
    )


def _save(cloud, path: Path) -> None:
    o3d = _open3d()
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Failed to write {path}")


def process_open3d(
    input_path: str | Path,
    config: TraditionalConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    o3d = _open3d()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = o3d.io.read_point_cloud(str(input_path))
    if raw.is_empty():
        raise RuntimeError(f"Open3D could not read point cloud: {input_path}")

    roi_box = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray(config.roi_min), np.asarray(config.roi_max)
    )
    roi = raw.crop(roi_box)
    down = roi.voxel_down_sample(config.voxel_size)
    clean, _ = down.remove_statistical_outlier(
        nb_neighbors=config.sor_k, std_ratio=config.sor_std_ratio
    )
    if config.radius_nb_points > 0:
        clean, _ = clean.remove_radius_outlier(
            nb_points=config.radius_nb_points, radius=config.radius
        )
    if len(clean.points) < 10:
        raise RuntimeError("Too few points remain after Open3D preprocessing")

    plane_model, plane_indices = clean.segment_plane(
        distance_threshold=config.ransac_distance,
        ransac_n=3,
        num_iterations=config.ransac_iterations,
    )
    parts = clean.select_by_index(plane_indices, invert=True)
    labels = np.asarray(
        parts.cluster_dbscan(
            eps=config.cluster_eps,
            min_points=config.cluster_min_samples,
            print_progress=False,
        )
    )
    raw_ids = [int(x) for x in np.unique(labels) if x >= 0]
    sizes = {cluster_id: int(np.sum(labels == cluster_id)) for cluster_id in raw_ids}
    cluster_ids = [
        cluster_id
        for cluster_id in raw_ids
        if sizes[cluster_id] >= config.cluster_min_points
    ]
    if not cluster_ids and raw_ids:
        cluster_ids = [max(raw_ids, key=sizes.get)]
    if not cluster_ids:
        raise RuntimeError("Open3D DBSCAN found no workpiece cluster")

    points = np.asarray(parts.points)
    centroids = {
        cluster_id: points[labels == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    }
    if config.target_xy is None:
        target_id = max(cluster_ids, key=sizes.get)
    else:
        target_xy = np.asarray(config.target_xy)
        target_id = min(
            cluster_ids, key=lambda x: np.linalg.norm(centroids[x][:2] - target_xy)
        )
    target_indices = np.flatnonzero(labels == target_id)
    target = parts.select_by_index(target_indices)
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=min(config.normal_k, len(target.points)))
    )
    target.orient_normals_consistent_tangent_plane(k=min(config.normal_k, len(target.points)))
    obb = (
        target.get_minimal_oriented_bounding_box()
        if hasattr(target, "get_minimal_oriented_bounding_box")
        else target.get_oriented_bounding_box()
    )
    inside = obb.get_point_indices_within_bounding_box(target.points)
    target = target.select_by_index(inside)
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=min(config.normal_k, len(target.points)))
    )

    edge_mask, _, variation = extract_edges(
        np.asarray(target.points),
        k=config.normal_k,
        variation_quantile=config.edge_quantile,
        boundary_angle_deg=config.boundary_angle_deg,
    )
    edges = o3d.geometry.PointCloud()
    edges.points = o3d.utility.Vector3dVector(np.asarray(target.points)[edge_mask])
    aabb = target.get_axis_aligned_bounding_box()

    _save(raw, output / "scene_raw.pcd")
    _save(clean, output / "scene_roi_denoised.pcd")
    _save(parts, output / "parts_without_plane.pcd")
    _save(target, output / "target.pcd")
    _save(edges, output / "target_edges.pcd")
    summary: dict[str, object] = {
        "backend": "Open3D",
        "counts": {
            "raw": len(raw.points),
            "roi": len(roi.points),
            "downsampled": len(down.points),
            "denoised": len(clean.points),
            "plane_inliers": len(plane_indices),
            "without_plane": len(parts.points),
            "target": len(target.points),
            "target_edges": int(edge_mask.sum()),
        },
        "plane": {
            "model": np.asarray(plane_model).tolist(),
            "inlier_ratio": len(plane_indices) / len(clean.points),
        },
        "clusters": sizes,
        "selected_cluster": int(target_id),
        "obb": {
            "center": np.asarray(obb.center).tolist(),
            "extent": np.asarray(obb.extent).tolist(),
            "rotation": np.asarray(obb.R).tolist(),
        },
        "aabb": {
            "min": np.asarray(aabb.get_min_bound()).tolist(),
            "max": np.asarray(aabb.get_max_bound()).tolist(),
        },
        "normal_variation_range": [
            float(variation.min()),
            float(variation.max()),
        ],
        "config": asdict(config),
    }
    _write_json(output / "open3d_summary.json", summary)
    return summary


def _registration_features(pcd, voxel_size: float):
    o3d = _open3d()
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return down, feature


def align_open3d(
    source_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    voxel_size: float = 0.015,
    ransac_iterations: int = 100000,
) -> dict[str, object]:
    o3d = _open3d()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    if source.is_empty() or target.is_empty():
        raise RuntimeError("Open3D failed to read source or target cloud")

    source_down, source_feature = _registration_features(source, voxel_size)
    target_down, target_feature = _registration_features(target, voxel_size)
    distance = voxel_size * 1.5
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_feature,
        target_feature,
        True,
        distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            ransac_iterations, 0.999
        ),
    )
    fine = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        voxel_size * 0.4,
        coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    aligned = copy.deepcopy(source)
    aligned.transform(fine.transformation)
    _save(aligned, output / "source_aligned.pcd")
    np.savetxt(output / "transform_4x4.txt", fine.transformation, fmt="%.12g")
    summary: dict[str, object] = {
        "backend": "Open3D",
        "coarse": {
            "fitness": coarse.fitness,
            "inlier_rmse": coarse.inlier_rmse,
            "correspondences": len(coarse.correspondence_set),
        },
        "fine": {
            "fitness": fine.fitness,
            "inlier_rmse": fine.inlier_rmse,
            "correspondences": len(fine.correspondence_set),
        },
        "transform_4x4": fine.transformation.tolist(),
    }
    _write_json(output / "open3d_registration_summary.json", summary)
    return summary

