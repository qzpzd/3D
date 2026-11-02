from pathlib import Path

import numpy as np

from industrial_pc3d.io import PointCloudData, load_point_cloud, save_pcd
from industrial_pc3d.pipeline import TraditionalConfig, TraditionalPipeline
from industrial_pc3d.synthetic import make_industrial_scene


def test_pcd_ascii_roundtrip(tmp_path: Path) -> None:
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    path = tmp_path / "cloud.pcd"
    save_pcd(path, points)
    loaded = load_point_cloud(path)
    assert isinstance(loaded, PointCloudData)
    assert np.allclose(loaded.xyz, points)


def test_traditional_pipeline_end_to_end() -> None:
    points, _ = make_industrial_scene(seed=11)
    config = TraditionalConfig(
        voxel_size=0.018,
        ransac_iterations=180,
        target_xy=(0.42, 0.0),
        boundary_angle_deg=120,
    )
    result = TraditionalPipeline(config).process(points)
    assert result.summary["counts"]["plane_inliers"] > 1000
    assert result.summary["counts"]["clusters"] == 2
    assert len(result.target) > 100
    assert len(result.target_edges) > 0
    assert result.summary["obb"]["extents"][0] > 0.2

