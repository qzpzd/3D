from pathlib import Path

import pytest

from industrial_pc3d.io import save_pcd
from industrial_pc3d.pipeline import TraditionalConfig
from industrial_pc3d.synthetic import make_industrial_scene

pytest.importorskip("open3d")

from industrial_pc3d.open3d_backend import process_open3d


def test_open3d_process_smoke(tmp_path: Path) -> None:
    points, _ = make_industrial_scene(seed=13)
    source = tmp_path / "scene.pcd"
    save_pcd(source, points)
    config = TraditionalConfig(
        voxel_size=0.018,
        ransac_iterations=120,
        target_xy=(0.42, 0.0),
    )
    summary = process_open3d(source, config, tmp_path / "out")
    assert summary["backend"] == "Open3D"
    assert summary["clusters"]
    assert summary["counts"]["target"] > 100
    assert (tmp_path / "out" / "target.pcd").exists()
