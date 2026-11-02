import numpy as np

from industrial_pc3d.geometry import (
    boundary_mask,
    crop_obb,
    estimate_normals,
    oriented_obb,
    segment_largest_plane,
    voxel_downsample,
)


def test_voxel_downsample_reduces_and_preserves_mean() -> None:
    points = np.array(
        [
            [0.21, 0.21, 0.21],
            [0.29, 0.29, 0.29],
            [0.41, 0.41, 0.41],
            [0.49, 0.49, 0.49],
        ]
    )
    down = voxel_downsample(points, 0.1)
    assert len(down) == 2
    assert np.allclose(down.mean(axis=0), points.mean(axis=0))


def test_ransac_finds_plane() -> None:
    rng = np.random.default_rng(2)
    plane = np.column_stack((rng.uniform(-1, 1, (2000, 2)), rng.normal(0, 1e-4, 2000)))
    object_points = rng.uniform([-0.2, -0.2, 0.1], [0.2, 0.2, 0.4], (500, 3))
    normal, d, inliers = segment_largest_plane(
        np.vstack((plane, object_points)), 0.005, 250, seed=2
    )
    assert abs(abs(normal[2]) - 1.0) < 1e-3
    assert len(inliers) >= len(plane) - 10
    assert abs(d) < 1e-2


def test_pca_obb_and_crop() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform([-1, -0.5, -0.25], [1, 0.5, 0.25], (5000, 3))
    box = oriented_obb(points)
    keep = crop_obb(points, box, 0.001)
    assert len(keep) == len(points)
    assert np.allclose(np.sort(box.extents), [0.5, 1.0, 2.0], atol=0.03)


def test_edges_have_normals_and_boundary() -> None:
    rng = np.random.default_rng(4)
    surface = np.column_stack((rng.uniform(-1, 1, (3000, 2)), np.zeros(3000)))
    normals, _, _ = estimate_normals(surface, 20)
    assert normals.shape == surface.shape
    flags = boundary_mask(surface, normals, 20, 120)
    assert flags.dtype == bool
    assert flags.sum() > 0

