from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sample_box_surface(
    rng: np.random.Generator,
    center: np.ndarray,
    size: np.ndarray,
    point_count: int,
) -> np.ndarray:
    size = np.asarray(size, dtype=np.float64)
    areas = np.array(
        [size[1] * size[2], size[1] * size[2], size[0] * size[2],
         size[0] * size[2], size[0] * size[1], size[0] * size[1]]
    )
    faces = rng.choice(6, size=point_count, p=areas / areas.sum())
    uv = rng.random((point_count, 2)) - 0.5
    local = np.empty((point_count, 3), dtype=np.float64)
    for face in range(6):
        mask = faces == face
        count = int(mask.sum())
        if count == 0:
            continue
        if face == 0:
            local[mask] = np.column_stack((np.full(count, -0.5), uv[mask]))
        elif face == 1:
            local[mask] = np.column_stack((np.full(count, 0.5), uv[mask]))
        elif face == 2:
            local[mask] = np.column_stack((uv[mask, 0], np.full(count, -0.5), uv[mask, 1]))
        elif face == 3:
            local[mask] = np.column_stack((uv[mask, 0], np.full(count, 0.5), uv[mask, 1]))
        elif face == 4:
            local[mask] = np.column_stack((uv[mask], np.full(count, -0.5)))
        else:
            local[mask] = np.column_stack((uv[mask], np.full(count, 0.5)))
    return np.asarray(center) + local * size


def _rotate_yaw(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return (points - center) @ rotation.T + center


def make_industrial_scene(seed: int = 7) -> tuple[np.ndarray, dict[str, object]]:
    """Synthetic fixture scene: one distant background and two separated boxes."""
    rng = np.random.default_rng(seed)
    plane_xy = rng.uniform(-1.25, 1.25, size=(48000, 2))
    plane = np.column_stack((plane_xy, rng.normal(0.0, 0.0006, len(plane_xy))))

    background = rng.uniform([-1.5, -1.5, 2.7], [1.5, 1.5, 3.3], size=(3500, 3))
    target_center = np.array([0.42, 0.0, 0.125])
    distractor_center = np.array([-0.43, 0.02, 0.105])
    target_size = np.array([0.42, 0.34, 0.25])
    distractor_size = np.array([0.32, 0.27, 0.21])
    target = _sample_box_surface(rng, target_center, target_size, 12500)
    distractor = _sample_box_surface(rng, distractor_center, distractor_size, 8500)
    target = _rotate_yaw(target, target_center, np.deg2rad(8.0))
    distractor = _rotate_yaw(distractor, distractor_center, np.deg2rad(-14.0))
    target += rng.normal(0.0, 0.00065, target.shape)
    distractor += rng.normal(0.0, 0.00065, distractor.shape)

    points = np.vstack((plane, background, target, distractor))
    permutation = rng.permutation(len(points))
    metadata: dict[str, object] = {
        "target_center": target_center.tolist(),
        "target_size": target_size.tolist(),
        "target_yaw_deg": 8.0,
        "distractor_center": distractor_center.tolist(),
        "distractor_size": distractor_size.tolist(),
        "ground_truth_note": "Rigid box dimensions before sensor noise",
    }
    return points[permutation], metadata


@dataclass
class DeepSample:
    xyz: np.ndarray
    anomaly_mask: np.ndarray
    defect_type: str
    object_label: int


def make_deep_sample(
    seed: int,
    defect_type: str = "none",
    point_count: int = 4200,
) -> DeepSample:
    """Canonical single object for the educational PatchCore-3D baseline."""
    rng = np.random.default_rng(seed)
    center = np.zeros(3)
    nominal_size = np.array([0.32, 0.24, 0.16])
    if defect_type == "none":
        size = nominal_size + rng.normal(0.0, 0.0015, 3)
    else:
        size = nominal_size.copy()
    xyz = _sample_box_surface(rng, center, size, point_count)
    anomaly = np.zeros(point_count, dtype=bool)

    if defect_type in {"dent", "bump"}:
        # Local geometric deformation on a nominally planar top surface.
        hotspot = np.array([0.0, 0.0, 0.08])
        distance = np.linalg.norm(xyz - hotspot, axis=1)
        anomaly = distance < 0.085
        amplitude = (1.0 - distance[anomaly] / 0.085) ** 2
        sign = -1.0 if defect_type == "dent" else 1.0
        xyz[anomaly, 2] += sign * 0.055 * amplitude
    elif defect_type != "none":
        raise ValueError(f"Unknown defect type: {defect_type}")

    keep = rng.random(point_count) > 0.035
    xyz = xyz[keep] + rng.normal(0.0, 0.00055, (int(keep.sum()), 3))
    anomaly = anomaly[keep]
    return DeepSample(
        xyz=xyz,
        anomaly_mask=anomaly,
        defect_type=defect_type,
        object_label=int(defect_type != "none"),
    )

