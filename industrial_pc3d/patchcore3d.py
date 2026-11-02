from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from .deep_models import PatchAutoEncoder3D
from .geometry import voxel_downsample


@dataclass
class PatchCore3DConfig:
    voxel_size: float = 0.009
    max_patches_per_cloud: int = 320
    patch_k: int = 24
    latent_dim: int = 24
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 1e-3
    memory_size: int = 2400
    calibration_quantile: float = 0.995
    random_seed: int = 7

    @classmethod
    def from_json(cls, path: str | Path) -> "PatchCore3DConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _farthest_centers(
    points: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    count = min(count, len(points))
    if count == len(points):
        return np.arange(len(points))
    selected = [int(rng.integers(len(points)))]
    minimum_distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        next_id = int(np.argmax(minimum_distance))
        selected.append(next_id)
        minimum_distance = np.minimum(
            minimum_distance,
            np.sum((points - points[next_id]) ** 2, axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def _make_patches(
    points: np.ndarray, center_indices: np.ndarray, patch_k: int
) -> np.ndarray:
    _, neighbors = cKDTree(points).query(points[center_indices], k=patch_k)
    return points[neighbors]


def _farthest_coreset(features: np.ndarray, size: int) -> np.ndarray:
    if len(features) <= size:
        return features
    # Start from the centroid-nearest sample for a stable, deterministic bank.
    center = features.mean(axis=0)
    selected = [int(np.argmin(np.sum((features - center) ** 2, axis=1)))]
    minimum_distance = np.sum((features - features[selected[0]]) ** 2, axis=1)
    for _ in range(1, size):
        next_id = int(np.argmax(minimum_distance))
        selected.append(next_id)
        minimum_distance = np.minimum(
            minimum_distance,
            np.sum((features - features[next_id]) ** 2, axis=1),
        )
    return features[selected]


class PatchCore3D:
    """A compact, from-scratch 3D PatchCore-style anomaly detector.

    The local autoencoder supplies patch descriptors. A coreset memory bank is
    fitted on normal data only; nearest-neighbor distance is the anomaly score.
    This is an educational baseline, not a substitute for a validated
    pretrained backbone or production threshold calibration.
    """

    def __init__(
        self,
        config: PatchCore3DConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.config = config or PatchCore3DConfig()
        self.device = torch.device(device)
        self.rng = np.random.default_rng(self.config.random_seed)
        self.model = PatchAutoEncoder3D(
            self.config.patch_k, self.config.latent_dim
        ).to(self.device)
        self.memory: torch.Tensor | None = None
        self.point_threshold: float | None = None
        self.object_threshold: float | None = None
        self.memory_scale: float | None = None
        self.reconstruction_scale: float | None = None
        self.training_history: list[float] = []

    def _prepare(self, points: np.ndarray) -> np.ndarray:
        down = voxel_downsample(points, self.config.voxel_size)
        if len(down) < self.config.patch_k:
            raise ValueError("Point cloud is too small after voxel downsampling")
        return down

    def _patches_for_cloud(
        self, points: np.ndarray, patch_limit: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        down = self._prepare(points)
        limit = patch_limit or self.config.max_patches_per_cloud
        center_indices = _farthest_centers(down, limit, self.rng)
        patches = _make_patches(down, center_indices, self.config.patch_k)
        return patches, down[center_indices], down

    def _descriptors(self, patches: np.ndarray, batch_size: int = 512) -> np.ndarray:
        outputs: list[np.ndarray] = []
        geometry_outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(patches), batch_size):
                raw_batch = patches[start : start + batch_size]
                batch = torch.from_numpy(raw_batch.astype(np.float32)).to(self.device)
                outputs.append(self.model.encode(batch).cpu().numpy())
                geometry_outputs.append(self._geometry_descriptor(raw_batch))
        learned = np.concatenate(outputs)
        geometry = np.concatenate(geometry_outputs)
        # Dual representation: learned shape code + explicit local geometry.
        features = np.concatenate((0.15 * learned, geometry), axis=1)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        return features.astype(np.float32)

    @staticmethod
    def _geometry_descriptor(patches: np.ndarray) -> np.ndarray:
        centered = patches - patches.mean(axis=1, keepdims=True)
        radius = np.sqrt(np.mean(np.sum(centered**2, axis=2), axis=1, keepdims=True))
        normalized = centered / np.maximum(radius[:, :, None], 1e-8)
        covariance = np.einsum("nki,nkj->nij", normalized, normalized) / normalized.shape[1]
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        total = np.maximum(eigenvalues.sum(axis=1), 1e-12)
        ratios = eigenvalues / total[:, None]
        radial = np.linalg.norm(normalized, axis=2)
        histograms = []
        edges = np.linspace(0.0, 2.0, 6)
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            histograms.append(np.mean((radial >= low) & (radial < high), axis=1))
        quantiles = np.transpose(
            np.quantile(normalized, [0.1, 0.5, 0.9], axis=1),
            (1, 0, 2),
        ).reshape(len(patches), -1)
        moments = np.concatenate(
            (
                np.std(normalized, axis=1),
                quantiles,
                np.column_stack(histograms),
            ),
            axis=1,
        )
        return np.concatenate((ratios, moments), axis=1)

    def fit(self, normal_clouds: Sequence[np.ndarray]) -> "PatchCore3D":
        if not normal_clouds:
            raise ValueError("At least one anomaly-free cloud is required")
        training_patches = [
            self._patches_for_cloud(cloud, min(220, self.config.max_patches_per_cloud))[0]
            for cloud in normal_clouds
        ]
        patches = torch.from_numpy(
            np.concatenate(training_patches).astype(np.float32)
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        loss_function = nn.MSELoss()
        self.model.train()
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(patches), device=self.device)
            epoch_loss = 0.0
            for start in range(0, len(patches), self.config.batch_size):
                batch = patches[permutation[start : start + self.config.batch_size]]
                reconstruction, target = self.model(batch)
                loss = loss_function(reconstruction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch)
            self.training_history.append(epoch_loss / len(patches))

        all_features = []
        for cloud in normal_clouds:
            patch_data, _, _ = self._patches_for_cloud(cloud)
            all_features.append(self._descriptors(patch_data))
        features = np.concatenate(all_features)
        features = _farthest_coreset(features, self.config.memory_size)
        self.memory = torch.from_numpy(features.astype(np.float32)).to(self.device)
        return self

    def calibrate(self, validation_clouds: Sequence[np.ndarray]) -> None:
        if self.memory is None:
            raise RuntimeError("Call fit before calibrate")
        raw_results = [self._raw_score(cloud) for cloud in validation_clouds]
        self.memory_scale = float(
            np.median(np.concatenate([x["patch_distances"] for x in raw_results]))
            + 1e-9
        )
        self.reconstruction_scale = float(
            np.median(
                np.concatenate([x["reconstruction_errors"] for x in raw_results])
            )
            + 1e-9
        )
        all_scores: list[np.ndarray] = []
        object_scores: list[float] = []
        for cloud in validation_clouds:
            result = self.score(cloud)
            all_scores.append(result["point_scores"])
            object_scores.append(float(result["object_score"]))
        self.point_threshold = float(
            np.quantile(
                np.concatenate(all_scores),
                self.config.calibration_quantile,
            )
        )
        self.object_threshold = float(max(object_scores) * 1.01 + 1e-9)

    def _raw_score(self, points: np.ndarray) -> dict[str, np.ndarray | float]:
        if self.memory is None:
            raise RuntimeError("Call fit before score")
        patches, center_points, down = self._patches_for_cloud(points)
        descriptors = torch.from_numpy(self._descriptors(patches)).to(self.device)
        patch_distances = torch.min(
            torch.cdist(descriptors, self.memory, p=2), dim=1
        ).values.cpu().numpy()
        patch_tensor = torch.from_numpy(patches.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            prediction, target = self.model(patch_tensor)
        reconstruction_errors = torch.mean(
            (prediction - target) ** 2, dim=1
        ).cpu().numpy()
        return {
            "patch_distances": patch_distances.astype(np.float64),
            "reconstruction_errors": reconstruction_errors.astype(np.float64),
            "center_points": center_points,
            "downsampled_points": down,
        }

    def score(self, points: np.ndarray) -> dict[str, np.ndarray | float]:
        raw = self._raw_score(points)
        patch_distances = raw["patch_distances"]
        reconstruction_errors = raw["reconstruction_errors"]
        patch_scores = patch_distances
        nearest_center = cKDTree(raw["center_points"]).query(points, k=1)[1]
        point_scores = patch_scores[nearest_center]
        return {
            "point_scores": point_scores.astype(np.float64),
            "patch_scores": patch_scores.astype(np.float64),
            "patch_distances": patch_distances,
            "reconstruction_errors": reconstruction_errors,
            "center_points": raw["center_points"],
            "downsampled_points": raw["downsampled_points"],
            "object_score": float(np.max(point_scores)),
        }

    def save(self, path: str | Path) -> None:
        if self.memory is None:
            raise RuntimeError("Cannot save an unfitted model")
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.model.state_dict(),
                "memory": self.memory.cpu(),
                "point_threshold": self.point_threshold,
                "object_threshold": self.object_threshold,
                "memory_scale": self.memory_scale,
                "reconstruction_scale": self.reconstruction_scale,
                "training_history": self.training_history,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "PatchCore3D":
        payload = torch.load(path, map_location=device, weights_only=False)
        detector = cls(PatchCore3DConfig(**payload["config"]), device=device)
        detector.model.load_state_dict(payload["state_dict"])
        detector.memory = payload["memory"].to(detector.device)
        detector.point_threshold = payload["point_threshold"]
        detector.object_threshold = payload["object_threshold"]
        detector.memory_scale = payload.get("memory_scale")
        detector.reconstruction_scale = payload.get("reconstruction_scale")
        detector.training_history = payload.get("training_history", [])
        return detector

