from __future__ import annotations

import torch
from torch import nn


def farthest_point_sample(xyz: torch.Tensor, sample_count: int) -> torch.Tensor:
    """xyz: (B, N, 3), returns indices (B, sample_count)."""
    batch, point_count, _ = xyz.shape
    sample_count = min(sample_count, point_count)
    device = xyz.device
    indices = torch.zeros(batch, sample_count, dtype=torch.long, device=device)
    distance = torch.full((batch, point_count), 1e10, device=device)
    farthest = torch.zeros(batch, dtype=torch.long, device=device)
    batch_ids = torch.arange(batch, device=device)
    for i in range(sample_count):
        indices[:, i] = farthest
        centroid = xyz[batch_ids, farthest].unsqueeze(1)
        current = torch.sum((xyz - centroid) ** 2, dim=2)
        distance = torch.minimum(distance, current)
        farthest = torch.max(distance, dim=1).indices
    return indices


class PointNetEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.global_layers = nn.Sequential(
            nn.Conv1d(128, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, N, C) -> (B, C, N)
        local = self.local(xyz.transpose(1, 2))
        global_feature = torch.max(self.global_layers(local), dim=2).values
        return local, global_feature


class PointNetClassifier(nn.Module):
    def __init__(self, class_count: int, input_channels: int = 3):
        super().__init__()
        self.encoder = PointNetEncoder(input_channels)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, class_count),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        _, global_feature = self.encoder(xyz)
        return self.head(global_feature)


class PointNetSegmenter(nn.Module):
    def __init__(self, class_count: int, input_channels: int = 3):
        super().__init__()
        self.encoder = PointNetEncoder(input_channels)
        self.head = nn.Sequential(
            nn.Conv1d(512 + 128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, class_count, 1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        local, global_feature = self.encoder(xyz)
        expanded = global_feature[:, :, None].expand(-1, -1, xyz.shape[1])
        return self.head(torch.cat((local, expanded), dim=1)).transpose(1, 2)


class SetAbstractionKNN(nn.Module):
    """PointNet++ style sample-group-MLP abstraction with kNN grouping."""

    def __init__(
        self,
        sample_count: int,
        neighbor_count: int,
        input_channels: int,
        output_channels: int,
    ):
        super().__init__()
        self.sample_count = sample_count
        self.neighbor_count = neighbor_count
        self.mlp = nn.Sequential(
            nn.Conv2d(input_channels + 3, output_channels // 2, 1),
            nn.BatchNorm2d(output_channels // 2),
            nn.ReLU(),
            nn.Conv2d(output_channels // 2, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    @staticmethod
    def _group(xyz: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, _, channels = xyz.shape
        flattened = indices.reshape(batch, -1)
        gathered = torch.gather(
            xyz,
            1,
            flattened[:, :, None].expand(-1, -1, channels),
        )
        return gathered.reshape(*indices.shape, channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sampled_indices = farthest_point_sample(xyz, self.sample_count)
        centers = self._group(xyz, sampled_indices)
        distances = torch.cdist(centers, xyz)
        neighbor_indices = torch.topk(
            distances, k=min(self.neighbor_count, xyz.shape[1]), dim=2, largest=False
        ).indices
        grouped_xyz = self._group(xyz, neighbor_indices)
        grouped_xyz = grouped_xyz - centers[:, :, None, :]
        grouped_features = self._group(features, neighbor_indices)
        grouped = torch.cat((grouped_xyz, grouped_features), dim=-1).permute(0, 3, 2, 1)
        abstracted = torch.max(self.mlp(grouped), dim=2).values.transpose(1, 2)
        return centers, abstracted


class PointNetPlusPlusClassifier(nn.Module):
    """Compact PointNet++ variant for education and ablation experiments."""

    def __init__(self, class_count: int):
        super().__init__()
        self.sa1 = SetAbstractionKNN(128, 24, 3, 64)
        self.sa2 = SetAbstractionKNN(32, 32, 64, 128)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, class_count),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        features = xyz
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        return self.head(torch.max(features, dim=1).values)


class PatchAutoEncoder3D(nn.Module):
    """Small local-geometry autoencoder used as the PatchCore descriptor extractor.

    The decoder reconstructs a permutation-invariant local geometry signature
    instead of unordered point coordinates. This avoids the trivial zero-output
    collapse that occurs when a point-set encoder is paired with fixed-order XYZ
    reconstruction.
    """

    def __init__(self, patch_points: int, latent_dim: int = 24):
        super().__init__()
        self.patch_points = patch_points
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
        )

    def normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        centered = patches - patches.mean(dim=1, keepdim=True)
        scale = torch.sqrt(torch.mean(torch.sum(centered**2, dim=2), dim=1, keepdim=True))
        return centered / torch.clamp(scale[:, :, None], min=1e-6)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_patches(patches)
        local = self.encoder(normalized.transpose(1, 2))
        descriptor = torch.max(local, dim=2).values
        return nn.functional.normalize(descriptor, dim=1)

    @staticmethod
    def geometry_signature(normalized: torch.Tensor) -> torch.Tensor:
        covariance = torch.einsum("nki,nkj->nij", normalized, normalized)
        covariance /= normalized.shape[1]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        ratios = eigenvalues / eigenvalues.sum(dim=1, keepdim=True).clamp_min(1e-12)
        radial = torch.linalg.norm(normalized, dim=2)
        histograms = []
        for low, high in zip(
            torch.linspace(0.0, 2.0, 6)[:-1],
            torch.linspace(0.0, 2.0, 6)[1:],
            strict=True,
        ):
            histograms.append(((radial >= low) & (radial < high)).float().mean(dim=1))
        ordered, _ = torch.sort(normalized, dim=1)
        point_count = normalized.shape[1]
        quantile_indices = [
            int(q * (point_count - 1)) for q in (0.1, 0.5, 0.9)
        ]
        quantiles = torch.cat(
            [ordered[:, index, :] for index in quantile_indices], dim=1
        )
        return torch.cat(
            (
                ratios,
                normalized.std(dim=1),
                quantiles,
                torch.stack(histograms, dim=1),
            ),
            dim=1,
        )

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalize_patches(patches)
        descriptor = self.encode(patches)
        reconstruction = self.decoder(descriptor)
        target = self.geometry_signature(normalized)
        return reconstruction, target

