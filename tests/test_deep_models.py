import torch

from industrial_pc3d.deep_models import (
    PointNetClassifier,
    PointNetPlusPlusClassifier,
    PointNetSegmenter,
)


def test_pointnet_heads() -> None:
    points = torch.randn(2, 128, 3)
    classifier = PointNetClassifier(4)
    segmenter = PointNetSegmenter(3)
    pointnet_plus_plus = PointNetPlusPlusClassifier(5)
    assert classifier(points).shape == (2, 4)
    assert segmenter(points).shape == (2, 128, 3)
    assert pointnet_plus_plus(points).shape == (2, 5)

