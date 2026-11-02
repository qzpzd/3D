from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import PipelineResult


def save_pipeline_preview(result: PipelineResult, path: str | Path) -> None:
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)

    ax = fig.add_subplot(131, projection="3d")
    stride = max(1, len(result.raw) // 25000)
    ax.scatter(
        result.raw[::stride, 0],
        result.raw[::stride, 1],
        result.raw[::stride, 2],
        s=0.5,
        c="0.65",
        linewidths=0,
    )
    ax.set_title(f"Raw scene ({len(result.raw):,} pts)")
    ax.set_box_aspect((1, 1, 0.45))

    ax = fig.add_subplot(132, projection="3d")
    valid = result.cluster_labels >= 0
    ax.scatter(
        result.without_plane[valid, 0],
        result.without_plane[valid, 1],
        result.without_plane[valid, 2],
        c=result.cluster_labels[valid],
        cmap="tab10",
        s=2,
        linewidths=0,
    )
    ax.set_title(f"Clusters after plane removal ({result.summary['counts']['clusters']})")
    ax.set_box_aspect((1, 1, 0.6))

    ax = fig.add_subplot(133, projection="3d")
    ax.scatter(*result.target.T, s=2, c="0.55", linewidths=0)
    if len(result.target_edges):
        ax.scatter(*result.target_edges.T, s=4, c="tab:red", linewidths=0)
    ax.set_title(
        f"Selected part ({len(result.target):,}) / edges ({len(result.target_edges):,})"
    )
    ax.set_box_aspect(tuple(result.obb.extents / max(result.obb.extents)))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.08)
    fig.savefig(path, dpi=180)
    plt.close(fig)

