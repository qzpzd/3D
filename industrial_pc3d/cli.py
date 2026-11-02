from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import apply_transform, rigid_transform
from .io import load_point_cloud, save_npz, save_pcd
from .patchcore3d import PatchCore3D, PatchCore3DConfig
from .pipeline import TraditionalConfig, TraditionalPipeline
from .registration import global_registration, refine_icp
from .synthetic import make_deep_sample, make_industrial_scene
from .visualization import save_pipeline_preview


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _run_traditional(
    points: np.ndarray,
    config: TraditionalConfig,
    output: Path,
    extra_summary: dict[str, object] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result = TraditionalPipeline(config).process(points)
    saved_target = result.target
    save_pcd(output / "scene_raw.pcd", points)
    save_pcd(output / "scene_roi_denoised.pcd", result.denoised)
    save_pcd(output / "parts_without_plane.pcd", result.without_plane)
    save_pcd(output / "target.pcd", saved_target)
    save_npz(
        output / "target_analysis.npz",
        saved_target,
        labels=result.variation,
        scores=(result.variation >= np.quantile(result.variation, config.edge_quantile)).astype(
            np.uint8
        ),
    )
    save_npz(output / "target_edges.npz", result.target_edges)
    save_pipeline_preview(result, output / "pipeline_preview.png")
    summary = dict(result.summary)
    if extra_summary:
        summary["synthetic_ground_truth"] = extra_summary
    _write_json(output / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_demo(args: argparse.Namespace) -> None:
    output = Path(args.output)
    points, ground_truth = make_industrial_scene(args.seed)
    config = TraditionalConfig.from_json(args.config)
    _run_traditional(points, config, output, ground_truth)


def command_process(args: argparse.Namespace) -> None:
    cloud = load_point_cloud(args.input)
    config = TraditionalConfig.from_json(args.config)
    _run_traditional(cloud.xyz, config, Path(args.output))


def command_process_open3d(args: argparse.Namespace) -> None:
    from .open3d_backend import process_open3d

    config = TraditionalConfig.from_json(args.config)
    summary = process_open3d(args.input, config, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


def command_align(args: argparse.Namespace) -> None:
    source = load_point_cloud(args.source).xyz
    target = load_point_cloud(args.target).xyz
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    initial = global_registration(
        source,
        target,
        voxel_size=args.voxel_size,
        ransac_iterations=args.ransac_iterations,
        seed=args.seed,
    )
    refined = refine_icp(
        source,
        target,
        initial=initial,
        distance_threshold=args.icp_distance,
    )
    aligned = apply_transform(source, refined.rotation, refined.translation)
    save_pcd(output / "source_aligned.pcd", aligned)
    np.save(output / "transform_4x4.npy", refined.transformation)
    summary = {
        "global": {
            "fitness": initial.fitness,
            "inlier_rmse": initial.inlier_rmse,
            "feature_matches": initial.feature_matches,
        },
        "refined": {
            "fitness": refined.fitness,
            "inlier_rmse": refined.inlier_rmse,
            "feature_matches": refined.feature_matches,
        },
        "transform_4x4": refined.transformation.tolist(),
    }
    _write_json(output / "registration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


def command_align_open3d(args: argparse.Namespace) -> None:
    from .open3d_backend import align_open3d

    summary = align_open3d(
        args.source,
        args.target,
        args.output,
        voxel_size=args.voxel_size,
        ransac_iterations=args.ransac_iterations,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


def command_registration_demo(args: argparse.Namespace) -> None:
    target = make_deep_sample(args.seed, "none", 5200).xyz
    angle = math.radians(13.0)
    truth_rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    truth_translation = np.array([0.31, -0.21, 0.07])
    source = apply_transform(target, truth_rotation, truth_translation)
    rng = np.random.default_rng(args.seed)
    source += rng.normal(0.0, 0.00045, source.shape)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_pcd(output / "source_misaligned.pcd", source)
    save_pcd(output / "target_template.pcd", target)
    initial = global_registration(
        source,
        target,
        voxel_size=args.voxel_size,
        ransac_iterations=args.ransac_iterations,
        seed=args.seed,
    )
    refined = refine_icp(
        source,
        target,
        initial=initial,
        distance_threshold=args.icp_distance,
    )
    aligned = apply_transform(source, refined.rotation, refined.translation)
    save_pcd(output / "source_aligned.pcd", aligned)
    np.save(output / "transform_4x4.npy", refined.transformation)

    inverse_truth_rotation = truth_rotation.T
    inverse_truth_translation = -truth_rotation.T @ truth_translation
    delta = refined.rotation @ inverse_truth_rotation.T
    rotation_error = math.degrees(
        math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
    )
    translation_error = inverse_truth_translation - refined.translation
    summary = {
        "global_fitness": initial.fitness,
        "refined_fitness": refined.fitness,
        "refined_inlier_rmse": refined.inlier_rmse,
        "rotation_error_deg_vs_inverse_truth": rotation_error,
        "translation_error_norm": float(np.linalg.norm(translation_error)),
        "note": "Synthetic metric evaluates convergence, not scanner acceptance",
    }
    _write_json(output / "registration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _save_anomaly_preview(
    points: np.ndarray, scores: np.ndarray, threshold: float, path: Path
) -> None:
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    order = np.argsort(scores)
    scatter = ax.scatter(
        points[order, 0],
        points[order, 1],
        points[order, 2],
        c=scores[order],
        cmap="inferno",
        s=4,
        linewidths=0,
        vmin=float(np.min(scores)),
        vmax=float(max(np.max(scores), threshold * 1.01)),
    )
    fig.colorbar(scatter, ax=ax, shrink=0.65, label="Patch distance")
    ax.set_title(f"PatchCore-3D anomaly score; calibrated threshold={threshold:.4f}")
    ax.set_box_aspect((1.3, 1.0, 0.6))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def command_deep_demo(args: argparse.Namespace) -> None:
    config = PatchCore3DConfig.from_json(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    config.max_patches_per_cloud = min(config.max_patches_per_cloud, args.max_patches)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train = [make_deep_sample(1000 + i, "none").xyz for i in range(args.train_samples)]
    validation = [
        make_deep_sample(2000 + i, "none").xyz for i in range(args.validation_samples)
    ]
    detector = PatchCore3D(config)
    detector.fit(train)
    detector.calibrate(validation)
    detector.save(output / "patchcore3d.pt")

    normal_tests = [
        make_deep_sample(3000 + i, "none") for i in range(args.normal_tests)
    ]
    defect_tests = [
        make_deep_sample(4000 + i, defect)
        for i, defect in enumerate(["dent", "bump"] * (args.defect_tests // 2))
    ]
    all_tests = normal_tests + defect_tests
    predicted_labels: list[int] = []
    actual_labels: list[int] = []
    point_scores: list[np.ndarray] = []
    point_labels: list[np.ndarray] = []
    sample_rows: list[dict[str, object]] = []
    thresholds = detector.object_threshold or 0.0
    for sample in all_tests:
        scored = detector.score(sample.xyz)
        predicted = int(float(scored["object_score"]) > thresholds)
        predicted_labels.append(predicted)
        actual_labels.append(sample.object_label)
        point_scores.append(scored["point_scores"])
        point_labels.append(sample.anomaly_mask)
        sample_rows.append(
            {
                "defect_type": sample.defect_type,
                "object_score": scored["object_score"],
                "predicted": predicted,
                "actual": sample.object_label,
            }
        )

    from sklearn.metrics import roc_auc_score

    point_auc = (
        roc_auc_score(np.concatenate(point_labels), np.concatenate(point_scores))
        if np.unique(np.concatenate(point_labels)).size == 2
        else None
    )
    top_sample = max(defect_tests, key=lambda item: detector.score(item.xyz)["object_score"])
    top_scored = detector.score(top_sample.xyz)
    _save_anomaly_preview(
        top_sample.xyz,
        top_scored["point_scores"],
        detector.point_threshold or 0.0,
        output / "anomaly_preview.png",
    )
    summary = {
        "method": "from-scratch 3D PatchCore-style local autoencoder + coreset memory",
        "point_threshold": detector.point_threshold,
        "object_threshold": detector.object_threshold,
        "training_loss": detector.training_history,
        "object_accuracy": float(
            np.mean(np.asarray(predicted_labels) == np.asarray(actual_labels))
        ),
        "object_recall": float(
            np.sum((np.asarray(predicted_labels) == 1) & (np.asarray(actual_labels) == 1))
            / max(1, np.sum(np.asarray(actual_labels) == 1))
        ),
        "normal_false_positive_rate": float(
            np.sum((np.asarray(predicted_labels) == 1) & (np.asarray(actual_labels) == 0))
            / max(1, np.sum(np.asarray(actual_labels) == 0))
        ),
        "point_auroc": point_auc,
        "samples": sample_rows,
        "warning": "Educational synthetic benchmark; not a production acceptance result.",
    }
    _write_json(output / "evaluation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Industrial 3D point-cloud traditional/deep baselines"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the full synthetic industrial pipeline")
    demo.add_argument("--config", default="configs/traditional.json")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=command_demo)

    process = subparsers.add_parser("process", help="Process a PCD/PLY/XYZ/NPZ file")
    process.add_argument("--input", required=True)
    process.add_argument("--config", default="configs/traditional.json")
    process.add_argument("--output", default="outputs/process")
    process.set_defaults(func=command_process)

    process_open3d = subparsers.add_parser(
        "process-open3d", help="Process a file with the Open3D backend"
    )
    process_open3d.add_argument("--input", required=True)
    process_open3d.add_argument("--config", default="configs/traditional.json")
    process_open3d.add_argument("--output", default="outputs/open3d_process")
    process_open3d.set_defaults(func=command_process_open3d)

    align = subparsers.add_parser("align", help="Register source to a template")
    align.add_argument("--source", required=True)
    align.add_argument("--target", required=True)
    align.add_argument("--output", default="outputs/align")
    align.add_argument("--voxel-size", type=float, default=0.015)
    align.add_argument("--ransac-iterations", type=int, default=500)
    align.add_argument("--icp-distance", type=float, default=0.03)
    align.add_argument("--seed", type=int, default=7)
    align.set_defaults(func=command_align)

    align_open3d = subparsers.add_parser(
        "align-open3d", help="Run Open3D FPFH-RANSAC + point-to-plane ICP"
    )
    align_open3d.add_argument("--source", required=True)
    align_open3d.add_argument("--target", required=True)
    align_open3d.add_argument("--output", default="outputs/open3d_align")
    align_open3d.add_argument("--voxel-size", type=float, default=0.015)
    align_open3d.add_argument("--ransac-iterations", type=int, default=100000)
    align_open3d.set_defaults(func=command_align_open3d)

    registration_demo = subparsers.add_parser(
        "registration-demo", help="Test synthetic FPFH-RANSAC + ICP recovery"
    )
    registration_demo.add_argument("--output", default="outputs/registration_demo")
    registration_demo.add_argument("--voxel-size", type=float, default=0.015)
    registration_demo.add_argument("--ransac-iterations", type=int, default=500)
    registration_demo.add_argument("--icp-distance", type=float, default=0.025)
    registration_demo.add_argument("--seed", type=int, default=7)
    registration_demo.set_defaults(func=command_registration_demo)

    deep = subparsers.add_parser(
        "deep-demo", help="Train and evaluate PatchCore-3D on synthetic dents/bumps"
    )
    deep.add_argument("--config", default="configs/patchcore3d.json")
    deep.add_argument("--output", default="outputs/deep_demo")
    deep.add_argument("--epochs", type=int)
    deep.add_argument("--max-patches", type=int, default=260)
    deep.add_argument("--train-samples", type=int, default=16)
    deep.add_argument("--validation-samples", type=int, default=5)
    deep.add_argument("--normal-tests", type=int, default=5)
    deep.add_argument("--defect-tests", type=int, default=10)
    deep.set_defaults(func=command_deep_demo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

