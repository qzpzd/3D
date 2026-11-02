from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass
class PointCloudData:
    xyz: np.ndarray
    color: np.ndarray | None = None
    normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {self.xyz.shape}")
        if self.color is not None:
            self.color = np.asarray(self.color)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)


_PCD_DTYPES = {
    ("F", 4): "f4",
    ("F", 8): "f8",
    ("I", 1): "i1",
    ("I", 2): "i2",
    ("I", 4): "i4",
    ("I", 8): "i8",
    ("U", 1): "u1",
    ("U", 2): "u2",
    ("U", 4): "u4",
    ("U", 8): "u8",
}


def _header_value(header: dict[str, list[str]], key: str) -> list[str] | None:
    return header.get(key.upper())


def _decode_packed_rgb(values: np.ndarray) -> np.ndarray:
    packed = np.asarray(values).copy().reshape(-1)
    if packed.dtype.kind == "f":
        packed = packed.astype(np.float32).view(np.uint32)
    else:
        packed = packed.astype(np.uint32)
    return np.column_stack(
        ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255)
    ).astype(np.uint8)


def _pcd_structured_data(
    fields: list[str], sizes: list[int], types: list[str], counts: list[int]
) -> np.ndarray:
    dtype_fields: list[tuple[str, object]] = []
    for name, size, kind, count in zip(fields, sizes, types, counts, strict=True):
        base = _PCD_DTYPES.get((kind, size))
        if base is None:
            raise ValueError(f"Unsupported PCD field type: {kind}{size}")
        dtype_fields.append((name, f"<{base}", (count,)) if count > 1 else (name, f"<{base}"))
    return np.dtype(dtype_fields)


def _read_pcd(path: Path) -> PointCloudData:
    header: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PCD header has no DATA line")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                data_mode = values[0].lower()
                break

        fields = _header_value(header, "FIELDS") or []
        sizes = [int(v) for v in (_header_value(header, "SIZE") or [])]
        types = _header_value(header, "TYPE") or []
        counts = [int(v) for v in (_header_value(header, "COUNT") or ["1"] * len(fields))]
        point_count = int((_header_value(header, "POINTS") or ["-1"])[0])
        if not fields or len(sizes) != len(fields) or len(types) != len(fields):
            raise ValueError("Malformed PCD header")

        if data_mode == "ascii":
            payload = stream.read().decode("ascii")
            table = np.loadtxt(payload.splitlines(), dtype=np.float64)
            table = np.asarray(table)
            if table.ndim == 1:
                table = table.reshape(1, -1)
            if table.shape[0] != point_count:
                raise ValueError(f"PCD POINTS={point_count}, read {table.shape[0]}")
            record: dict[str, np.ndarray] = {}
            cursor = 0
            for name, count in zip(fields, counts, strict=True):
                record[name] = table[:, cursor : cursor + count]
                cursor += count
        elif data_mode == "binary":
            if point_count < 0:
                raise ValueError("Binary PCD requires POINTS")
            dtype = _pcd_structured_data(fields, sizes, types, counts)
            raw = stream.read(point_count * dtype.itemsize)
            if len(raw) != point_count * dtype.itemsize:
                raise ValueError("Truncated binary PCD payload")
            table = np.frombuffer(raw, dtype=dtype, count=point_count)
            record = {name: np.asarray(table[name]) for name in fields}
        else:
            raise ValueError(f"Unsupported PCD DATA mode: {data_mode}")

    xyz = np.column_stack((record["x"], record["y"], record["z"])).astype(np.float64)
    color = None
    if "rgb" in record:
        color = _decode_packed_rgb(record["rgb"])
    elif {"r", "g", "b"}.issubset(record):
        color = np.column_stack((record["r"], record["g"], record["b"])).astype(np.uint8)
    normals = None
    if {"normal_x", "normal_y", "normal_z"}.issubset(record):
        normals = np.column_stack(
            (record["normal_x"], record["normal_y"], record["normal_z"])
        ).astype(np.float64)
    elif {"nx", "ny", "nz"}.issubset(record):
        normals = np.column_stack((record["nx"], record["ny"], record["nz"])).astype(
            np.float64
        )
    return PointCloudData(xyz, color=color, normals=normals)


def _ply_scalar_dtype(name: str) -> str:
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY scalar: {name}") from exc


def _read_ply(path: Path) -> PointCloudData:
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        fmt = next((x for x in header_lines if x.startswith("format ")), "")
        if "ascii" not in fmt:
            raise ValueError("This dependency-free reader supports ASCII PLY only")
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for text in header_lines:
            tokens = text.split()
            if len(tokens) >= 2 and tokens[0] == "element":
                in_vertex = tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
            elif in_vertex and tokens and tokens[0] == "property":
                if len(tokens) != 3 or tokens[1] == "list":
                    raise ValueError("PLY vertex list properties are not supported")
                properties.append((tokens[2], _ply_scalar_dtype(tokens[1])))
        payload = stream.read().decode("ascii").splitlines()[:vertex_count]
        table = np.loadtxt(payload, dtype=np.float64)
        if table.ndim == 1:
            table = table.reshape(1, -1)
        columns = {name: table[:, i] for i, (name, _) in enumerate(properties)}

    xyz = np.column_stack((columns["x"], columns["y"], columns["z"]))
    color = None
    if {"red", "green", "blue"}.issubset(columns):
        color = np.column_stack(
            (columns["red"], columns["green"], columns["blue"])
        ).astype(np.uint8)
    normals = None
    if {"nx", "ny", "nz"}.issubset(columns):
        normals = np.column_stack((columns["nx"], columns["ny"], columns["nz"]))
    return PointCloudData(xyz, color=color, normals=normals)


def load_point_cloud(path: str | Path) -> PointCloudData:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pcd":
        return _read_pcd(source)
    if suffix == ".ply":
        return _read_ply(source)
    if suffix == ".npy":
        return PointCloudData(np.load(source))
    if suffix == ".npz":
        archive = np.load(source)
        return PointCloudData(
            archive["xyz"],
            color=archive["color"] if "color" in archive else None,
            normals=archive["normals"] if "normals" in archive else None,
        )
    if suffix in {".xyz", ".csv"}:
        delim = "," if suffix == ".csv" else None
        table = np.loadtxt(source, delimiter=delim)
        if table.ndim == 1:
            table = table.reshape(1, -1)
        return PointCloudData(table[:, :3])
    raise ValueError(f"Unsupported point-cloud format: {suffix}")


def save_pcd(path: str | Path, cloud: PointCloudData | np.ndarray) -> None:
    if isinstance(cloud, np.ndarray):
        cloud = PointCloudData(cloud)
    xyz = cloud.xyz
    fields = ["x", "y", "z"]
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(fields)}",
        "SIZE 8 8 8",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(xyz)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(xyz)}",
        "DATA ascii",
    ]
    np.savetxt(path, xyz, fmt="%.9g", header="\n".join(header), comments="")


def save_npz(
    path: str | Path,
    xyz: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {"xyz": np.asarray(xyz)}
    if labels is not None:
        payload["labels"] = np.asarray(labels)
    if scores is not None:
        payload["scores"] = np.asarray(scores)
    np.savez_compressed(path, **payload)


def save_point_cloud(path: str | Path, cloud: PointCloudData | np.ndarray) -> None:
    target = Path(path)
    if target.suffix.lower() == ".pcd":
        save_pcd(target, cloud)
        return
    if target.suffix.lower() == ".npy":
        xyz = cloud.xyz if isinstance(cloud, PointCloudData) else np.asarray(cloud)
        np.save(target, xyz)
        return
    if target.suffix.lower() == ".npz":
        xyz = cloud.xyz if isinstance(cloud, PointCloudData) else np.asarray(cloud)
        save_npz(target, xyz)
        return
    raise ValueError("Save supports .pcd, .npy, and .npz")

