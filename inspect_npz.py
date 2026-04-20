from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


# 直接在这里改要查看的 npz 文件。
# 单文件模式：把 FILE_B 设为 None。
# 对比模式：给 FILE_B 赋第二个 npz 路径。
FILE_A = Path("results/biaozhundandao_unlimited.npz")
FILE_B = Path("results/fault_replay_case.npz")
""" FILE_B = None """

def _fmt_shape(arr: np.ndarray) -> str:
    return "x".join(str(v) for v in arr.shape) if arr.shape else "scalar"


def _basic_stats(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    if np.issubdtype(arr.dtype, np.number):
        amin = np.nanmin(arr)
        amax = np.nanmax(arr)
        return f"min={amin:.6g}, max={amax:.6g}"
    return "non-numeric"


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _print_single(npz_path: Path) -> None:
    npz_path = _resolve_path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"file not found: {npz_path}")

    data = np.load(npz_path)
    print(f"File: {npz_path}")
    print(f"Keys: {len(data.files)}")
    for key in data.files:
        arr = data[key]
        print(
            f"- {key}: shape={arr.shape} ({_fmt_shape(arr)}), dtype={arr.dtype}, { _basic_stats(arr) }"
        )


def _print_compare(path_a: Path, path_b: Path) -> None:
    path_a = _resolve_path(path_a)
    path_b = _resolve_path(path_b)
    if not path_a.exists():
        raise FileNotFoundError(f"file not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"file not found: {path_b}")

    a = np.load(path_a)
    b = np.load(path_b)

    keys = sorted(set(a.files) | set(b.files))
    print(f"Compare:\n  A={path_a}\n  B={path_b}")
    print(f"Total keys: {len(keys)}")

    for key in keys:
        in_a = key in a.files
        in_b = key in b.files

        if not in_a:
            print(f"- {key}: only in B")
            continue
        if not in_b:
            print(f"- {key}: only in A")
            continue

        arr_a = a[key]
        arr_b = b[key]
        same_shape = arr_a.shape == arr_b.shape
        shape_note = "same" if same_shape else "DIFF"
        line = (
            f"- {key}: A={arr_a.shape}, B={arr_b.shape}, shape={shape_note}, "
            f"dtype A/B={arr_a.dtype}/{arr_b.dtype}"
        )

        if same_shape and np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number):
            max_abs = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
            line += f", max|A-B|={max_abs:.6g}"

        print(line)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect .npz arrays (keys, shapes, dtypes, and optional pairwise comparison)."
    )
    parser.add_argument("file_a", type=Path, nargs="?", default=None, help="First .npz file path")
    parser.add_argument(
        "file_b",
        type=Path,
        nargs="?",
        default=None,
        help="Optional second .npz file path for comparison",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # 允许直接点“运行 Python 文件”：无命令行参数时直接读取上面的 FILE_A / FILE_B。
    if args.file_a is None:
        if FILE_B is not None:
            print(f"Auto compare files:\n  A={_resolve_path(FILE_A)}\n  B={_resolve_path(FILE_B)}")
            _print_compare(FILE_A, FILE_B)
            return
        print(f"Auto inspect file: {_resolve_path(FILE_A)}")
        _print_single(FILE_A)
        return

    if args.file_b is None:
        _print_single(args.file_a)
        return

    _print_compare(args.file_a, args.file_b)


if __name__ == "__main__":
    main()
