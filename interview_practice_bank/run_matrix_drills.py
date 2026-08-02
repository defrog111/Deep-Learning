"""逐个隔离执行 NumPy/PyTorch 矩阵运算强化题。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parent / "matrix_operations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        choices=("all", "numpy", "pytorch"),
        default="all",
    )
    parser.add_argument("--show-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = ("numpy", "pytorch") if args.category == "all" else (args.category,)
    files = [path for category in categories for path in sorted((ROOT / category).glob("*.py"))]
    started = time.perf_counter()
    for index, path in enumerate(files, 1):
        result = subprocess.run(
            [sys.executable, str(path)],
            text=True,
            capture_output=not args.show_output,
        )
        if result.returncode != 0:
            if not args.show_output:
                print(result.stdout, end="")
                print(result.stderr, end="", file=sys.stderr)
            raise SystemExit(f"运行失败：{path}")
        if index % 10 == 0 or index == len(files):
            print(f"progress: {index}/{len(files)}")
    elapsed = time.perf_counter() - started
    print(f"全部 {len(files)} 道矩阵强化题运行成功，用时 {elapsed:.2f}s。")


if __name__ == "__main__":
    main()
