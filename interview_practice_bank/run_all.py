"""批量执行题库，用于验证每道参考代码可运行。"""

from __future__ import annotations

import argparse
import contextlib
import io
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).parent
CATEGORIES = ("pandas", "numpy", "pytorch", "python", "ml", "sklearn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        choices=("all",) + CATEGORIES,
        default="all",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="显示每题print输出；默认静默，仅显示进度和失败。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.category == "all":
        started = time.perf_counter()
        for category in CATEGORIES:
            command = [
                sys.executable,
                str(Path(__file__)),
                "--category",
                category,
            ]
            if args.show_output:
                command.append("--show-output")
            subprocess.run(command, check=True)
        elapsed = time.perf_counter() - started
        print(f"全部 550 题分六个隔离进程运行成功，用时 {elapsed:.2f}s。")
        return
    categories = CATEGORIES if args.category == "all" else (args.category,)
    files = [
        path
        for category in categories
        for path in sorted((ROOT / category).glob("*.py"))
    ]
    failures: list[tuple[Path, str]] = []
    started = time.perf_counter()
    for index, path in enumerate(files, start=1):
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(
                sys.stdout if args.show_output else output
            ):
                runpy.run_path(str(path), run_name="__main__")
        except Exception:
            failures.append((path, traceback.format_exc()))
        if index % 25 == 0 or index == len(files):
            print(f"progress: {index}/{len(files)}")
    elapsed = time.perf_counter() - started
    if failures:
        for path, details in failures:
            print(f"\nFAILED: {path}\n{details}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} questions failed.")
    print(f"全部 {len(files)} 题运行成功，用时 {elapsed:.2f}s。")


if __name__ == "__main__":
    main()
