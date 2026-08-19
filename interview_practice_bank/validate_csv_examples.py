"""验证并运行全部CSV数据处理专项练习。"""

from __future__ import annotations

import ast
import contextlib
import io
import runpy
import traceback
import warnings
from pathlib import Path


ROOT = Path(__file__).parent / "csv_data_processing"
EXPECTED_COUNT = 76


def has_read_csv(tree: ast.AST) -> bool:
    """确认代码中真实调用了pd.read_csv。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "pd" and node.func.attr == "read_csv":
            return True
    return False


def comment_errors(path: Path, source: str) -> list[str]:
    """检查题目说明之后每一行参考代码都有中文解释注释。"""
    errors: list[str] = []
    in_docstring = False
    docstring_finished = False
    for line_number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not docstring_finished and stripped == '"""':
            in_docstring = not in_docstring
            if not in_docstring:
                docstring_finished = True
            continue
        if in_docstring or not stripped:
            continue
        if "#" not in line:
            errors.append(f"{path}:{line_number}: 参考代码缺少逐行注释")
    return errors


def main() -> None:
    """执行结构检查和严格运行检查。"""
    files = sorted(ROOT.glob("*.py"))
    errors: list[str] = []
    if len(files) != EXPECTED_COUNT:
        errors.append(f"expected {EXPECTED_COUNT} files, found {len(files)}")

    warnings.simplefilter("error")
    for index, path in enumerate(files, 1):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            errors.append(f"{path}: syntax error: {error}")
            continue
        if "操作过程：" not in source or "完成标准：" not in source:
            errors.append(f"{path}: 缺少中文操作过程或完成标准")
        if not has_read_csv(tree):
            errors.append(f"{path}: 没有真实调用pd.read_csv")
        errors.extend(comment_errors(path, source))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(path), run_name="__main__")
        except Exception:
            errors.append(f"{path}: 运行失败\n{traceback.format_exc()}")
        if index % 10 == 0 or index == len(files):
            print(f"progress: {index}/{len(files)}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"全部 {len(files)} 个CSV练习通过语法、read_csv、逐行注释和严格运行检查。")


if __name__ == "__main__":
    main()
