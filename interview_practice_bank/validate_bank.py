"""检查题目数量、语法以及参考代码逐行注释。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parent
EXPECTED = {
    "pandas": 100,
    "numpy": 100,
    "pytorch": 100,
    "python": 100,
    "ml": 100,
    "sklearn": 50,
}


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as error:
        errors.append(f"{path}: syntax error: {error}")
        return errors

    in_docstring = False
    docstring_finished = False
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not docstring_finished and stripped == '"""':
            in_docstring = not in_docstring
            if not in_docstring:
                docstring_finished = True
            continue
        if in_docstring or not stripped:
            continue
        if "#" not in line:
            errors.append(f"{path}:{line_number}: 缺少逐行注释")
    return errors


def main() -> None:
    errors: list[str] = []
    total = 0
    for category, expected_count in EXPECTED.items():
        files = sorted((ROOT / category).glob("*.py"))
        total += len(files)
        if len(files) != expected_count:
            errors.append(
                f"{category}: expected {expected_count}, found {len(files)}"
            )
        for path in files:
            errors.extend(validate_file(path))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"题库结构、语法和逐行注释检查通过：{total} 题。")


if __name__ == "__main__":
    main()
