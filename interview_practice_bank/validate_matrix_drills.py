"""检查矩阵强化题数量、语法、中文步骤和逐行注释。"""

from __future__ import annotations

from pathlib import Path

from validate_bank import validate_file


ROOT = Path(__file__).parent / "matrix_operations"
EXPECTED = {"numpy": 21, "pytorch": 21}


def main() -> None:
    errors: list[str] = []
    total = 0
    for category, expected_count in EXPECTED.items():
        files = sorted((ROOT / category).glob("*.py"))
        total += len(files)
        if len(files) != expected_count:
            errors.append(f"{category}: expected {expected_count}, found {len(files)}")
        for path in files:
            errors.extend(validate_file(path))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"矩阵强化题结构、语法、中文步骤和逐行注释检查通过：{total} 题。")


if __name__ == "__main__":
    main()
