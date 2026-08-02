"""从统一题目定义生成 42 个矩阵运算强化练习。"""

from __future__ import annotations

from pathlib import Path

from matrix_cases import NUMPY_MATRIX_CASES, PYTORCH_MATRIX_CASES


ROOT = Path(__file__).parent


def render(index: int, library: str, title: str, steps: list[str], code: list[str]) -> str:
    step_text = "\n".join(f"{number}. {step}" for number, step in enumerate(steps, 1))
    return f'''"""
题目 {index:03d}：{library} {title}

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
{step_text}

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

{chr(10).join(code)}
'''


def generate(folder_name: str, library: str, cases: list[tuple[str, list[str], list[str]]]) -> None:
    folder = ROOT / "matrix_operations" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    for index, (title, steps, code) in enumerate(cases, 1):
        path = folder / f"{index:03d}_{title}.py"
        path.write_text(render(index, library, title, steps, code), encoding="utf-8")


def main() -> None:
    generate("numpy", "NumPy", NUMPY_MATRIX_CASES)
    generate("pytorch", "PyTorch", PYTORCH_MATRIX_CASES)
    print("已生成 NumPy 21 题和 PyTorch 21 题矩阵运算强化练习。")


if __name__ == "__main__":
    main()
