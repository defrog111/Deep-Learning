"""审计同一知识点的不同难度是否只是改了数字或字符串。

普通的语法检查只能证明脚本可运行，不能证明四个难度是四道不同的题。
本脚本会删除题目文档、print、assert、注释，并把常量归一化后比较 AST。
如果核心算法完全相同，则审计失败并列出对应文件。
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parent
GROUP_SIZES = {
    "pandas": 4,
    "numpy": 4,
    "pytorch": 4,
    "python": 4,
    "ml": 4,
    "sklearn": 2,
}


class NormalizeExercise(ast.NodeTransformer):
    """移除展示代码并归一化常量，保留真正的运算结构。"""

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:  # noqa: N802
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # 删除模块题目说明等字符串文档。
        if isinstance(node.value, ast.Call):
            function = node.value.func
            if isinstance(function, ast.Name) and function.id == "print":
                return None  # print差异不算不同算法。
        return self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:  # noqa: N802
        return None  # 只修改assert也不能算新题。

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        value = node.value
        if isinstance(value, bool) or value is None:
            return node
        if isinstance(value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value="NUMBER"), node)
        if isinstance(value, str):
            return ast.copy_location(ast.Constant(value="STRING"), node)
        return node


def core_signature(path: Path) -> str:
    """返回忽略表面变化后的核心代码签名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    normalized = NormalizeExercise().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def main() -> None:
    duplicate_groups: list[str] = []
    for category, group_size in GROUP_SIZES.items():
        files = sorted((ROOT / category).glob("*.py"))
        if len(files) % group_size:
            raise SystemExit(f"{category}: 文件数不能被每组题数 {group_size} 整除")
        for start in range(0, len(files), group_size):
            group = files[start : start + group_size]
            signatures = [core_signature(path) for path in group]
            if len(set(signatures)) != len(signatures):
                names = ", ".join(path.name for path in group)
                duplicate_groups.append(f"{category}: {names}")

    if duplicate_groups:
        details = "\n".join(duplicate_groups)
        raise SystemExit(
            "变式质量审计失败：以下同组题存在仅改常量或展示代码的情况：\n"
            f"{details}"
        )
    print("变式质量审计通过：每组基础、变式、易错点和综合题的核心代码均不同。")


if __name__ == "__main__":
    main()
