"""题库生成器共享的依赖提取工具。"""

from __future__ import annotations

import ast
import textwrap


def _names(code_lines: list[str]) -> tuple[set[str], set[str]]:
    source = textwrap.dedent("\n".join(code_lines))
    if len(code_lines) == 1 and source.rstrip().endswith(":"):
        source += "\n    pass"
    try:
        tree = ast.parse(source)
    except (IndentationError, SyntaxError):
        return set(), set()
    loaded = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    defined = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    defined.update(
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    defined.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
    return loaded, defined


def independent_variant(base_code: list[str], extra_code: list[str]) -> list[str]:
    """只提取进阶题真正依赖的基础准备语句，不复制基础题完整算法。"""
    extra_loaded, extra_defined = _names(extra_code)
    needed = extra_loaded - extra_defined
    source = "\n".join(base_code)
    tree = ast.parse(source)
    entry_ranges: list[tuple[int, int]] = []
    next_line = 1
    for entry in base_code:
        line_count = entry.count("\n") + 1
        entry_ranges.append((next_line, next_line + line_count - 1))
        next_line += line_count

    selected_entries: set[int] = set()
    for statement in reversed(tree.body):
        statement_source = ast.get_source_segment(source, statement) or ""
        loaded, defined = _names([statement_source])
        is_import = isinstance(statement, (ast.Import, ast.ImportFrom))
        if is_import or defined & needed:
            needed.difference_update(defined)
            needed.update(loaded - defined)
            for index, (start, end) in enumerate(entry_ranges):
                if start <= statement.end_lineno and end >= statement.lineno:
                    selected_entries.add(index)

    selected = [base_code[index] for index in sorted(selected_entries)]
    return selected + list(extra_code)
