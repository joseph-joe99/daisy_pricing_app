"""Row filters. Filters are pandas-eval expressions."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd


def _r_to_pandas(expr: str) -> str:
    """Translate the list-membership shorthand (%in% c(...)) users may type into pandas.eval syntax.

    ``x %in% c('a','b')`` -> ``x in ['a','b']``; ``&&``/``||`` -> ``&``/``|``.
    """
    expr = re.sub(r"%in%\s*c\(([^)]*)\)", r" in [\1]", expr)
    return expr.replace("&&", "&").replace("||", "|")


def combine_filters(
    filters: list[str],
    operation: str = "AND",
) -> str | None:
    """Join filter expressions with AND (or OR) into one pandas-eval expression. Returns None when there are no filters."""
    if not filters:
        return None
    op = " & " if operation in ("AND", "NAND") else " | "
    expr = filters[0] if len(filters) == 1 else op.join(f"({f})" for f in filters)
    if operation in ("NAND", "NOR"):
        expr = f"~({expr})"
    return expr


def apply_filter(
    d: pd.DataFrame,
    filter_expr: str | None,
    train_test: str = "All",
):
    """Return (user_filter, total_filter, message) as int8 arrays aligned to ``d``."""
    n = len(d)
    message = "no filter"
    user = np.ones(n, dtype=np.int8)
    if filter_expr and filter_expr.strip() not in ("", "no filter"):
        try:
            f = d.eval(_r_to_pandas(filter_expr), engine="python")
            f = pd.Series(f).fillna(False).astype(bool).to_numpy()
            user = f.astype(np.int8)
            message = filter_expr
        except Exception as e:  # bad filters fall back to no filter
            message = f"{type(e).__name__}: {e}"
    total = user.copy()
    if train_test in ("Train", "Test") and "train_test" in d.columns:
        tt = d["train_test"].to_numpy()
        total = (user * (1 - tt)).astype(np.int8) if train_test == "Train" else (user * tt).astype(np.int8)
    return user, total, message


def filter_text(total_filter: np.ndarray) -> str:
    """Label such as 'Filter (120/1,000)' showing how many rows the filter keeps."""
    return f"Filter ({int(np.nansum(total_filter)):,}/{len(total_filter):,})"
