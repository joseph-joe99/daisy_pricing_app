"""Convert GLMs / GBMs to tabular 'ratebooks' (one-way relativities) and export to Excel.

Method: for every feature, build a grid of levels
(numeric features: min..max in ``banding`` steps from the feature spec; categorical: observed levels),
hold all other features at their *base level*, predict, and express each level relative to the base level.
This is exact for additive log-link GLMs and a standard approximation for GBMs.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from .specs import numeric_or_none


def _mode(s: pd.Series):
    """Most common value of a series, ignoring missing values."""
    return s.mode(dropna=True).iloc[0]


def _base_value(
    d: pd.DataFrame,
    feature: str,
    spec_row: dict,
):
    """Base level for a feature: the spec's base level when set, otherwise a default."""
    b = spec_row.get("base_level", "")
    s = d[feature]
    if b not in ("", None):
        if pd.api.types.is_numeric_dtype(s):
            v = numeric_or_none(b)
            if v is not None:
                return v
        else:
            return str(b)
    return float(s.median()) if pd.api.types.is_numeric_dtype(s) else _mode(s.astype(str))


def _grid(
    d: pd.DataFrame,
    feature: str,
    spec_row: dict,
    max_levels: int = 2000,
):
    """Grid of levels to tabulate for a feature: min to max in banding steps for numbers, observed levels for categories."""
    s = d[feature]
    if pd.api.types.is_numeric_dtype(s):
        lo = numeric_or_none(spec_row.get("min"))
        hi = numeric_or_none(spec_row.get("max"))
        step = numeric_or_none(spec_row.get("banding"))
        lo = s.min() if lo is None else lo
        hi = s.max() if hi is None else hi
        if not step or step <= 0:
            from .banding import banding_guesser

            step = banding_guesser(s)
        n = int(np.floor((hi - lo) / step)) + 1
        if n > max_levels:
            # too many levels: widen the bands by a whole multiple, so levels stay on the specified banding
            step = step * int(np.ceil(n / max_levels))
            n = int(np.floor((hi - lo) / step)) + 1
        return list(lo + step * np.arange(n))
    return sorted(s.astype(str).unique())


def check_ratebook_link(link: str | None) -> None:
    """A ratebook multiplies relativities together, which only reproduces a log-link model."""
    if link is not None and link != "log":
        raise ValueError(
            f"a ratebook multiplies relativities, so it needs a log-link model (this model's link is "
            f"'{link}'): use Poisson, quasi-Poisson, Gamma, Tweedie or Gaussian (log link)"
        )


def tabulate_model(
    d: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    features: list[str],
    feature_spec: pd.DataFrame,
    link: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return {feature: DataFrame(level, prediction, relativity)} plus 'base' info under key '__base__'.

    ``link`` (the model's link, when known) is checked: relativities are only meaningful for a log link.
    """
    check_ratebook_link(link)
    spec = {r["feature"]: r for r in feature_spec.to_dict("records")}
    bases = {f: _base_value(d, f, spec.get(f, {})) for f in features}
    base_row = d.iloc[[0]].copy()
    for f in features:
        base_row[f] = bases[f]
    base_pred = float(predict_fn(base_row)[0])
    out = {"__base__": pd.DataFrame({"feature": list(bases), "base_level": [str(v) for v in bases.values()]})}
    out["__base__"].attrs["base_prediction"] = base_pred
    for f in features:
        levels = _grid(d, f, spec.get(f, {}))
        grid = pd.concat([base_row] * len(levels), ignore_index=True)
        if pd.api.types.is_numeric_dtype(d[f]):
            grid[f] = np.asarray(levels, dtype=float).astype(d[f].dtype if pd.api.types.is_float_dtype(d[f]) else float)
        else:
            grid[f] = (
                pd.Categorical(levels, categories=levels) if isinstance(d[f].dtype, pd.CategoricalDtype) else levels
            )
        pred = predict_fn(grid)
        out[f] = pd.DataFrame({"level": levels, "prediction": pred, "relativity": pred / base_pred})
    return out


def write_ratebook_excel(
    tables: dict[str, pd.DataFrame],
    path: str,
    model_name: str = "model",
    weight_basis: str | None = None,
    weight: str = "N",
) -> str:
    """Write the ratebook workbook. ``weight_basis`` / ``weight`` (how the model used its weight) are recorded
    beside the base prediction so scoring can turn the per-unit rates back into totals."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    base = tables["__base__"]
    ws.append([f"{model_name} ratebook"])
    ws["A1"].font = Font(bold=True, size=13)
    base_row = ["Base prediction", base.attrs.get("base_prediction")]
    if weight_basis and weight != "N":
        base_row += ["Weight basis", weight_basis, "Weight", weight]
    ws.append(base_row)
    ws.append([])
    ws.append(["feature", "base_level"])
    for c in ws[4]:
        c.font = Font(bold=True)
    for r in base.itertuples(index=False):
        ws.append(list(r))
    for f, t in tables.items():
        if f == "__base__":
            continue
        w = wb.create_sheet(f[:31])
        w.append(["level", "prediction", "relativity"])
        for c in w[1]:
            c.font = Font(bold=True)
        for r in t.itertuples(index=False):
            w.append(
                [
                    r.level if not isinstance(r.level, (np.floating,)) else float(r.level),
                    float(r.prediction),
                    float(r.relativity),
                ]
            )
    wb.save(path)
    return path
