"""Adjustment overlays: band-level relativities applied on top of a base model's predictions.

For one feature, the rows are grouped into bands; each band's actual / expected (A/E) is bootstrapped to get
90%, 95% and 99% ranges. A band is adjusted when its 99% range lies entirely above or entirely below 1, flagged for
review when only its 95% or 90% range does, and otherwise kept at 1. Pure logic (no Shiny), used by the Overlays tab.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import numpy as np
import pandas as pd

LEVELS = {"90%": (5.0, 95.0), "95%": (2.5, 97.5), "99%": (0.5, 99.5)}  # central intervals (percentiles)
RULE = (
    "Adjust when the 99% range lies entirely above or below 1 (it does not contain 1); review when only the 95% "
    "or 90% range does not contain 1; otherwise keep the relativity at 1."
)


def category_levels(values) -> list[str]:
    """The levels of a categorical feature in the training rows (sorted; blanks left out)."""
    s = pd.Series(values)
    return sorted({str(v) for v in s[s.notna()]})


def band_starts(
    values,
    banding: float,
    levels=None,
    minimum=None,
    maximum=None,
) -> np.ndarray:
    """The band each value falls in: the start of its numeric band, or for a categorical feature the index of its
    level in `levels` (NaN for blank, unseen or non-numeric values). For a numeric feature, values below ``minimum``
    and above ``maximum`` are capped first, so they join the first and last bands as they do in Charts."""
    if levels is not None:
        lookup = {str(level): float(i) for i, level in enumerate(levels)}
        return np.array([lookup.get(str(v), np.nan) if pd.notna(v) else np.nan for v in values], dtype=float)
    width = float(banding)
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if minimum is not None or maximum is not None:
        x = np.clip(x, -np.inf if minimum is None else float(minimum), np.inf if maximum is None else float(maximum))
    return np.where(np.isfinite(x) & (width > 0), np.floor(x / width) * width, np.nan)


def band_label(
    start: float,
    banding: float,
    levels=None,
) -> str:
    """e.g. [20,25) for a band starting at 20 with banding 5; the level's name for a categorical feature."""
    if levels is not None:
        i = int(start) if np.isfinite(start) else -1
        return str(levels[i]) if 0 <= i < len(levels) else f"level {start:g}"
    return f"[{start:g},{start + banding:g})"


def band_table(
    actual,
    expected,
    keys,
) -> pd.DataFrame:
    """Rows, actual total, expected total and A/E for each band (rows with a missing value are left out)."""
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    k = np.asarray(keys, dtype=float)
    keep = np.isfinite(a) & np.isfinite(e) & np.isfinite(k)
    frame = pd.DataFrame({"band": k[keep], "A": a[keep], "E": e[keep]})
    table = frame.groupby("band").agg(n=("A", "size"), A=("A", "sum"), E=("E", "sum"))
    table["AE"] = table["A"] / table["E"].where(table["E"] != 0)
    return table


def bootstrap_intervals(
    actual,
    expected,
    keys,
    samples: int = 500,
    seed: int = 1,
) -> pd.DataFrame:
    """The 90%, 95% and 99% intervals of each band's A/E, from resampling the band's rows with replacement."""
    rng = np.random.default_rng(seed)
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    k = np.asarray(keys, dtype=float)
    keep = np.isfinite(a) & np.isfinite(e) & np.isfinite(k)
    rows = []
    for band in np.unique(k[keep]):
        idx = np.flatnonzero(keep & (k == band))
        draws = rng.integers(0, len(idx), size=(samples, len(idx)))
        ratio = a[idx][draws].sum(axis=1) / e[idx][draws].sum(axis=1)
        record = {"band": band}
        for level, (low, high) in LEVELS.items():
            tag = level.rstrip("%")
            record[f"lo{tag}"], record[f"hi{tag}"] = np.percentile(ratio, [low, high])
        rows.append(record)
    return pd.DataFrame(rows).set_index("band")


def suggest(
    table: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Status and suggested relativity of each band under the rule in RULE."""
    out = table.join(intervals, how="left")
    excl = {tag: (out[f"lo{tag}"] > 1) | (out[f"hi{tag}"] < 1) for tag in ("99", "95", "90")}
    status = np.where(excl["99"], "Adjust", np.where(excl["95"] | excl["90"], "Review", "Keep at 1"))
    out["status"] = status
    # a band with no actual or no expected (A/E of 0 or undefined) keeps a relativity of 1, never 0
    usable = np.isfinite(out["AE"]) & (out["AE"] > 0)
    out["relativity"] = np.where((out["status"] == "Adjust") & usable, out["AE"], 1.0)
    return out


@dataclass
class Overlay:
    """A saved overlay: the band relativities for one feature of one base model."""

    name: str
    base_name: str
    feature: str
    banding: float
    relativities: dict = field(default_factory=dict)  # band start -> relativity
    training_rows: int = 0
    rule: str = RULE
    levels: list = field(default_factory=list)  # the levels of a categorical feature (empty when numeric)
    minimum: float | None = None  # the specification's Min and Max, which cap the bands of a numeric feature
    maximum: float | None = None

    def band_keys(
        self,
        values,
    ) -> np.ndarray:
        """The band of each value of this overlay's feature."""
        return band_starts(values, self.banding, self.levels or None, self.minimum, self.maximum)

    def relativity_for(
        self,
        keys,
    ) -> np.ndarray:
        """The relativity of each row's band; blank, unseen or unlisted bands use 1."""
        k = np.asarray(keys, dtype=float)
        out = np.ones(len(k))
        for start, value in self.relativities.items():
            out[k == start] = value
        return out


def apply_overlay(
    overlay: Overlay,
    data: pd.DataFrame,
    expected,
) -> np.ndarray:
    """The base predictions multiplied by the relativity of each row's band of the overlay's feature."""
    keys = overlay.band_keys(data[overlay.feature])
    return np.asarray(expected, dtype=float) * overlay.relativity_for(keys)


def apply_overlays(
    overlays,
    data: pd.DataFrame,
    expected,
) -> np.ndarray:
    """The base predictions with several overlays applied one after another (their relativities multiply)."""
    out = np.asarray(expected, dtype=float)
    for overlay in overlays:
        out = apply_overlay(overlay, data, out)
    return out


def check_relativities(relativities: dict) -> list[str]:
    """Problems with the relativities to save (empty when they are fine)."""
    problems = []
    for start, value in relativities.items():
        if value is None or not np.isfinite(value):
            problems.append(f"band starting {start:g}: the relativity is not a number")
        elif value <= 0:
            problems.append(f"band starting {start:g}: the relativity must be above 0")
    return problems


def evaluate(
    overlay: Overlay,
    data: pd.DataFrame,
    actual,
    base_expected,
) -> tuple[pd.DataFrame, dict]:
    """Test-rows comparison of the base model and the adjusted model: per band and in total."""
    a = np.asarray(actual, dtype=float)
    base = np.asarray(base_expected, dtype=float)
    adjusted = apply_overlay(overlay, data, base)
    keys = overlay.band_keys(data[overlay.feature])
    base_t = band_table(a, base, keys)
    adj_t = band_table(a, adjusted, keys)
    table = pd.DataFrame(
        {
            "n": base_t["n"],
            "A": base_t["A"],
            "E base": base_t["E"],
            "E adjusted": adj_t["E"],
            "A/E base": base_t["AE"],
            "A/E adjusted": adj_t["AE"],
        }
    )
    keep = np.isfinite(a) & np.isfinite(base) & np.isfinite(adjusted)
    summary = {
        "A/E total base": float(a[keep].sum() / base[keep].sum()),
        "A/E total adjusted": float(a[keep].sum() / adjusted[keep].sum()),
        "RMSE base": float(np.sqrt(np.mean((a[keep] - base[keep]) ** 2))),
        "RMSE adjusted": float(np.sqrt(np.mean((a[keep] - adjusted[keep]) ** 2))),
        "rows": int(keep.sum()),
    }
    return table, summary


def _sheet_name(
    feature: str,
    taken: set,
) -> str:
    """An Excel tab name for a feature: at most 31 characters, no []:*?/\\, unique in the workbook."""
    base = re.sub(r"[\[\]:*?/\\]", " ", str(feature)).strip()[:31] or "feature"
    name, k = base, 2
    while name.lower() in taken:
        suffix = f" ({k})"
        name, k = base[: 31 - len(suffix)] + suffix, k + 1
    taken.add(name.lower())
    return name


def adjustments_workbook(items) -> bytes:
    """An Excel workbook with one tab per feature; each tab lists its bands' Feature, Band and Relativity.
    items are (Overlay, response name) pairs; the response is not written."""
    import io

    frames = {}
    for overlay, _response in items:
        for start in sorted(overlay.relativities):
            value = overlay.relativities[start]
            frames.setdefault(overlay.feature, []).append(
                {
                    "Feature": overlay.feature,
                    "Band": band_label(start, overlay.banding, overlay.levels or None),
                    "Relativity": np.nan if value is None else float(value),
                }
            )
    buffer = io.BytesIO()
    taken = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for feature, rows in frames.items():
            pd.DataFrame(rows, columns=["Feature", "Band", "Relativity"]).to_excel(
                writer, sheet_name=_sheet_name(feature, taken), index=False
            )
        if not frames:
            pd.DataFrame(columns=["Feature", "Band", "Relativity"]).to_excel(writer, sheet_name="Overlays", index=False)
    return buffer.getvalue()
