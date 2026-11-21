"""One-way actual-vs-expected summaries."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .banding import (
    apply_treatment,
    bin_count,
    band_numeric,
    band_date,
    banding_guesser,
    cap_numeric,
    resolve_treatment,
)

# A one-way chart shows at most this many groups. A numeric feature with more distinct values is banded wider until
# it fits; a categorical one keeps its most common levels and puts the rest in "Other". Without this a feature with
# hundreds of thousands of values makes a figure too large for the browser.
MAX_GROUPS = 200


def _limit_groups(
    key: pd.Series,
    g: pd.Series,
    mask: np.ndarray,
    numeric: bool,
    banding: str,
    minimum,
    maximum,
    group_low_exposure: str,
) -> tuple[pd.Series, str | None]:
    """Keep the groups within ``MAX_GROUPS``: wider bands for numbers, an "Other" group for categories."""
    if key[mask].nunique(dropna=False) <= MAX_GROUPS:
        return key, None
    if numeric:
        try:
            width = float(banding)
        except (TypeError, ValueError):
            width = 0.0
        if width <= 0:
            width = float(banding_guesser(g[mask]))
        capped = cap_numeric(g, minimum, maximum)
        while True:
            banded = band_numeric(capped, width, mask, group_low_exposure)
            if banded[mask].nunique(dropna=False) <= MAX_GROUPS or width > 1e15:
                break
            width *= 2
        return banded, f"Grouped into bands of {width:g}: the feature has more than {MAX_GROUPS} groups."
    text = key.astype(str)
    counts = text[mask].value_counts()
    keep = set(counts.index[: MAX_GROUPS - 1])
    return (
        text.where(text.isin(keep), "Other"),
        f"Showing the {MAX_GROUPS - 1} most common levels; the rest are grouped as Other.",
    )


def one_way_summary(
    d: pd.DataFrame,
    response: str,
    weight: str,
    group_by: str,
    add_cols: list[str] | None = None,
    banding="0",
    group_low_exposure: str = "0",
    total_filter=None,
    per_unit_cols=(),
    minimum=None,
    maximum=None,
    bins=None,
) -> pd.DataFrame:
    """Summarise ``response`` (and ``add_cols``, e.g. model predictions) by ``group_by``.

    Returns one row per band with: ``count`` (rows), ``weight`` (sum of weight, or count
    when weight == 'N'), and for each of response/add_cols the *rate* (sum / weight) and
    the sum itself (``<col>_sum``). ``sigma`` is the standard error of the response rate.

    Columns are totals (ClaimCount, total claim amount, predicted totals) unless named in ``per_unit_cols``, which
    are per unit of weight (average severity, frequency, per-unit predictions): those are multiplied by the weight
    first, so every rate is sum of totals / sum of weight and the two kinds can share a chart.

    For a numeric ``group_by``, ``minimum`` and ``maximum`` cap its values before banding (rows outside the range
    join the first / last band). ``bins`` replaces the banding: the range is split into that many equal bins,
    labelled [0,5), [5,10), [10,15), [15,Inf).
    """
    per_unit = set(per_unit_cols or ())
    add_cols = [c for c in (add_cols or []) if c in d.columns]
    mask = np.ones(len(d), bool) if total_filter is None else np.asarray(total_filter) == 1
    if weight != "N":
        w = d[weight].to_numpy(dtype=float)
        mask &= ~np.isnan(w) & (w > 0)
    g = d[group_by]
    banding_s = str(banding)
    numeric = pd.api.types.is_numeric_dtype(g) and not pd.api.types.is_bool_dtype(g)
    if numeric and bin_count(bins):
        key = apply_treatment(g, resolve_treatment(g[mask], minimum, maximum, bins))
    elif numeric and banding_s != "0":
        key = band_numeric(cap_numeric(g, minimum, maximum), float(banding_s), mask, group_low_exposure)
    elif numeric:
        key = cap_numeric(g, minimum, maximum)
    elif pd.api.types.is_datetime64_any_dtype(g) and banding_s != "0":
        key = band_date(g, banding_s)
    else:
        key = g.astype(object).where(g.notna(), "NA").astype(str) if not pd.api.types.is_numeric_dtype(g) else g
        if group_low_exposure != "0" and weight != "no weights":
            wt = pd.Series(1.0 if weight == "N" else d[weight].to_numpy(dtype=float), index=d.index)[mask]
            tot = wt.groupby(key[mask]).sum()
            thresh = {"1%": 0.01 * tot.sum(), "0.1%": 0.001 * tot.sum()}.get(group_low_exposure)
            if thresh is None:
                thresh = float(group_low_exposure)
            small = set(tot.index[tot < thresh])
            key = key.astype(str).where(~key.astype(str).isin({str(s) for s in small}), "Small weights")
    key, grouping_note = _limit_groups(
        key, g, mask, numeric and not bin_count(bins), banding_s, minimum, maximum, group_low_exposure
    )
    df = pd.DataFrame({"key": key[mask].reset_index(drop=True)})  # keeps the order of bins
    df["weight"] = 1.0 if weight == "N" else d.loc[mask, weight].to_numpy(dtype=float)
    cols = [response] + add_cols
    for c in cols:
        values = d.loc[mask, c].to_numpy(dtype=float)
        df[c + "_sum"] = values * df["weight"].to_numpy() if c in per_unit else values
    # for rate sigma: per-row rate variance
    if weight == "N":
        df["_r2"] = df[response + "_sum"] ** 2
    else:
        df["_r2"] = df[response + "_sum"] ** 2 / df["weight"]
    agg = (
        df.groupby("key", observed=True, dropna=False)
        .agg(
            count=("weight", "size"),
            weight=("weight", "sum"),
            **{c + "_sum": (c + "_sum", "sum") for c in cols},
            _r2=("_r2", "sum"),
        )
        .reset_index()
    )
    for c in cols:
        agg[c] = agg[c + "_sum"] / agg["weight"]
    # sigma: sqrt(var of rate) approximated by weighted variance / sqrt(n)
    mean_r = agg[response]
    var = (agg["_r2"] / agg["weight"] - mean_r**2).clip(lower=0)
    agg["sigma"] = np.sqrt(var / agg["count"].clip(lower=1))
    agg = agg.drop(columns="_r2").sort_values("key").reset_index(drop=True)
    if isinstance(agg["key"].dtype, pd.CategoricalDtype):  # bins in their own order, as text for charts and tables
        agg["key"] = agg["key"].astype(object).where(agg["key"].notna(), "NA").astype(str)
    agg = agg.rename(columns={"key": group_by})
    ordered = [group_by, "count", "weight"] + cols + [c + "_sum" for c in cols] + ["sigma"]
    agg = agg[ordered]
    agg.attrs["grouping_note"] = grouping_note  # set when the groups were limited to MAX_GROUPS
    return agg


def add_total_row(
    summary: pd.DataFrame,
    response: str,
    add_cols: list[str],
) -> pd.DataFrame:
    """Append a Total row to a summary table, summing counts, weights and the response and added columns."""
    first = summary.columns[0]
    cols = [response] + [c for c in add_cols if c in summary.columns]
    tot = {first: "Total", "count": summary["count"].sum(), "weight": summary["weight"].sum()}
    for c in cols:
        tot[c + "_sum"] = summary[c + "_sum"].sum()
        tot[c] = tot[c + "_sum"] / tot["weight"] if tot["weight"] else np.nan
    tot["sigma"] = np.nan
    out = summary.copy()
    out[first] = out[first].astype(str)
    return pd.concat([out, pd.DataFrame([tot])], ignore_index=True)[summary.columns]


def column_summary(d: pd.DataFrame) -> pd.DataFrame:
    """Summary of each column."""
    rows = []
    for c in d.columns:
        s = d[c]
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rows.append(
            dict(
                column=c,
                type=str(s.dtype),
                unique=int(s.nunique(dropna=True)),
                NAs=int(s.isna().sum()),
                mean=float(s.mean()) if num else np.nan,
                min=float(s.min()) if num else np.nan,
                max=float(s.max()) if num else np.nan,
            )
        )
    return pd.DataFrame(rows)
