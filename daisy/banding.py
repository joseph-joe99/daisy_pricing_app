"""Banding helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd


def banding_guesser(x) -> float:
    """Guess a band width for a numeric series from its values (at most the first 10,000 non-missing ones). Returns 1.0 when no values are left."""
    x = pd.Series(x).dropna()
    x = x.iloc[:10000]
    if x.empty:
        return 1.0
    s = x.std(ddof=1) / 20 if len(x) > 1 else 1.0
    if not np.isfinite(s) or s == 0:
        s = 1.0
    exponent = np.floor(np.log10(s))
    mantissa = s / 10**exponent
    m = 1 if mantissa < 2 else (2 if mantissa < 5 else 5)
    banding = m * 10 ** (exponent + 1)
    n_unique = x.nunique()
    if n_unique <= 5 and x.min() == 0 and x.max() == 1:
        banding = 1
    if n_unique <= 100 and x.min() <= 20 and 1 <= x.max() <= 100:
        banding = 1
    return float(banding)


def banding_guesser_for(series: pd.Series):
    """Banding guess for any column: 'Mnth' for dates, a numeric guess for numbers, and 0 (no banding) for everything else."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Mnth"
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return banding_guesser(series)
    return 0


def band_numeric(
    g: pd.Series,
    banding: float,
    rows: np.ndarray,
    group_low_exposure: str = "0",
) -> pd.Series:
    """floor(g / banding) * banding, optionally clamping the 1% / 0.1% tails."""
    g = g.astype(float)
    banded = np.floor(g / banding) * banding
    q = {"1%": 0.01, "0.1%": 0.001}.get(group_low_exposure)
    if q is not None and rows.any():
        lo = np.nanquantile(g[rows], q)
        hi = np.nanquantile(g[rows], 1 - q)
        lo_b = np.floor(lo / banding) * banding
        hi_b = (1 + np.floor(hi / banding)) * banding
        banded = banded.clip(lower=lo_b, upper=hi_b)
    return banded


def band_date(
    g: pd.Series,
    banding: str,
) -> pd.Series:
    """Band a date series to the chosen period (Day, Week, Month or Year)."""
    g = pd.to_datetime(g)
    if banding == "Day":
        return g.dt.normalize()
    if banding == "Week":
        iso = g.dt.isocalendar()
        return 100 * iso["year"].astype(int) + iso["week"].astype(int)
    if banding == "Mnth":
        return 100 * g.dt.year + g.dt.month
    if banding == "Qtr":
        return 100 * g.dt.year + ((g.dt.month - 1) // 3 + 1)
    if banding == "Year":
        return g.dt.year
    return g


# ---------------------------------------------------------------- minimum / maximum and bins
MAX_BINS = 500


def _number(value) -> float | None:
    """A finite float, or None for blanks, text and missing values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _text(
    value: float,
    digits: int = 6,
) -> str:
    """Format a number for a band label: whole numbers without decimals, others to the given significant figures."""
    return str(int(value)) if float(value).is_integer() and abs(value) < 1e15 else f"{value:.{digits}g}"


def bin_count(value) -> int | None:
    """The number of bins asked for: None for blank or 0, else a whole number from 1 to ``MAX_BINS``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    number = _number(value)
    if number is None or number < 0 or not float(number).is_integer():
        raise ValueError("bins must be a whole number, e.g. 4")
    if number > MAX_BINS:
        raise ValueError(f"bins cannot be more than {MAX_BINS}")
    return int(number) or None


def cap_numeric(
    g: pd.Series,
    minimum=None,
    maximum=None,
) -> pd.Series:
    """Values below ``minimum`` become the minimum and values above ``maximum`` become the maximum."""
    lower, upper = _number(minimum), _number(maximum)
    if lower is None and upper is None:
        return g
    return g.clip(lower=lower, upper=upper)


def bin_labels(edges) -> list[str]:
    """[0,5), [5,10), [10,15), [15,Inf) for the bin starts 0, 5, 10, 15."""
    for digits in (6, 12, 17):
        starts = [_text(edge, digits) for edge in edges]
        if len(set(starts)) == len(starts):
            break
    return [f"[{start},{end})" for start, end in zip(starts, starts[1:] + ["Inf"])]


def resolve_treatment(
    series: pd.Series,
    minimum=None,
    maximum=None,
    bins=None,
) -> dict | None:
    """Fix how a numeric feature is capped / binned, so the same rule can be applied to any other data.

    Returns None when there is nothing to do. ``min`` / ``max`` are the caps asked for (None = no cap). With
    ``bins`` the range (the caps, else the range of ``series``) is split into that many bins of equal width and
    the dictionary also holds where each bin starts (``edges``) and the labels: 4 bins from 0 to 20 are
    [0,5), [5,10), [10,15), [15,Inf). The last bin has no upper end, and values below the first bin join it.
    """
    lower, upper, count = _number(minimum), _number(maximum), bin_count(bins)
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("the minimum must be below the maximum")
    if not count:
        return None if lower is None and upper is None else {"min": lower, "max": upper, "bins": 0}
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) and (lower is None or upper is None):
        raise ValueError("the feature has no values to put into bins")
    low = lower if lower is not None else float(values.min())
    high = upper if upper is not None else float(values.max())
    if high <= low:  # a single value, or one cap given and all the data on the wrong side of it
        edges = [float(low if lower is not None or upper is None else high)]
    else:
        edges = [float(edge) for edge in np.linspace(low, high, count + 1)[:-1]]
    return {"min": lower, "max": upper, "bins": len(edges), "edges": edges, "labels": bin_labels(edges)}


def apply_treatment(
    series: pd.Series,
    treatment: dict | None,
) -> pd.Series:
    """Cap a numeric feature, or turn it into its ordered bins. Anything not numeric is left alone, so applying
    a treatment twice does nothing the second time."""
    if not treatment or not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series
    if not treatment.get("bins"):
        return cap_numeric(series, treatment.get("min"), treatment.get("max"))
    labels, edges = treatment["labels"], np.asarray(treatment["edges"], dtype=float)
    values = series.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    codes = np.searchsorted(edges, np.where(missing, edges[0], values), side="right") - 1
    codes = np.where(missing, -1, np.clip(codes, 0, len(labels) - 1)).astype(int)
    return pd.Series(
        pd.Categorical.from_codes(codes, categories=labels, ordered=True), index=series.index, name=series.name
    )


def treatments_from_spec(
    d: pd.DataFrame,
    feature_spec: pd.DataFrame | None,
    features,
) -> dict:
    """The treatment of each numeric feature whose ``banding_type`` in the feature specification is ``capped`` or
    ``bins``, resolved against ``d`` (feature -> treatment). These are what a model applies to its data."""
    from .specs import feature_treatment_spec

    treatments = {}
    for feature in features:
        if feature not in d.columns:
            continue
        chosen = feature_treatment_spec(feature_spec, feature)
        series = d[feature]
        if chosen is None or not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        treatment = resolve_treatment(series, chosen["min"], chosen["max"], chosen["bins"])
        if treatment is not None:
            treatments[feature] = treatment
    return treatments


def treatment_text(treatment: dict) -> str:
    """One line describing a treatment, for tables and notes."""
    parts = []
    if treatment.get("min") is not None:
        parts.append(f"minimum {_text(treatment['min'])}")
    if treatment.get("max") is not None:
        parts.append(f"maximum {_text(treatment['max'])}")
    if treatment.get("bins"):
        labels = treatment["labels"]
        shown = ", ".join(labels) if len(labels) <= 4 else f"{labels[0]}, {labels[1]}, ... {labels[-1]}"
        parts.append(f"{len(labels)} bin{'s' if len(labels) != 1 else ''}: {shown}")
    return ", ".join(parts)
