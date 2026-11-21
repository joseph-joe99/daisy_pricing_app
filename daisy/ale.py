"""Accumulated local effects using PyALE and Daisy's fitted model predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def observation_counts(
    series: pd.Series,
    grid,
    numeric: bool,
) -> np.ndarray:
    """Observations behind each row of a one-feature ALE table (they always add up to the rows used).

    Categorical features: the number of rows at each level. Numeric features: the number of rows in the
    band that starts at each grid point and runs up to the next one (the last point is closed, so it holds
    the rows at the maximum). PyALE's own ``size`` column counts the band *ending* at each grid point (and
    the discrete levels used for each step), so its first row is always 0.
    """
    index = pd.Index(grid)
    if numeric:
        points = np.asarray(index, dtype=float)
        values = series.to_numpy(dtype=float)
        position = np.clip(np.searchsorted(points, values, side="right") - 1, 0, len(points) - 1)
        return np.bincount(position, minlength=len(points)).astype(int)
    counts = series.astype(object).value_counts()
    return np.array([int(counts.get(level, 0)) for level in index], dtype=int)


def calculate_ale(
    data: pd.DataFrame,
    model,
    features: list[str],
    grid_size: int = 20,
):
    """Accumulated local effects (ALE) for one or two features, computed with PyALE and centred so the average effect is zero."""
    from PyALE import ale

    if not 1 <= len(features) <= 2 or len(set(features)) != len(features):
        raise ValueError("Select one feature or two different features for ALE.")
    if any(feature not in data.columns for feature in features):
        raise ValueError("ALE features must be columns in the data.")
    sample = data.dropna(subset=features).copy()
    if sample.empty or any(sample[feature].nunique() < 2 for feature in features):
        raise ValueError("ALE needs at least two distinct values for each selected feature.")
    numeric = [
        pd.api.types.is_numeric_dtype(sample[feature]) and not pd.api.types.is_bool_dtype(sample[feature])
        for feature in features
    ]
    if len(features) == 2 and not all(numeric):
        raise ValueError("Two-feature ALE requires two numeric features.")
    options = dict(X=sample, model=model, feature=features, grid_size=int(grid_size), include_CI=False, plot=False)
    if len(features) == 1 and not numeric[0]:
        feature = features[0]
        series = sample[feature]
        levels = (
            list(series.cat.categories)
            if isinstance(series.dtype, pd.CategoricalDtype)
            else sorted(series.unique(), key=str)
        )
        observed = set(series.dropna())
        sample[feature] = pd.Categorical(
            series, categories=[level for level in levels if level in observed], ordered=True
        )
        options.update(
            feature_type="categorical", encode_fun=lambda frame: frame.copy(), predictors=list(sample.columns)
        )
    elif len(features) == 1:
        options["feature_type"] = "continuous"
    result = ale(**options)
    if not isinstance(result, pd.DataFrame) or not np.isfinite(result.to_numpy(dtype=float)).any():
        raise ValueError("ALE could not be calculated for the selected data.")
    if len(features) == 1:
        try:  # replace PyALE's size (0 in the first row) with the observations behind every row
            result["size"] = observation_counts(sample[features[0]], result.index, numeric[0])
        except Exception:  # pragma: no cover - keep PyALE's own column if the levels cannot be matched
            pass
    return result
