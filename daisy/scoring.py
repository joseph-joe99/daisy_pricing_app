"""Score out-of-time data on a fitted model: read the data, score it and measure the predictions.

Pure logic (no Shiny), used by the Scoring tab:

* :func:`read_csv_with_progress` reads a cleaned CSV in chunks and reports progress;
* GBMs are scored from a ``model.pkl`` (:func:`read_gbm_pickle`, :func:`gbm_scoring_offset`);
* GLMs are scored from a ratebook (:func:`read_ratebook`, :func:`score_ratebook`): the base prediction
  multiplied by the relativity of each feature's level, as written by *Ratebook (.xlsx)*;
* models already built in the app are scored directly.
"""
from __future__ import annotations
import io
import pickle
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from .gbm import GBMModel, normalise_objective, prediction_performance_metrics, score_gbm

DATASET_LABEL = "Unseen Data"
__all__ = [
    "DATASET_LABEL",
    "Ratebook",
    "iter_csv_chunks",
    "read_csv_with_progress",
    "read_gbm_pickle",
    "read_ratebook",
    "score_ratebook",
    "score_gbm",
    "model_objective",
    "score_metrics",
    "scored_frame",
]


# ---------------------------------------------------------------- data
def iter_csv_chunks(
    path,
    chunksize: int = 50_000,
):
    """Yield ``(chunk, fraction_read, rows_so_far)`` while reading a CSV in chunks (so callers can show progress)."""
    file = Path(str(path).strip().strip("\"'")).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {file}")
    total = max(file.stat().st_size, 1)
    compression = "gzip" if file.suffix.lower() == ".gz" else None
    rows = 0
    with open(file, "rb") as handle:
        for chunk in pd.read_csv(handle, chunksize=chunksize, compression=compression):
            rows += len(chunk)
            yield chunk, min(handle.tell() / total, 0.99), rows


def read_csv_with_progress(
    path,
    on_progress: Callable[[float, int], None] | None = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Read a CSV in chunks, calling ``on_progress(fraction_read, rows_so_far)`` after each chunk."""
    chunks = []
    for chunk, fraction, rows in iter_csv_chunks(path, chunksize):
        chunks.append(chunk)
        if on_progress:
            on_progress(fraction, rows)
    data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if data.empty:
        raise ValueError(f"{Path(str(path)).name} has no rows")
    if on_progress:
        on_progress(1.0, len(data))
    return data


# ---------------------------------------------------------------- GBM (model.pkl)
def read_gbm_pickle(path) -> GBMModel:
    """Load a GBM from ``model.pkl`` (or from the *Model bundle (.zip)* that contains it).

    Only open model files you trust: unpickling can run code.
    """
    file = Path(path)
    if file.suffix.lower() == ".zip":
        with zipfile.ZipFile(file) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith("model.pkl")]
            if not names:
                raise ValueError("the zip file has no model.pkl")
            payload = archive.read(names[0])
        model = pickle.load(io.BytesIO(payload))
    else:
        with open(file, "rb") as handle:
            model = pickle.load(handle)
    if not isinstance(model, GBMModel):
        raise ValueError("the file does not contain a Daisy GBM (expected the model.pkl written by Daisy)")
    return model


# ---------------------------------------------------------------- GLM (ratebook)
@dataclass
class Ratebook:
    """Base prediction and per-feature relativities read from a ratebook workbook."""

    base_prediction: float
    features: dict[str, pd.DataFrame]  # feature -> DataFrame(level, relativity)
    base_levels: dict[str, str] = field(default_factory=dict)
    model_name: str = ""
    weight_basis: str | None = None  # how the model used its weight; None for ratebooks written before it was recorded
    weight: str = "N"


def read_ratebook(path) -> Ratebook:
    """Read a ratebook written by Daisy (*Ratebook (.xlsx)*): a Summary sheet and one sheet per feature."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        if "Summary" not in workbook.sheetnames:
            raise ValueError("this does not look like a Daisy ratebook: there is no 'Summary' sheet")
        rows = [list(row) for row in workbook["Summary"].iter_rows(values_only=True)]
        title = str(rows[0][0] or "") if rows else ""
        base_prediction, base_levels, in_table = None, {}, False
        weight_basis, weight = None, "N"
        for row in rows[1:]:
            first = row[0] if row else None
            if first == "Base prediction":
                base_prediction = row[1]
                if len(row) > 5 and row[2] == "Weight basis" and row[4] == "Weight" and row[3] and row[5]:
                    weight_basis, weight = str(row[3]), str(row[5])
            elif first == "feature":
                in_table = True
            elif in_table and first not in (None, ""):
                base_levels[str(first)] = "" if row[1] is None else str(row[1])
        if base_prediction is None or not np.isfinite(float(base_prediction)) or float(base_prediction) <= 0:
            raise ValueError("the ratebook has no positive 'Base prediction' on its Summary sheet")
        features = {}
        for feature in base_levels:
            sheet = feature[:31]  # Excel limits sheet names to 31 characters
            if sheet not in workbook.sheetnames:
                raise ValueError(f"the ratebook has no sheet for feature '{feature}'")
            table = pd.DataFrame(
                list(workbook[sheet].iter_rows(values_only=True))[1:], columns=["level", "prediction", "relativity"]
            )
            table = table.dropna(subset=["level", "relativity"])
            if table.empty:
                raise ValueError(f"the ratebook sheet for '{feature}' has no relativities")
            features[feature] = table[["level", "relativity"]].reset_index(drop=True)
    finally:
        workbook.close()
    return Ratebook(
        float(base_prediction), features, base_levels, title.removesuffix(" ratebook"), weight_basis, weight
    )


def score_ratebook(
    ratebook: Ratebook,
    d: pd.DataFrame,
    weight_column: str = "N",
) -> tuple[np.ndarray, list[str]]:
    """Score rows as base prediction x the relativity of each feature's level.

    A ratebook is per unit of weight. When its model used the weight as an offset or fitted per unit of weight
    (the response was a total), the rates are multiplied by ``weight_column`` to give expected totals, like the
    model's own predictions; ``Average response`` ratebooks, and ratebooks written before the setting was recorded,
    stay as they are.

    Numeric features use the band that starts at the nearest ratebook level at or below the value (values
    outside the range take the first / last band). Levels that are not in the ratebook get a relativity of 1
    and are listed in the returned notes.
    """
    missing = [feature for feature in ratebook.features if feature not in d.columns]
    if missing:
        raise ValueError("the scoring data is missing ratebook features: " + ", ".join(missing))
    notes, log_total = [], np.zeros(len(d))
    for feature, table in ratebook.features.items():
        relativity = table["relativity"].to_numpy(dtype=float)
        levels_numeric = pd.to_numeric(table["level"], errors="coerce")
        if levels_numeric.notna().all() and pd.api.types.is_numeric_dtype(d[feature]):
            order = np.argsort(levels_numeric.to_numpy())
            levels, relativity = levels_numeric.to_numpy()[order], relativity[order]
            values = pd.to_numeric(d[feature], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            position = np.clip(
                np.searchsorted(levels, np.where(finite, values, levels[0]), side="right") - 1, 0, len(levels) - 1
            )
            applied = np.where(finite, relativity[position], 1.0)
            outside = int(np.sum(finite & ((values < levels[0]) | (values > levels[-1]))))
            if outside:
                notes.append(
                    f"{feature}: {outside:,} rows are outside the ratebook range "
                    f"({levels[0]:g} to {levels[-1]:g}) and use the nearest band."
                )
            if (~finite).sum():
                notes.append(f"{feature}: {int((~finite).sum()):,} missing values use a relativity of 1.")
        else:
            lookup = dict(zip(table["level"].astype(str), relativity))
            mapped = d[feature].astype(str).map(lookup)
            unseen = mapped.isna()
            if unseen.any():
                examples = ", ".join(sorted(d.loc[unseen, feature].astype(str).unique())[:5])
                notes.append(
                    f"{feature}: {int(unseen.sum()):,} rows have levels not in the ratebook "
                    f"({examples}) and use a relativity of 1."
                )
            applied = mapped.fillna(1.0).to_numpy(dtype=float)
        with np.errstate(divide="ignore"):
            log_total += np.log(applied)
    predictions = ratebook.base_prediction * np.exp(log_total)
    if ratebook.weight_basis in ("total", "rate_weighted"):
        if weight_column == "N":
            notes.append(
                "This ratebook is per unit of weight but no weight column is selected, so the predictions "
                "are per unit and are not scaled to totals."
            )
        elif weight_column not in d.columns:
            raise ValueError(f"the scoring data has no '{weight_column}' weight column to scale the ratebook to totals")
        else:
            predictions = predictions * pd.to_numeric(d[weight_column], errors="coerce").to_numpy(dtype=float)
            notes.append(f"Predictions are the ratebook rates multiplied by {weight_column} (expected totals).")
    return predictions, notes


# ---------------------------------------------------------------- objective, metrics, output
def model_objective(model) -> tuple[str | None, float | None]:
    """``(objective, tweedie power)`` of a GLM or GBM, or ``(None, None)`` when unknown."""
    if isinstance(model, GBMModel):
        params = model.params or {}
        return params.get("objective"), params.get("tweedie_variance_power")
    return getattr(model, "objective", None), getattr(model, "var_power", None)


def score_metrics(
    d: pd.DataFrame,
    response: str,
    predictions: np.ndarray,
    weight: str = "N",
    objective: str | None = None,
    tweedie_power: float = 1.5,
    weight_basis: str = "total",
) -> pd.DataFrame:
    """The standard Daisy metrics table (GINI, PoDE, RMSE, Within ...) for one out-of-time dataset."""
    if response not in d.columns:
        raise ValueError(f"the scoring data has no '{response}' column to compare the predictions with")
    if weight != "N" and weight not in d.columns:
        raise ValueError(f"the scoring data has no '{weight}' weight column")
    return prediction_performance_metrics(
        d.reset_index(drop=True),
        response,
        predictions,
        weight,
        objective=normalise_objective(objective),
        tweedie_power=float(tweedie_power or 1.5),
        dataset_label=DATASET_LABEL,
        weight_basis=weight_basis,
    )


def scored_frame(
    d: pd.DataFrame,
    predictions: np.ndarray,
    prediction_name: str = "prediction",
) -> pd.DataFrame:
    """The scoring data with the predictions added as the last column (the file users download)."""
    out = d.reset_index(drop=True).copy()
    out[prediction_name] = np.asarray(predictions, dtype=float)
    return out
