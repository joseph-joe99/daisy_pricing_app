"""How a model's weight column is used (GLM and GBM builds) and what that means for the response and predictions."""
from __future__ import annotations
import numpy as np
import pandas as pd

# select choices: key -> label (the hover text in WEIGHT_BASIS_HELP says when to use each)
WEIGHT_BASES = {
    "rate_weighted": "Total response, per unit of weight",
    "average": "Average response",
}

# hover text for each option and the note shown under the select
WEIGHT_BASIS_HELP = {
    "rate_weighted": (
        "For frequency and severity models. The response is a total (e.g. ClaimCount or total claim amount) and "
        "the model is fitted on response / weight with the weight as a prior weight, so rows with more exposure "
        "count for more. Predictions are returned as totals (expected claims, or expected claim amount)."
    ),
    "average": (
        "For a response that is already per unit of weight (e.g. average severity, or a frequency column), "
        "with the weight as a prior weight. Predictions are per unit of weight."
    ),
}

# short description for the model parameter tables
WEIGHT_BASIS_SHORT = {
    "rate_weighted": "prior weight, fitted per unit (response is a total)",
    "average": "prior weight (response is an average)",
}

class NonPositiveResponseError(ValueError):
    """The gamma objective was given rows whose response is zero, negative or missing (``count`` rows)."""

    def __init__(
        self,
        count: int,
        total: int,
    ):
        super().__init__(
            f"the gamma objective needs a response above 0 on every training and test row, but {count} of {total} "
            "row(s) have a zero, negative or missing response or weight. Use poisson or tweedie for responses with "
            "zeros, or model only the rows with a response and a weight above 0"
        )
        self.count = count  # rows left out: a zero, negative or missing response or weight (one count)
        self.total = total


def fit_basis(model) -> str:
    """The setting a model was built with. Models built with the removed ``total`` (log offset) basis are refused."""
    basis = getattr(model, "weight_basis", None)
    if basis not in WEIGHT_BASES:
        raise ValueError(
            f"the model was built with the '{basis}' weight basis, which is no longer supported; rebuild it with "
            "'Total response, per unit of weight' or 'Average response'"
        )
    return basis


def scale_basis(basis: str) -> str:
    """``total`` if the response and the stored predictions are totals, ``average`` if they are per unit of weight."""
    return "average" if basis == "average" else "total"


def to_total_scale(
    model,
    d: pd.DataFrame,
    predictions,
) -> np.ndarray:
    """Per-unit predictions -> totals for a model fitted per unit of weight; other models are unchanged."""
    predictions = np.asarray(predictions, dtype=float)
    if fit_basis(model) != "rate_weighted" or model.weight == "N":
        return predictions
    if model.weight not in d.columns:
        raise ValueError(f"the data has no '{model.weight}' column, which the model uses as its weight")
    return predictions * pd.to_numeric(d[model.weight], errors="coerce").to_numpy(dtype=float)


def require_positive_weights(
    d: pd.DataFrame,
    weight: str,
) -> None:
    """Refuse to build a model when the weight column has zeros or negative values: remove those rows first."""
    if weight == "N" or weight not in d.columns:
        return
    bad = int((pd.to_numeric(d[weight], errors="coerce") <= 0).sum())
    if bad:
        raise ValueError(
            f"the weight '{weight}' has {bad:,} row(s) with a value of 0 or less. Remove those rows from the data "
            "before building the model"
        )


def zero_weight_predictions(
    predictions,
    weights,
) -> np.ndarray:
    """Totals of a rate_weighted model with the weight applied, for rows without a usable weight.

    A weight of 0 (no exposure) gives a total of 0; a missing or negative weight gives a missing prediction, so
    scoring data cannot produce negative totals.
    """
    predictions = np.array(predictions, dtype=float)
    weights = pd.to_numeric(pd.Series(np.asarray(weights)), errors="coerce").to_numpy(dtype=float)
    predictions[weights == 0] = 0.0
    predictions[~(weights >= 0)] = np.nan
    return predictions
