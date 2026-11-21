"""LightGBM engine: training, SHAP, importances, constraints."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .banding import apply_treatment
from .weights import (
    NonPositiveResponseError,
    WEIGHT_BASES,
    WEIGHT_BASIS_SHORT,
    fit_basis,
    require_positive_weights,
    scale_basis,
    to_total_scale,
    zero_weight_predictions,
)

# objective -> (link, default metric, monotonicity supported)
LGBM_OBJECTIVES = {
    "regression": dict(link="identity", metric="l2", monotone=True),
    "regression_l1": dict(link="identity", metric="l1", monotone=True),
    "huber": dict(link="identity", metric="huber", monotone=True),
    "poisson": dict(link="log", metric="poisson", monotone=True),
    "gamma": dict(link="log", metric="gamma", monotone=True),
    "tweedie": dict(link="log", metric="tweedie", monotone=True),
    "binary": dict(link="logit", metric="binary_logloss", monotone=True),
    "cross_entropy": dict(link="logit", metric="cross_entropy", monotone=True),
}


def _lgb():
    """Import LightGBM on first use, with a clear error if it is not installed."""
    try:
        import lightgbm as lgb

        return lgb
    except ImportError as e:  # pragma: no cover
        raise ImportError("lightgbm is required for GBMs: pip install lightgbm") from e


def convert_monotonicity(x) -> list[int]:
    """Map monotonicity labels (increasing, decreasing or none) to LightGBM's 1, -1 and 0 constraints."""
    m = []
    for v in x:
        v = str(v).strip().lower()
        m.append(1 if v in ("increasing", "1") else -1 if v in ("decreasing", "-1") else 0)
    return m


def make_interaction_constraints(
    feature_table: pd.DataFrame,
    groups: list[str] | None,
    features: list[str],
):
    """Port of make_fics: features sharing an interaction_grouping may interact; others stay 1-D."""
    if not groups:
        return None
    sub = feature_table[feature_table["feature"].isin(features)].copy()
    sub["g"] = np.where(sub["interaction_grouping"].isin(groups), sub["interaction_grouping"], "non_grouped")
    idx = {f: i for i, f in enumerate(features)}
    out = []
    for g, rows in sub.groupby("g"):
        feats = list(rows["feature"])
        if g == "non_grouped":
            out.extend([[idx[f]] for f in feats])
        else:
            out.append([idx[f] for f in feats])
    return out


@dataclass
class GBMModel:
    booster: object
    response: str
    weight: str
    init_score: str
    params: dict
    features: list
    categories: dict  # feature -> list of category levels used in training
    link: str
    offset: np.ndarray | None
    predictions: np.ndarray
    pred_rows: np.ndarray  # boolean mask over the data used to predict
    importances: pd.DataFrame
    evaluation_log: dict
    best_iteration: int
    SHAP: pd.DataFrame | None = None
    SHAP_rows: np.ndarray | None = None
    tree_table: pd.DataFrame | None = None
    gain_summary: pd.DataFrame | None = None
    feature_table: pd.DataFrame | None = None
    interaction_groups: list[str] | None = None
    training_config: dict = field(default_factory=dict)
    run_time: float = 0.0
    name: str = "GBM"
    weight_basis: str = "rate_weighted"  # rate_weighted / average: the weight is a sample weight
    # starting level (link scale) of a model trained with an offset: LightGBM does not boost from the average when
    # it is given an init_score, so the level is fitted first and is part of every prediction
    offset_base: float = 0.0
    # feature -> minimum / maximum caps and bins (banding.resolve_treatment) the model was trained
    # with; they are applied to any data it predicts, so the model always takes the features as they are stored
    treatments: dict = field(default_factory=dict)

    def _matrix(
        self,
        d: pd.DataFrame,
    ) -> pd.DataFrame:
        """Model matrix for rows d: applies the treatments and the training categories."""
        X = d[self.features].copy()
        for f, treatment in (getattr(self, "treatments", None) or {}).items():
            X[f] = apply_treatment(X[f], treatment)
        for f, lv in self.categories.items():
            X[f] = pd.Categorical(X[f].astype(str), categories=lv)
        return X

    def predict_raw(
        self,
        d: pd.DataFrame,
    ) -> np.ndarray:
        """Predict on the link scale (before any inverse link) using the best iteration."""
        return self.booster.predict(self._matrix(d), num_iteration=self.best_iteration or None)

    def predict(
        self,
        d: pd.DataFrame,
        offset: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predictions on the response scale. Without ``offset`` this is the *rate* (no exposure/offset)."""
        raw = self.booster.predict(self._matrix(d), num_iteration=self.best_iteration or None, raw_score=True)
        raw = raw + getattr(self, "offset_base", 0.0)
        if offset is not None:
            raw = raw + offset
        if self.link == "log":
            return np.exp(raw)
        if self.link == "logit":
            return 1 / (1 + np.exp(-raw))
        return raw


def _prepare(
    d,
    features,
    cat_features,
    treatments=None,
):
    """Copy the modelling columns, apply any treatments and encode categorical features ready for LightGBM."""
    X = d[features].copy()
    for f, treatment in (treatments or {}).items():
        X[f] = apply_treatment(X[f], treatment)
    cats = {}
    for f in cat_features:
        s = X[f].astype(str)
        lv = sorted(s.unique())
        cats[f] = lv
        X[f] = pd.Categorical(s, categories=lv)
    return X, cats


def build_gbm(
    d: pd.DataFrame,
    response: str,
    weight: str,
    features: list[str],
    params: dict | None = None,
    init_score: str = "no offset",
    feature_table: pd.DataFrame | None = None,
    interaction_groups: list[str] | None = None,
    shap_sample: str = "10k",
    num_threads: int = -1,
    seed: int = 1,
    weight_basis: str = "rate_weighted",
    treatments: dict | None = None,
    positive_response: bool = False,
) -> GBMModel:
    """Train on rows with train_test == 0, validate on train_test == 1.

    ``treatments`` (feature -> ``banding.resolve_treatment``) caps features at a minimum / maximum or turns them
    into bins before training; the model keeps them and applies them to whatever it predicts.

    ``weight_basis``: ``rate_weighted`` - the response is a total (ClaimCount or total claim amount) but the model is
    fitted on response / weight with the weight as a LightGBM sample weight, and the stored predictions are totals;
    ``average`` - the response is already per unit of weight and the weight is a sample weight, so predictions are
    per unit. An ``init_score`` column is an expected total; it is divided by the weight to give the rate-scale offset.

    ``positive_response`` leaves out rows whose response is zero, negative or missing (for the gamma objective).
    """
    import time

    lgb = _lgb()
    t0 = time.time()
    p = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=5,
        max_depth=-1,
        feature_fraction=1.0,
        bagging_fraction=0.75,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=0.0,
        min_data_in_leaf=100,
        num_iterations=1000,
        early_stopping_round=50,
        verbose=-1,
        seed=seed,
        num_threads=num_threads,
    )
    p.update(params or {})
    obj = p["objective"]
    info = LGBM_OBJECTIVES.get(obj, dict(link="identity", metric="l2", monotone=False))
    p.setdefault("metric", info["metric"])
    d = d.reset_index(drop=True)
    if "train_test" not in d.columns:
        raise ValueError("GBMs need a train_test column (0 = train, 1 = test)")
    rows_include = np.ones(len(d), bool) if weight == "N" else d[weight].to_numpy(float) > 0
    if positive_response:  # leave out rows whose response is zero, negative or missing (training, testing, predictions)
        response_values = d[response].to_numpy(float)
        rows_include &= np.isfinite(response_values) & (response_values > 0)
    tt = d["train_test"].to_numpy()
    treatments = {
        f: t for f, t in (treatments or {}).items() if f in features and t and pd.api.types.is_numeric_dtype(d[f])
    }
    cat_features = [f for f in features if not pd.api.types.is_numeric_dtype(d[f]) or treatments.get(f, {}).get("bins")]
    X, cats = _prepare(d, features, cat_features, treatments)
    link = info["link"]
    if weight_basis not in WEIGHT_BASES:
        raise ValueError(f"weight basis must be one of {list(WEIGHT_BASES)}")
    offset = None
    weights = None if weight == "N" else d[weight].to_numpy(float)
    sample_weight = weights if weights is not None and weight_basis in ("average", "rate_weighted") else None
    if init_score != "no offset":
        offset = d[init_score].to_numpy(float)
        if weights is not None and weight_basis == "rate_weighted":
            with np.errstate(divide="ignore", invalid="ignore"):
                offset = offset / weights  # an expected total -> an expected rate, the scale the model is fitted on
    if offset is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            if link == "log":
                offset = np.log(offset)
            elif link == "logit":
                offset = np.log(offset / (1 - offset))
        offset[~np.isfinite(offset)] = 0.0
    y = d[response].to_numpy(float)
    if weights is not None and weight_basis == "rate_weighted":
        y = np.divide(y, weights, out=y.copy(), where=rows_include)  # fit per unit of weight (weights > 0 here)
    tr = rows_include & (tt == 0)
    te = rows_include & (tt == 1)
    if tr.sum() == 0:
        raise ValueError("no training rows (train_test == 0 with non-zero weight)")
    if obj == "gamma":
        # LightGBM's gamma loss and metric stop with a fatal error on any response that is zero, negative or missing
        raw_response = d[response].to_numpy(float)
        response_ok = np.isfinite(raw_response) & (raw_response > 0)
        if not positive_response and not response_ok.all():
            # the rows left out are the union of a bad response and a bad weight: one count
            raise NonPositiveResponseError(int(np.sum(~(rows_include & response_ok))), len(d))
    if not positive_response:  # with the response > 0 option the rows with a zero weight are left out and reported
        require_positive_weights(d, weight)
    # LightGBM skips boost_from_average when an init_score is supplied, so an offset model would start from a
    # rate of exp(0) = 1 and the trees would have to learn the overall level. Fit the level first instead (the
    # intercept-only model with the offset, on the training rows) and train on offset + level.
    offset_base = 0.0
    if offset is not None:
        level_weight = np.ones(len(y)) if sample_weight is None else sample_weight
        usable = tr & np.isfinite(y) & np.isfinite(offset) & np.isfinite(level_weight)
        if link == "log":
            actual = float(np.sum(level_weight[usable] * y[usable]))
            expected = float(np.sum(level_weight[usable] * np.exp(offset[usable])))
            if actual > 0 and expected > 0 and np.isfinite(expected):
                offset_base = float(np.log(actual / expected))
        elif link == "identity" and usable.any() and level_weight[usable].sum() > 0:
            offset_base = float(np.average(y[usable] - offset[usable], weights=level_weight[usable]))
    init = None if offset is None else offset + offset_base
    if info["monotone"] and feature_table is not None and "monotonicity" in feature_table.columns:
        ft = feature_table.set_index("feature")
        mono = convert_monotonicity([ft["monotonicity"].get(f, "") for f in features])
        # a feature banded into categories has no order for LightGBM to constrain
        mono = [0 if treatments.get(f, {}).get("bins") else m for f, m in zip(features, mono)]
        if any(mono):
            p["monotone_constraints"] = mono
    ic = (
        make_interaction_constraints(feature_table, interaction_groups, features) if feature_table is not None else None
    )
    if ic is not None:
        p["interaction_constraints"] = ic
    dataset_params = {"feature_pre_filter": False}
    dtrain = lgb.Dataset(
        X[tr],
        label=y[tr],
        init_score=None if init is None else init[tr],
        weight=None if sample_weight is None else sample_weight[tr],
        categorical_feature=cat_features or "auto",
        free_raw_data=False,
        params=dataset_params,
    )
    valids, names = [dtrain], ["train"]
    if te.sum() > 0:
        valids.append(
            lgb.Dataset(
                X[te],
                label=y[te],
                init_score=None if init is None else init[te],
                weight=None if sample_weight is None else sample_weight[te],
                reference=dtrain,
                params=dataset_params,
            )
        )
        names.append("test")
    evals: dict = {}
    n_rounds = int(p.pop("num_iterations"))
    es = int(p.pop("early_stopping_round", 0) or 0)
    callbacks = [lgb.record_evaluation(evals)]
    if es > 0 and te.sum() > 0:
        callbacks.append(lgb.early_stopping(es, first_metric_only=True, verbose=False))
    booster = lgb.train(p, dtrain, num_boost_round=n_rounds, valid_sets=valids, valid_names=names, callbacks=callbacks)
    best = booster.best_iteration or booster.current_iteration()
    raw = booster.predict(X[rows_include], num_iteration=best)  # lightgbm applies the objective link, but no init_score
    pred = raw.copy()
    if offset is not None:
        # LightGBM predict() excludes init_score: add it back on the link scale
        raw_link = booster.predict(X[rows_include], num_iteration=best, raw_score=True) + init[rows_include]
        pred = np.exp(raw_link) if link == "log" else 1 / (1 + np.exp(-raw_link)) if link == "logit" else raw_link
    if weights is not None and weight_basis == "rate_weighted":
        pred = pred * weights[rows_include]  # stored as totals, like the response
    imp = pd.DataFrame(
        {"Feature": features, "Gain": booster.feature_importance("gain"), "Split": booster.feature_importance("split")}
    )
    for c in ("Gain", "Split"):
        tot = imp[c].sum()
        imp[c] = 100 * imp[c] / tot if tot else 0.0
    imp = imp.sort_values("Gain", ascending=False).reset_index(drop=True)
    metric = p["metric"] if isinstance(p["metric"], str) else p["metric"][0]
    log = {k: list(v[metric]) for k, v in evals.items() if metric in v}
    model = GBMModel(
        booster=booster,
        response=response,
        weight=weight,
        init_score=init_score,
        params=p,
        features=features,
        categories=cats,
        link=link,
        offset=offset,
        predictions=pred,
        pred_rows=rows_include,
        importances=imp,
        evaluation_log=log,
        best_iteration=best,
        feature_table=feature_table,
        interaction_groups=list(interaction_groups or []),
        weight_basis=weight_basis,
        offset_base=offset_base,
        treatments=treatments,
        training_config={
            "num_iterations": n_rounds,
            "early_stopping_round": es,
            "shap_sample": shap_sample,
            "num_threads": num_threads,
            "seed": seed,
        },
    )
    # SHAP values (link scale, one column per feature + base score) on a sample of rows
    if shap_sample != "No" and p.get("linear_tree") not in (True, "true", "TRUE"):
        idx = np.flatnonzero(rows_include)
        if shap_sample == "10k" and len(idx) > 10000:
            idx = np.sort(np.random.default_rng(seed).choice(idx, 10000, replace=False))
        contrib = booster.predict(X.iloc[idx], num_iteration=best, pred_contrib=True)
        contrib[:, -1] += offset_base  # the fitted starting level belongs to the base score
        model.SHAP = pd.DataFrame(contrib, columns=[f"lgbm_SHAP_{f}" for f in features] + ["lgbm_SHAP_base_score"])
        model.SHAP_rows = idx
    try:
        tt_df = booster.trees_to_dataframe()
        model.tree_table = tt_df[tt_df["tree_index"] < best]
        model.gain_summary = gain_summary_from_trees(model.tree_table)
    except Exception:
        pass
    model.run_time = time.time() - t0
    return model


def gbm_performance_metrics(
    d: pd.DataFrame,
    model: GBMModel,
    weight: str = "N",
    rows_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return exposure-weighted fit metrics by train/test split for a fitted GBM (PoDE on the GBM's objective)."""
    return model_performance_metrics(d, model, weight, rows_mask)


def model_objective_power(model) -> tuple[str | None, float]:
    """``(objective, Tweedie variance power)`` of a fitted GLM or GBM, for the deviance behind PoDE."""
    params = getattr(model, "params", None)
    if isinstance(params, dict):  # GBM
        objective, power = params.get("objective"), params.get("tweedie_variance_power")
    else:  # GLM
        objective, power = getattr(model, "objective", None), getattr(model, "var_power", None)
    try:
        power = float(power)
    except (TypeError, ValueError):
        power = 1.5
    return objective, power if np.isfinite(power) else 1.5


def model_performance_metrics(
    d: pd.DataFrame,
    model,
    weight: str | None = None,
    rows_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """The metrics table of a fitted GLM or GBM: PoDE uses the deviance of the model's own objective, and the
    weight defaults to the weight the model was built with (its stored predictions are on that basis)."""
    objective, power = model_objective_power(model)
    return prediction_performance_metrics(
        d,
        model.response,
        model.predictions,
        model.weight if weight is None else weight,
        model.pred_rows,
        rows_mask,
        objective=objective,
        tweedie_power=power,
        weight_basis=model_weight_basis(model),
    )


def model_weight_basis(model) -> str:
    """``total`` if the model's response and stored predictions are totals, ``average`` if per unit of weight.

    (``rate_weighted`` models are fitted per unit of weight but their response and stored predictions are totals.)
    """
    return scale_basis(fit_basis(model))


def _prediction_positions(
    pred_rows: np.ndarray | None,
    row_count: int,
) -> np.ndarray:
    """Row positions to predict: all rows when none are given, otherwise a boolean mask or a list of positions."""
    if pred_rows is None:
        return np.arange(row_count)
    rows = np.asarray(pred_rows)
    return np.flatnonzero(rows) if rows.dtype == bool else rows.astype(int, copy=False)


DEVIANCE_OBJECTIVES = ["poisson", "gamma", "tweedie", "gaussian", "binomial"]
_OBJECTIVE_ALIASES = {
    "poisson": "poisson",
    "quasipoisson": "poisson",
    "gamma": "gamma",
    "tweedie": "tweedie",
    "gaussian": "gaussian",
    "gaussian (log link)": "gaussian",
    "regression": "gaussian",
    "regression_l2": "gaussian",
    "l2": "gaussian",
    "regression_l1": "gaussian",
    "huber": "gaussian",
    "binomial": "binomial",
    "binary": "binomial",
    "cross_entropy": "binomial",
    "xentropy": "binomial",
}


def normalise_objective(
    objective: str | None,
    default: str | None = "poisson",
) -> str | None:
    """Map a GLM family / LightGBM objective name to one of :data:`DEVIANCE_OBJECTIVES`."""
    return _OBJECTIVE_ALIASES.get(str(objective).strip().lower(), default) if objective is not None else default


def unit_deviance(
    objective: str,
    y: np.ndarray,
    mu: np.ndarray,
    power: float = 1.5,
) -> np.ndarray | None:
    """Unit deviance of ``mu`` for ``y`` under the objective, or ``None`` if the data do not suit it."""
    from scipy.special import xlogy

    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    if objective == "gaussian":
        return (y - mu) ** 2
    if objective == "poisson":
        if np.any(y < 0) or np.any(mu < 0):
            return None
        terms = mu.copy()
        positive = y > 0
        terms[positive] = y[positive] * np.log(y[positive] / np.maximum(mu[positive], np.finfo(float).tiny)) - (
            y[positive] - mu[positive]
        )
        return 2.0 * terms
    if objective == "gamma":
        if np.any(y <= 0) or np.any(mu <= 0):
            return None
        return 2.0 * (-np.log(y / mu) + (y - mu) / mu)
    if objective == "tweedie":
        if np.any(y < 0) or np.any(mu <= 0) or not 1.0 < power < 2.0:
            return None
        return 2.0 * (
            np.power(y, 2 - power) / ((1 - power) * (2 - power))
            - y * np.power(mu, 1 - power) / (1 - power)
            + np.power(mu, 2 - power) / (2 - power)
        )
    if objective == "binomial":
        if np.any(y < 0) or np.any(y > 1) or np.any(mu <= 0) or np.any(mu >= 1):
            return None
        return 2.0 * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu)))
    return None


def prediction_performance_metrics(
    d: pd.DataFrame,
    response: str,
    predictions: np.ndarray,
    weight: str = "N",
    pred_rows: np.ndarray | None = None,
    rows_mask: np.ndarray | None = None,
    objective: str | None = None,
    tweedie_power: float = 1.5,
    dataset_label: str | None = None,
    weight_basis: str = "total",
) -> pd.DataFrame:
    """Return exposure-weighted fit metrics for response-scale predictions aligned to data rows.

    PoDE (proportion of deviance explained) uses the deviance of ``objective`` (a GLM family or LightGBM
    objective name; Poisson when omitted). With ``dataset_label`` all rows form one dataset with that name,
    otherwise rows are split by a ``train_test`` column when there is one.

    ``weight_basis`` is ``total`` when the response and predictions are totals (ClaimCount, total claim amount; the
    weight is exposure / claim count) and ``average`` when they are per unit of weight. Gini ranks rows by predicted
    rate and measures the share of the *total* actual captured; PoDE, RMSE and Within % are measured per unit of
    weight, so all weight settings are comparable.
    """
    family = normalise_objective(objective)
    positions = _prediction_positions(pred_rows, len(d))
    aligned = d.iloc[positions]
    actual = aligned[response].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    exposure = np.ones(len(aligned)) if weight == "N" else aligned[weight].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(exposure) & (exposure > 0)
    if rows_mask is not None:
        valid &= np.asarray(rows_mask, dtype=bool)[positions]

    rows = []
    if dataset_label is None and "train_test" in aligned.columns:
        split_values = aligned["train_test"].to_numpy()
        splits = ((0, "Train"), (1, "Test"))
    else:
        split_values = None
        splits = ((None, dataset_label or "All"),)
    for split_value, split_name in splits:
        selected = valid if split_value is None else valid & (split_values == split_value)
        y, pred, w = actual[selected], predicted[selected], exposure[selected]
        if not len(y) or w.sum() <= 0:
            continue

        average_basis = weight_basis == "average"
        order = np.argsort(pred if average_basis else pred / w, kind="mergesort")
        actual_total = w * y if average_basis else y
        cumulative_exposure = np.r_[0.0, np.cumsum(w[order]) / w.sum()]
        total_actual = np.sum(actual_total)
        if total_actual > 0 and np.all(y >= 0):
            cumulative_actual = np.r_[0.0, np.cumsum(actual_total[order]) / total_actual]
            area = np.sum((cumulative_actual[1:] + cumulative_actual[:-1]) * np.diff(cumulative_exposure) / 2.0)
            gini = 1.0 - 2.0 * area
        else:
            gini = np.nan

        # PoDE, RMSE and Within % are measured per unit of weight (frequency, average severity), weighted by the
        # weight, whatever the basis: totals are converted, so the null model scales with exposure and the three
        # weight settings give comparable numbers.
        if not average_basis:
            y, pred = y / w, pred / w
        null_prediction = np.full_like(y, np.average(y, weights=w))
        model_terms = unit_deviance(family, y, pred, tweedie_power)
        null_terms = unit_deviance(family, y, null_prediction, tweedie_power)
        pode = np.nan
        if model_terms is not None and null_terms is not None:
            null_deviance = np.average(null_terms, weights=w)
            if null_deviance > 0 and np.isfinite(null_deviance):
                pode = 1.0 - np.average(model_terms, weights=w) / null_deviance
        rmse = np.sqrt(np.average((y - pred) ** 2, weights=w))
        nonzero_actual = y != 0
        relative_error = np.abs(pred[nonzero_actual] - y[nonzero_actual]) / np.abs(y[nonzero_actual])
        tolerance_weights = w[nonzero_actual]
        within = [
            100.0 * np.average(relative_error <= threshold, weights=tolerance_weights)
            if len(relative_error) and tolerance_weights.sum() > 0
            else np.nan
            for threshold in (0.05, 0.10, 0.20)
        ]
        rows.append(
            {
                "Dataset": split_name,
                "GINI": gini,
                "PoDE": pode,
                "RMSE": rmse,
                "Within 5% (%)": within[0],
                "Within 10% (%)": within[1],
                "Within 20% (%)": within[2],
            }
        )
    return pd.DataFrame(rows)


OVERVIEW_STATISTICS = [
    "Total Weight",
    "Total Response",
    "Avg Actual - Weighted",
    "Avg Predicted - Weighted",
    "Minimum Predicted",
    "25% Quantile Predicted",
    "Mean Predicted",
    "75% Quantile Predicted",
    "Maximum Predicted",
]


def model_overview(
    d: pd.DataFrame,
    response: str,
    predictions: np.ndarray,
    weight: str = "N",
    pred_rows: np.ndarray | None = None,
    rows_mask: np.ndarray | None = None,
    weight_basis: str = "total",
) -> pd.DataFrame:
    """Summary of the actual and predicted response, one column per dataset (Train / Test, or All).

    Total Weight is the sum of the weight column (row counts for weight ``N``) and Total Response the sum of the
    response; Avg Actual - Weighted and
    Avg Predicted - Weighted are the actual and predicted per unit of weight: total / total weight when the response
    is a total (``weight_basis="total"``), else averages weighted by the weight column. The minimum, quartiles, mean and maximum describe the unweighted row-level predictions.
    Rows follow the same validity rules as :func:`prediction_performance_metrics`.

    """
    positions = _prediction_positions(pred_rows, len(d))
    aligned = d.iloc[positions]
    actual = aligned[response].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    exposure = np.ones(len(aligned)) if weight == "N" else aligned[weight].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(exposure) & (exposure > 0)
    if rows_mask is not None:
        valid &= np.asarray(rows_mask, dtype=bool)[positions]

    if "train_test" in aligned.columns:
        split_values = aligned["train_test"].to_numpy()
        splits = ((0, "Train"), (1, "Test"))
    else:
        split_values = None
        splits = ((None, "All"),)
    table = {"Statistic": OVERVIEW_STATISTICS}
    for split_value, split_name in splits:
        selected = valid if split_value is None else valid & (split_values == split_value)
        y, pred, w = actual[selected], predicted[selected], exposure[selected]
        if not len(y):
            table[split_name] = [0.0] + [np.nan] * (len(OVERVIEW_STATISTICS) - 1)
            continue
        table[split_name] = [
            float(w.sum()),
            float(y.sum()),
            float(y.sum() / w.sum() if weight_basis == "total" else np.average(y, weights=w)),
            float(pred.sum() / w.sum() if weight_basis == "total" else np.average(pred, weights=w)),
            float(pred.min()),
            float(np.quantile(pred, 0.25)),
            float(pred.mean()),
            float(np.quantile(pred, 0.75)),
            float(pred.max()),
        ]
    return pd.DataFrame(table)


_PARAMETER_ORDER = [
    "objective",
    "metric",
    "tweedie_variance_power",
    "learning_rate",
    "num_iterations",
    "best_iteration",
    "early_stopping_round",
    "num_leaves",
    "max_depth",
    "min_data_in_leaf",
    "bagging_fraction",
    "bagging_freq",
    "feature_fraction",
    "lambda_l1",
    "lambda_l2",
]
_PARAMETERS_SHOWN_ELSEWHERE = {"verbose", "monotone_constraints", "interaction_constraints", "num_threads", "seed"}

# what each parameter means and how to use it (shown when hovering over it in the GBM tab and the report)
PARAMETER_HELP = {
    "objective": "What the model is trained to predict. gamma, poisson and tweedie suit positive amounts or counts "
    "(log link); regression suits any continuous value. Choose the one that matches the response.",
    "metric": "The loss reported during training and used for early stopping. It usually matches the objective and "
    "does not change the fitted model.",
    "tweedie_variance_power": "Tweedie power between 1 and 2: nearer 1 behaves like counts (Poisson), nearer 2 like "
    "amounts (gamma). Only used by the tweedie objective.",
    "learning_rate": "How much each tree adds to the prediction. Lower is more reliable but needs more trees; if you "
    "lower it, raise num_iterations too.",
    "num_iterations": "The most trees that can be built. Training can stop earlier when early stopping is on.",
    "best_iteration": "The number of trees the model uses for predictions: the one with the best validation loss when "
    "early stopping is on.",
    "early_stopping_round": "Stop adding trees once the validation loss has not improved for this many trees. "
    "0 switches early stopping off.",
    "num_leaves": "The most leaves in each tree, which sets how complex each tree is. Higher captures more "
    "interactions but can overfit; keep it below 2 to the power of max_depth.",
    "max_depth": "The deepest any tree can be (-1 = no limit). Shallower trees limit interactions and overfitting.",
    "min_data_in_leaf": "The fewest rows a leaf may hold. Higher values smooth the model and stop it fitting small, "
    "noisy groups.",
    "bagging_fraction": "The share of rows sampled for each tree. Below 1 adds randomness and can reduce overfitting. "
    "Used together with bagging_freq.",
    "bagging_freq": "Resample the rows every this many trees (0 = never, which switches the sampling off).",
    "feature_fraction": "The share of features offered to each tree. Below 1 adds randomness and can reduce overfitting.",
    "lambda_l1": "L1 regularisation on the leaf values, which pushes small effects towards zero. 0 = off.",
    "lambda_l2": "L2 regularisation on the leaf values, which shrinks extreme values. 0 = off.",
    "response": "The column the model predicts.",
    "weight": "The exposure or weight column. N means every row counts once.",
    "weight used as": "How the weight enters the model: as a prior weight, with the response fitted per unit of weight "
    "(for totals such as claim counts or claim amounts) or taken as already per unit (averages).",
    "offset (init_score)": "An expected total for each row (e.g. an expected claim count that includes the exposure). "
    "It is divided by the weight and used as a starting value on the rate scale that every prediction starts from.",
    "starting level (link scale)": "The level of the model before any tree is added, fitted from the data when the "
    "model has an offset.",
    "number_of_features": "How many features the model uses.",
    "interaction_groups": "Groups of features allowed to interact. Features in different groups are never combined in "
    "the same tree. none = no restriction.",
    "shap_sample": "How many rows get SHAP values: 10k, All or No. The SHAP tab and the report need SHAP values.",
    "seed": "The random seed for bagging and feature sampling. The same seed rebuilds the same model.",
}


def parameter_help(name: str) -> str:
    """The hover text for a parameter row of the GBM parameter table."""
    return PARAMETER_HELP.get(str(name), "A LightGBM setting this model was built with.")


def _parameter_text(value) -> str:
    """Format a LightGBM parameter value for display: booleans as true or false, numbers in short form."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_parameter_text(v) for v in value) if value else "none"
    return str(value)


def gbm_parameter_table(model: GBMModel) -> pd.DataFrame:
    """The parameters a GBM was built with, as a two-column ``Parameter`` / ``Value`` table."""
    params, config = dict(model.params), dict(model.training_config or {})
    values = {
        **params,
        "num_iterations": config.get("num_iterations"),
        "early_stopping_round": config.get("early_stopping_round"),
        "best_iteration": getattr(model, "best_iteration", None),
    }
    rows = [(name, values[name]) for name in _PARAMETER_ORDER if values.get(name) is not None]
    extras = sorted(name for name in params if name not in _PARAMETER_ORDER and name not in _PARAMETERS_SHOWN_ELSEWHERE)
    rows += [(name, params[name]) for name in extras]
    rows += [
        ("response", model.response),
        ("weight", model.weight),
        ("weight used as", None if model.weight == "N" else WEIGHT_BASIS_SHORT[fit_basis(model)]),
        ("offset (init_score)", model.init_score),
        ("starting level (link scale)", getattr(model, "offset_base", 0.0) or None),
        ("number_of_features", len(model.features)),
        ("interaction_groups", list(model.interaction_groups or [])),
        ("shap_sample", config.get("shap_sample")),
        ("seed", params.get("seed", config.get("seed"))),
    ]
    return pd.DataFrame(
        [(name, _parameter_text(value)) for name, value in rows if value is not None], columns=["Parameter", "Value"]
    )


def gbm_monotonicity_table(model: GBMModel) -> pd.DataFrame:
    """The monotonicity constraint applied to each feature of a GBM (Increasing / Decreasing / None)."""
    constraints = list(model.params.get("monotone_constraints") or [0] * len(model.features))
    names = {1: "Increasing", -1: "Decreasing"}
    return pd.DataFrame(
        {"Feature": list(model.features), "Monotonicity": [names.get(int(c), "None") for c in constraints]}
    )


def compare_gbm_metrics(
    new: pd.DataFrame,
    old: pd.DataFrame,
) -> pd.DataFrame:
    """Compare metric tables and classify each change using the metric's better direction."""
    metric_columns = [column for column in new.columns if column != "Dataset"]
    new_long = new.melt(id_vars="Dataset", value_vars=metric_columns, var_name="Metric", value_name="New")
    old_long = old.melt(id_vars="Dataset", value_vars=metric_columns, var_name="Metric", value_name="Old")
    comparison = new_long.merge(old_long, on=["Dataset", "Metric"], how="inner")
    raw_change = comparison["New"] - comparison["Old"]
    comparison["Change"] = raw_change
    higher_is_better = {"GINI", "PoDE", "Within 5% (%)", "Within 10% (%)", "Within 20% (%)"}
    statuses = []
    status_rows = comparison[["Metric", "New", "Old"]].assign(Change=raw_change)
    for metric, new_value, old_value, change in status_rows.itertuples(index=False, name=None):
        if not np.isfinite(new_value) or not np.isfinite(old_value):
            statuses.append("Unavailable")
        elif np.isclose(new_value, old_value):
            statuses.append("Unchanged")
        else:
            improved = change > 0 if metric in higher_is_better else change < 0
            statuses.append("Improved" if improved else "Deteriorated")
    comparison["Status"] = statuses
    percentage_rows = comparison["Metric"].isin({"GINI", "RMSE"})
    comparison.loc[percentage_rows, "Change"] = np.divide(
        raw_change[percentage_rows] * 100,
        np.abs(comparison.loc[percentage_rows, "Old"]),
        out=np.full(percentage_rows.sum(), np.nan),
        where=np.abs(comparison.loc[percentage_rows, "Old"].to_numpy()) != 0,
    )
    return comparison


def gain_summary_from_trees(trees: pd.DataFrame) -> pd.DataFrame:
    """Gain summarised by the feature combination used in each tree."""
    splits = trees[trees["split_feature"].notna()]
    per_tree = splits.groupby("tree_index").agg(
        feats=("split_feature", lambda s: tuple(sorted(set(s)))), gain=("split_gain", "sum")
    )
    out = per_tree.groupby("feats")["gain"].agg(["sum", "count"]).reset_index()
    out["order"] = out["feats"].map(len)
    out["feature_1"] = out["feats"].map(lambda t: t[0] if len(t) > 0 else "")
    out["feature_2"] = out["feats"].map(lambda t: t[1] if len(t) > 1 else "")
    out["gain_%"] = 100 * out["sum"] / out["sum"].sum()
    return (
        out.sort_values("sum", ascending=False)
        .rename(columns={"sum": "gain", "count": "trees"})
        .drop(columns="feats")
        .reset_index(drop=True)
    )


def gbm_scoring_offset(
    model: GBMModel,
    d: pd.DataFrame,
) -> np.ndarray | None:
    """The starting value a GBM was trained with (its ``init_score`` column), for new data; None without one.

    The column is an expected total; a rate_weighted model divides it by the weight to get the rate-scale offset.
    """
    if model.init_score == "no offset":
        return None
    column = model.init_score
    if column not in d.columns:
        raise ValueError(f"the scoring data has no '{column}' column, which the model uses as its offset")
    offset = pd.to_numeric(d[column], errors="coerce").to_numpy(dtype=float)
    if model.weight != "N" and fit_basis(model) == "rate_weighted":
        if model.weight not in d.columns:
            raise ValueError(f"the scoring data has no '{model.weight}' column, which the model uses as its weight")
        with np.errstate(divide="ignore", invalid="ignore"):
            offset = offset / pd.to_numeric(d[model.weight], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if model.link == "log":
            offset = np.log(offset)
        elif model.link == "logit":
            offset = np.log(offset / (1 - offset))
    offset[~np.isfinite(offset)] = 0.0
    return offset


def score_gbm(
    model: GBMModel,
    d: pd.DataFrame,
) -> np.ndarray:
    """Response-scale predictions of a fitted GBM on new data (same offset convention as training)."""
    missing = [feature for feature in model.features if feature not in d.columns]
    if missing:
        raise ValueError("the scoring data is missing model features: " + ", ".join(missing))
    d = d.reset_index(drop=True)
    offset = gbm_scoring_offset(model, d)
    predictions = to_total_scale(model, d, model.predict(d, offset))
    if model.weight != "N" and fit_basis(model) == "rate_weighted":
        # no exposure gives an expected total of 0; a missing or negative weight gives a missing prediction
        predictions = zero_weight_predictions(predictions, d[model.weight])
    return predictions
