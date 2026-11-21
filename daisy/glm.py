"""GLM engine: Patsy formulas + IRLS.

Formula text uses Patsy's Python-style right-hand-side syntax. The weight column is a prior weight
(``rate_weighted`` / ``average``); the weight must be above 0 on every row.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import qr as scipy_qr
from .banding import apply_treatment
from .weights import (
    WEIGHT_BASES,
    NonPositiveResponseError,
    require_positive_weights,
    to_total_scale,
    zero_weight_predictions,
)


# --------------------------------------------------------------------------- formula parsing
def strip_comments(text: str) -> str:
    """Remove # comments from a model formula."""
    return re.sub(r"(?m)#.*$", "", text).strip()


def term_variables(
    term: str,
    columns,
) -> list[str]:
    """Columns used by a formula term, found by name (including Q('...') quoted names)."""
    quoted_columns = set(re.findall(r"Q\(['\"]([^'\"]+)['\"]\)", term))
    names = set(re.findall(r"[A-Za-z_][\w\.]*", re.sub(r"(['\"]).*?\1", "", term)))
    names.update(quoted_columns)
    return [c for c in columns if c in names]


# --------------------------------------------------------------------------- design matrix
class Design:
    """Build a Patsy model matrix and preserve its fitted encoding for future data."""

    def __init__(
        self,
        formula_text: str,
        base_levels: dict | None = None,
    ):
        """Store the formula, rejecting an empty one once comments are removed."""
        self.formula_text = strip_comments(formula_text)
        if not self.formula_text:
            raise ValueError("check model formula: no terms")
        self.formula = self.formula_text
        self.terms: list[str] = []
        self.base_levels = base_levels or {}
        self.design_info = None
        self.columns: list[str] = []
        self.term_of_column: list[int] = []
        self.col_means: np.ndarray | None = None
        self.fitted = False
        # feature -> minimum / maximum caps and bins, applied to the data before the model matrix is
        # built (when fitting and whenever the model predicts)
        self.treatments: dict = {}

    def _data_with_base_levels(
        self,
        d: pd.DataFrame,
    ) -> pd.DataFrame:
        """Copy the data with each feature's treatment (such as its base level) applied."""
        data = d.reset_index(drop=True).copy()
        for column, treatment in (getattr(self, "treatments", None) or {}).items():
            if column in data:
                data[column] = apply_treatment(data[column], treatment)
        for column, base in self.base_levels.items():
            if column not in data:
                continue
            series = data[column]
            if isinstance(series.dtype, pd.CategoricalDtype):
                levels = list(series.cat.categories)
            else:
                levels = sorted(series.dropna().unique(), key=str)
            base_level = next((level for level in levels if str(level) == str(base)), None)
            if base_level is not None:
                categories = [base_level] + [level for level in levels if level != base_level]
                data[column] = pd.Categorical(series, categories=categories, ordered=True)
        if self.design_info is not None:
            for factor, factor_info in self.design_info.factor_infos.items():
                if factor_info.type != "categorical":
                    continue
                factor_name = factor.name()
                if factor_name in data.columns:
                    columns = [factor_name]
                else:
                    match = re.match(r"\s*C\(\s*(?:Q\(['\"]([^'\"]+)['\"]\)|([A-Za-z_]\w*))", factor_name)
                    columns = [next(group for group in match.groups() if group)] if match else []
                for column in columns:
                    data[column] = pd.Categorical(data[column], categories=factor_info.categories, ordered=True)
        return data

    def build(
        self,
        d: pd.DataFrame,
        fit: bool,
    ) -> pd.DataFrame:
        """Build the design matrix with patsy. fit=True sets up the design from the data; fit=False reuses it for new data."""
        if fit:
            import patsy

            X = patsy.dmatrix(
                self.formula,
                self._data_with_base_levels(d),
                return_type="dataframe",
                NA_action="raise",
                eval_env=patsy.EvalEnvironment([{"np": np}]),
            )
            self.design_info = X.design_info
            term_slices = self.design_info.term_name_slices
            self.terms = [term for term in term_slices if term != "Intercept"]
            self.columns = list(X.columns)
            self.term_of_column = [-1] * len(self.columns)
            for term_idx, term in enumerate(self.terms):
                for column_idx in range(term_slices[term].start, term_slices[term].stop):
                    self.term_of_column[column_idx] = term_idx
            X = X.rename(columns={"Intercept": "(Intercept)"})
            self.columns = list(X.columns)
            self.fitted = True
            return X
        import patsy

        matrices = patsy.build_design_matrices(
            [self.design_info], self._data_with_base_levels(d), return_type="dataframe", NA_action="raise"
        )
        return matrices[0].rename(columns={"Intercept": "(Intercept)"})


# --------------------------------------------------------------------------- families & IRLS
@dataclass
class Family:
    name: str
    link: str  # 'identity' | 'log' | 'logit'
    var_power: float | None = None
    estimate_dispersion: bool = True

    # link functions
    def linkfun(
        self,
        mu,
    ):
        """Link function: maps means to the linear predictor (identity, log or logit)."""
        return {"identity": lambda m: m, "log": np.log, "logit": lambda m: np.log(m / (1 - m))}[self.link](mu)

    def linkinv(
        self,
        eta,
    ):
        """Inverse link: maps the linear predictor back to the mean (identity, exp or logistic)."""
        if self.link == "identity":
            return eta
        if self.link == "log":
            return np.exp(np.clip(eta, -700, 700))
        return 1 / (1 + np.exp(-np.clip(eta, -700, 700)))

    def mu_eta(
        self,
        eta,
    ):
        """Derivative of the mean with respect to the linear predictor, used in IRLS."""
        if self.link == "identity":
            return np.ones_like(eta)
        if self.link == "log":
            return np.exp(np.clip(eta, -700, 700))
        e = np.exp(-np.abs(eta))
        return np.maximum(e / (1 + e) ** 2, 2.2e-16)

    def variance(
        self,
        mu,
    ):
        """Variance function of the family at the means mu."""
        n = self.name
        if n == "gaussian":
            return np.ones_like(mu)
        if n in ("poisson", "quasipoisson"):
            return mu
        if n == "gamma":
            return mu**2
        if n == "tweedie":
            return mu**self.var_power
        if n == "binomial":
            return mu * (1 - mu)
        raise ValueError(n)

    def dev_resids(
        self,
        y,
        mu,
        wt,
    ):
        """Deviance residuals for observations y, fitted means mu and prior weights wt."""
        n = self.name
        if n == "gaussian":
            return wt * (y - mu) ** 2
        if n in ("poisson", "quasipoisson"):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
            return 2 * wt * (t - (y - mu))
        if n == "gamma":
            return -2 * wt * (np.log(np.where(y == 0, 1, y / mu)) - (y - mu) / mu)
        if n == "tweedie":
            p = self.var_power
            y1 = np.where(y > 0, y, 0.0)
            return (
                2
                * wt
                * (
                    np.where(y > 0, y1 ** (2 - p) / ((1 - p) * (2 - p)), 0.0)
                    - y * mu ** (1 - p) / (1 - p)
                    + mu ** (2 - p) / (2 - p)
                )
            )
        if n == "binomial":
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
                t2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
            return 2 * wt * (t1 + t2)
        raise ValueError(n)

    def mustart(
        self,
        y,
    ):
        """Starting means for IRLS, kept away from zero or one where the family needs it."""
        n = self.name
        if n in ("poisson", "quasipoisson", "tweedie"):
            return y + 0.1
        if n == "binomial":
            return (y + 0.5) / 2
        if n == "gaussian" and self.link == "log":
            return np.maximum(y, 1e-3)
        return np.asarray(y, float).copy()


GLM_OBJECTIVES = {
    "gaussian": Family("gaussian", "identity"),
    "binomial": Family("binomial", "logit", estimate_dispersion=False),
    "poisson": Family("poisson", "log", estimate_dispersion=False),
    "gaussian (log link)": Family("gaussian", "log"),
    "gamma": Family("gamma", "log"),
    "tweedie": Family("tweedie", "log", var_power=1.5),
    "quasipoisson": Family("quasipoisson", "log"),
}


def make_family(
    objective: str,
    var_power: float | None = None,
) -> Family:
    """Build the GLM family (variance, link and deviance) for an objective. Tweedie needs a variance power between 1 and 2."""
    f = GLM_OBJECTIVES[objective]
    if objective == "tweedie":
        if var_power is None or not np.isfinite(var_power):
            raise ValueError("specify numeric var.power for Tweedie (between 1 and 2)")
        return Family("tweedie", "log", var_power=float(var_power))
    return Family(f.name, f.link, f.var_power, f.estimate_dispersion)


def _independent_columns(
    X: np.ndarray,
    tol: float = 1e-7,
):
    """Pivoted QR rank detection -> aliased columns get NA coefficients."""
    _, R, piv = scipy_qr(X, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    rank = int((d > tol * d[0]).sum()) if d.size else 0
    return np.sort(piv[:rank])


@dataclass
class GLMFit:
    columns: list
    coef: np.ndarray  # NaN for aliased
    se: np.ndarray
    pvalue: np.ndarray
    deviance: float
    null_deviance: float
    aic: float
    dispersion: float
    df_residual: int
    n_obs: int
    iterations: int
    converged: bool
    family: Family
    cov: np.ndarray | None = None


def fit_irls(
    X: pd.DataFrame,
    y,
    family: Family,
    maxit: int = 25,
    epsilon: float = 1e-8,
    prior_weights=None,
) -> GLMFit:
    """IRLS fit on the link scale. ``prior_weights`` weight the rows."""
    cols = list(X.columns)
    Xm = X.to_numpy(float)
    y = np.asarray(y, float)
    n = len(y)
    wt = np.ones(n) if prior_weights is None else np.asarray(prior_weights, float)
    keep = _independent_columns(Xm)
    Xk = Xm[:, keep]
    mu = family.mustart(y)
    eta = family.linkfun(mu)
    dev_old = family.dev_resids(y, mu, wt).sum()
    beta = np.zeros(Xk.shape[1])
    converged = False
    it = 0
    for it in range(1, maxit + 1):
        me = family.mu_eta(eta)
        var = family.variance(mu)
        z = eta + (y - mu) / me
        w = np.sqrt(wt * me**2 / var)
        sol, *_ = np.linalg.lstsq(Xk * w[:, None], z * w, rcond=None)
        eta_new = Xk @ sol
        mu_new = family.linkinv(eta_new)
        dev = family.dev_resids(y, mu_new, wt).sum()
        step = 0
        while (not np.isfinite(dev)) and step < 20:  # step-halving
            sol = (sol + beta) / 2
            eta_new = Xk @ sol
            mu_new = family.linkinv(eta_new)
            dev = family.dev_resids(y, mu_new, wt).sum()
            step += 1
        beta, eta, mu = sol, eta_new, mu_new
        if abs(dev - dev_old) / (abs(dev) + 0.1) < epsilon:
            converged = True
            dev_old = dev
            break
        dev_old = dev
    dev = dev_old
    me = family.mu_eta(eta)
    W = wt * me**2 / family.variance(mu)
    XtWX = (Xk * W[:, None]).T @ Xk
    cov_unscaled = np.linalg.inv(XtWX)
    rank = len(keep)
    df_res = n - rank
    if family.estimate_dispersion:
        pearson = (wt * (y - mu) ** 2 / family.variance(mu)).sum()
        disp = pearson / df_res if df_res > 0 else np.nan
    else:
        disp = 1.0
    se_k = np.sqrt(np.diag(cov_unscaled) * disp)
    coef = np.full(len(cols), np.nan)
    se = np.full(len(cols), np.nan)
    coef[keep] = beta
    se[keep] = se_k
    with np.errstate(invalid="ignore", divide="ignore"):
        tval = coef / se
    pval = 2 * (stats.t.sf(np.abs(tval), df_res) if family.estimate_dispersion else stats.norm.sf(np.abs(tval)))
    # null deviance (intercept only, same link)
    null_mu = np.full(n, np.average(y, weights=wt))
    null_dev = family.dev_resids(y, null_mu, wt).sum()
    aic = _aic(family, y, mu, wt, dev, rank, disp)
    return GLMFit(
        cols,
        coef,
        se,
        pval,
        float(dev),
        float(null_dev),
        aic,
        float(disp),
        int(df_res),
        n,
        it,
        converged,
        family,
        cov_unscaled * disp,
    )


def _aic(
    fam: Family,
    y,
    mu,
    wt,
    dev,
    rank,
    disp,
):
    """AIC using the prior weights ``wt`` (all 1 without prior weights).

    With prior weights, ``y`` and ``mu`` are per unit of weight, so the counts are ``y * wt`` with mean ``mu * wt``
    (a Poisson of the totals, e.g. claim counts with exposure). NaN where there is no likelihood to evaluate: quasi /
    Tweedie families and a weighted binomial.
    """
    n = len(y)
    wt = np.asarray(wt, float)
    weighted = not np.allclose(wt, 1.0)
    if fam.name == "poisson":
        return float(-2 * stats.poisson.logpmf(np.round(y * wt), mu * wt).sum() + 2 * rank)
    if fam.name == "binomial":
        if weighted:
            return float("nan")
        from scipy.special import xlogy, xlog1py

        return float(-2 * (xlogy(y, mu) + xlog1py(1 - y, -mu)).sum() + 2 * rank)
    if fam.name == "gaussian":
        nobs = n
        return float(nobs * (np.log(2 * np.pi * dev / nobs) + 1) + 2 - np.log(wt).sum() + 2 * rank)
    if fam.name == "gamma":
        d = dev / wt.sum()
        shape = 1 / d
        return float(-2 * (wt * stats.gamma.logpdf(y, a=shape, scale=mu * d)).sum() + 2 + 2 * rank)
    return float("nan")  # quasi / tweedie: no AIC


# --------------------------------------------------------------------------- model object
@dataclass
class GLMModel:
    design: Design
    fit: GLMFit
    response: str
    weight: str
    objective: str
    var_power: float | None
    formula: str
    rows_used: np.ndarray
    predictions: np.ndarray | None = None
    pred_rows: np.ndarray | None = None
    link: str = "log"
    name: str = "GLM"
    weight_basis: str = "rate_weighted"  # rate_weighted / average: the weight is a prior weight

    # -- prediction
    def linear_predictor(
        self,
        d: pd.DataFrame,
    ) -> np.ndarray:
        """Linear predictor (link scale) for new rows, using the fitted coefficients (missing ones count as zero)."""
        X = self.design.build(d.reset_index(drop=True), fit=False)
        beta = np.nan_to_num(self.fit.coef)
        return X.to_numpy(float) @ beta

    def predict(
        self,
        d: pd.DataFrame,
    ) -> np.ndarray:
        """Response-scale predictions: the rate per unit of weight for a model fitted with prior weights."""
        return self.fit.family.linkinv(self.linear_predictor(d))

    def score(
        self,
        d: pd.DataFrame,
    ) -> np.ndarray:
        """Predictions on the scale the model was fitted on: totals for a rate_weighted model, per unit for average."""
        predictions = to_total_scale(self, d, self.predict(d))
        if self.weight != "N" and self.weight_basis == "rate_weighted":
            # no exposure gives an expected total of 0; a missing or negative weight gives a missing prediction
            predictions = zero_weight_predictions(predictions, d[self.weight])
        return predictions

    def coefficients(self) -> pd.DataFrame:
        """Table of coefficient estimates, standard errors and p-values for each term."""
        f = self.fit
        return pd.DataFrame({"term": f.columns, "estimate": f.coef, "std.error": f.se, "p.value": f.pvalue})

    def term_contributions(
        self,
        d: pd.DataFrame,
    ) -> pd.DataFrame:
        """Centred per-term linear-predictor contributions."""
        X = self.design.build(d.reset_index(drop=True), fit=False)
        beta = np.nan_to_num(self.fit.coef)
        centred = X.to_numpy(float) - self.design.col_means
        out = {}
        for i, term in enumerate(self.design.terms):
            idx = [j for j, t in enumerate(self.design.term_of_column) if t == i]
            out[term] = centred[:, idx] @ beta[idx] if idx else np.zeros(len(X))
        return pd.DataFrame(out)

    def feature_contributions(
        self,
        d: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sum term contributions by feature -> glm_LP_<feature> columns."""
        tc = self.term_contributions(d)
        feats = sorted({v for t in self.design.terms for v in term_variables(t, d.columns)})
        out = {}
        for f in feats:
            cols = [t for t in self.design.terms if f in term_variables(t, d.columns)]
            out[f"glm_LP_{f}"] = tc[cols].sum(axis=1).to_numpy()
        return pd.DataFrame(out)

    def features(
        self,
        columns,
    ) -> list[str]:
        """Sorted feature names used by the model's terms, found among the given columns."""
        return sorted({v for t in self.design.terms for v in term_variables(t, columns)})


def build_glm(
    d: pd.DataFrame,
    response: str,
    weight: str,
    formula_text: str,
    objective: str = "gamma",
    var_power: float | None = None,
    data_to_use: str = "All",
    base_levels: dict | None = None,
    weight_basis: str = "rate_weighted",
    treatments: dict | None = None,
    positive_response: bool = False,
) -> GLMModel:
    """Fit a GLM (all rows or training rows; zero-weight rows removed).

    ``treatments`` (feature -> ``banding.resolve_treatment``) caps features at a minimum / maximum or turns them
    into bins before fitting; the model keeps them and applies them to whatever it predicts.

    ``weight_basis`` says what the response is when a weight is chosen. ``rate_weighted`` (e.g. ClaimCount or total
    claim amount with exposure as the weight): the response is a total, the model is fitted on response / weight with
    the weight as a prior weight, and predictions are returned as totals. ``average``: the response is already per
    unit of weight, the weight is a prior weight and predictions are per unit.
    """
    if response not in d.columns:
        raise ValueError("no response selected")
    if weight != "N" and weight not in d.columns:
        raise ValueError("no weight selected")
    if data_to_use == "Training" and "train_test" not in d.columns:
        raise ValueError("Training selected but there is no train_test column in the dataset")
    formula_text = strip_comments(formula_text)
    if not formula_text:
        raise ValueError("check model formula: no terms")
    if response in term_variables(formula_text, d.columns):
        raise ValueError("response in formula")
    d = d.reset_index(drop=True)
    include = np.arange(len(d)) if data_to_use == "All" else np.flatnonzero(d["train_test"].to_numpy() == 0)
    if weight != "N":
        nz = np.flatnonzero(d[weight].to_numpy(float) > 0)
        include = np.intersect1d(include, nz)
    else:
        nz = np.arange(len(d))
    # rows that can be used: a response and a weight above 0 (the gamma objective needs a response above 0)
    weight_ok = np.ones(len(d), bool) if weight == "N" else d[weight].to_numpy(float) > 0
    raw_response = d[response].to_numpy(float)
    response_ok = np.isfinite(raw_response) & (raw_response > 0)
    usable = weight_ok & response_ok
    if objective == "gamma" and not positive_response and not response_ok.all():
        # the rows left out are the union of a bad response and a bad weight: one count
        raise NonPositiveResponseError(int(np.sum(~usable)), len(d))
    if not positive_response:
        require_positive_weights(d, weight)
    else:  # build on the usable rows only; the data itself is not changed
        include = np.intersect1d(include, np.flatnonzero(usable))
        nz = np.flatnonzero(usable)
    family = make_family(objective, var_power)
    if weight_basis not in WEIGHT_BASES:
        raise ValueError(f"weight basis must be one of {list(WEIGHT_BASES)}")
    prior_weights = None
    if weight != "N":
        weights = d[weight].to_numpy(float)
        prior_weights = weights
    design = Design(formula_text, base_levels)
    used = set(term_variables(formula_text, d.columns)) - {response, weight}
    design.treatments = {f: t for f, t in (treatments or {}).items() if f in used and t}
    X = design.build(d.iloc[include], fit=True)
    design.col_means = X.to_numpy(float).mean(axis=0)
    y = d[response].to_numpy(float)[include]
    if weight != "N" and weight_basis == "rate_weighted":
        y = y / weights[include]  # fit the response per unit of weight (the weights are positive for included rows)
    fit = fit_irls(
        X,
        y,
        family,
        prior_weights=None if prior_weights is None else prior_weights[include],
    )
    model = GLMModel(
        design,
        fit,
        response,
        weight,
        objective,
        var_power,
        formula_text,
        include,
        link=family.link,
        weight_basis=weight_basis,
    )
    model.pred_rows = nz
    model.predictions = model.predict(d.iloc[nz])
    if weight != "N" and weight_basis == "rate_weighted":
        model.predictions = model.predictions * weights[nz]  # stored as totals, like the response
    return model
