"""Plotly figure builders. plotly is imported lazily."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .banding import apply_treatment, band_numeric, bin_count, cap_numeric, resolve_treatment
from .gbm import _prediction_positions, model_weight_basis

LIFT_MAX_POINTS = 500
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def _go():
    """Import plotly.graph_objects on first use so the module loads without plotly."""
    import plotly.graph_objects as go

    return go


def ale_figure(
    result: pd.DataFrame,
    features: list[str],
    labels: list[str] | None = None,
):
    """Plotly line (one feature) or heatmap (two features) of ALE effects, with the feature labels as axis titles."""
    go = _go()
    labels = list(labels) if labels else list(features)  # axis titles (feature labels), default to names
    if len(features) == 1:
        fig = go.Figure(go.Scatter(x=result.index, y=result["eff"], mode="lines+markers", name="ALE"))
        fig.update_layout(xaxis_title=labels[0], yaxis_title="Centered effect on prediction")
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
    else:
        fig = go.Figure(
            go.Heatmap(
                x=result.index,
                y=result.columns,
                z=result.to_numpy().T,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(title="ALE"),
            )
        )
        fig.update_layout(xaxis_title=labels[0], yaxis_title=labels[1])
    fig.update_layout(template="plotly_white", margin=dict(l=50, r=30, t=30, b=50))
    return fig


def line_bar_figure(
    summary: pd.DataFrame,
    group_by: str,
    response: str,
    add_cols: list[str],
    show_bars: bool = True,
    sigma_bars: bool = False,
    show_labels: bool = False,
    title: str | None = None,
    y_title: str | None = None,
    weight_label: str | None = None,
    x_title: str | None = None,
):
    """Exposure bars (secondary axis) with response / prediction lines.

    The bars are each band's share of the total weight (weight / total weight, e.g. Exposure / Total Exposure),
    on a percentage axis; hovering a bar also shows the weight itself.

    ``y_title`` (the KPI name) titles the primary y-axis and labels the response line in the legend and
    hover text; it defaults to the response column name. ``weight_label`` (the weight column name) labels
    the exposure bars and the secondary axis; it defaults to "weight". ``x_title`` (the feature label)
    titles the x-axis; it defaults to the grouping column name.
    """
    go = _go()
    weight_label = weight_label or "weight"
    s = summary[summary[group_by].astype(str) != "Total"]
    x = s[group_by].astype(str) if not pd.api.types.is_numeric_dtype(s[group_by]) else s[group_by]
    fig = go.Figure()
    total = float(s["weight"].sum())
    share = s["weight"] / total if total > 0 else s["weight"] * 0.0
    share_label = f"Prop. of {weight_label}"
    if show_bars:
        fig.add_bar(
            x=x,
            y=share,
            customdata=s["weight"],
            name=share_label,
            marker_color="rgba(31, 119, 180, 0.4)",
            yaxis="y2",
            hovertemplate="%{y:.1%} (%{customdata:,.0f})<extra>" + share_label.replace("<", "&lt;") + "</extra>",
        )
    for i, c in enumerate([response] + [c for c in add_cols if c in s.columns]):
        kw = {}
        if sigma_bars and c == response and "sigma" in s:
            kw["error_y"] = dict(type="data", array=2 * s["sigma"].to_numpy(), visible=True, thickness=1)
        fig.add_scatter(
            x=x,
            y=s[c],
            mode="lines+markers" + ("+text" if show_labels else ""),
            name=(y_title or c) if c == response else c,
            line=dict(color=PALETTE[i % len(PALETTE)]),
            **kw,
            **({"text": s[c].round(2).astype(str), "textposition": "top center"} if show_labels else {}),
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50 if title else 20, b=40),
        yaxis=dict(title=y_title or response),
        yaxis2=dict(
            title=share_label, overlaying="y", side="right", showgrid=False, tickformat=".1%", rangemode="tozero"
        ),
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(title=x_title or group_by),
    )
    return fig


# ------------------------------------------------------------------ SHAP charts
def _shap_frame(
    d: pd.DataFrame,
    model,
    feature: str,
    weight: str,
    rows_mask: np.ndarray,
):
    """Join a model's SHAP column for ``feature`` to the data rows it was computed on."""
    col = f"lgbm_SHAP_{feature}"
    if model.SHAP is None or col not in model.SHAP.columns:
        raise ValueError("no SHAP values for this feature - rebuild the GBM with SHAP sampling on")
    idx = model.SHAP_rows
    df = d.iloc[idx].copy()
    df["shap"] = model.SHAP[col].to_numpy()
    df["_w"] = 1.0 if weight == "N" else df[weight].to_numpy(float)
    keep = rows_mask[idx]
    return df[keep]


def mean_shap_summary(
    d: pd.DataFrame,
    model,
    feature: str,
    weight: str,
    banding: float,
    rows_mask: np.ndarray,
) -> pd.DataFrame:
    """Exposure-weighted mean centered SHAP contributions or relativities by feature band."""
    frame = _shap_frame(d, model, feature, weight, np.asarray(rows_mask, dtype=bool)).copy()
    frame = frame[
        np.isfinite(frame["shap"]) & np.isfinite(frame["_w"]) & (frame["_w"] > 0) & frame[feature].notna()
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["level", "mean", "weight", "count"])
    if pd.api.types.is_numeric_dtype(frame[feature]) and not pd.api.types.is_bool_dtype(frame[feature]) and banding > 0:
        frame["level"] = band_numeric(frame[feature], banding, np.ones(len(frame), dtype=bool))
    else:
        frame["level"] = frame[feature].astype(object)
    centered = frame["shap"] - np.average(frame["shap"], weights=frame["_w"])
    frame["value"] = np.exp(centered) if model.link == "log" else centered
    frame["weighted_value"] = frame["value"] * frame["_w"]
    summary = frame.groupby("level", observed=True, dropna=False).agg(
        weight=("_w", "sum"), count=("value", "size"), weighted_value=("weighted_value", "sum")
    )
    summary["mean"] = summary.pop("weighted_value") / summary["weight"]
    return summary.reset_index()[["level", "mean", "weight", "count"]]


def mean_shap_figure(
    summary: pd.DataFrame,
    feature: str,
    link: str,
    x_title: str | None = None,
):
    """Line chart of the mean SHAP relativity by band for one feature, with a reference line at 1 (log link) or 0 (identity link)."""
    go = _go()
    fig = go.Figure(go.Scatter(x=summary["level"], y=summary["mean"], mode="lines+markers"))
    fig.add_hline(y=1.0 if link == "log" else 0.0, line_dash="dot", line_color="grey")
    fig.update_layout(
        template="plotly_white",
        title=f"{feature}: mean SHAP",
        xaxis_title=x_title or feature,
        yaxis_title="Mean SHAP relativity" if link == "log" else "Mean centered SHAP",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _positive_weight_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The rows with a weight above 0: a zero, negative or missing weight adds nothing to a weighted SHAP average."""
    return df[(df["_w"] > 0) & np.isfinite(df["_w"])]


def band_by_settings(
    values: pd.Series,
    banding,
    minimum=None,
    maximum=None,
    bins=None,
) -> pd.Series:
    """Band a feature for a SHAP chart the way the 1-way chart does: bins replace the banding, and a minimum or
    maximum caps the values first (values beyond them join the first or last band). Non-numeric features are text."""
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return values.astype(str)
    count = bin_count(bins)
    if count:
        return apply_treatment(values, resolve_treatment(values, minimum, maximum, count))
    capped = cap_numeric(values, minimum, maximum)
    if str(banding) != "0":
        return band_numeric(capped, float(banding), np.ones(len(values), bool))
    if minimum is not None or maximum is not None:
        return capped
    return values.astype(str)


def _no_rows_figure(title: str):
    """An empty figure with a title, for a SHAP chart with no rows of weight above 0 in the selection."""
    fig = _go().Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def shap_flame_figure(
    d: pd.DataFrame,
    model,
    feature: str,
    weight: str,
    banding: str,
    rows_mask: np.ndarray,
    quantile: float = 0.05,
    rebase: bool = True,
    weight_label: str | None = None,
    x_title: str | None = None,
    minimum=None,
    maximum=None,
    bins=None,
):
    """1-D SHAP: mean relativity by band with a (quantile, 1 - quantile) ribbon.

    ``weight_label`` (the weight column name) labels the exposure bars and the secondary axis; it defaults
    to "weight". ``minimum``, ``maximum`` and ``bins`` are the Charts settings, applied as in the 1-way chart.
    """
    go = _go()
    weight_label = weight_label or "weight"
    df = _positive_weight_frame(_shap_frame(d, model, feature, weight, rows_mask))
    if df.empty:
        return _no_rows_figure("SHAP: no rows with a weight above 0 in this selection")
    df["band"] = band_by_settings(df[feature], banding, minimum, maximum, bins)
    wmean = np.average(df["shap"], weights=df["_w"]) if rebase else 0.0
    df["rel"] = np.exp(df["shap"] - wmean) if model.link == "log" else df["shap"] - wmean
    agg = (
        df.groupby("band", observed=True)
        .apply(
            lambda t: pd.Series(
                dict(
                    mean=np.average(t["rel"], weights=t["_w"]),
                    lo=t["rel"].quantile(quantile),
                    hi=t["rel"].quantile(1 - quantile),
                    weight=t["_w"].sum(),
                )
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("band")
    )
    fig = go.Figure()
    fig.add_bar(
        x=agg["band"].astype(str),
        y=agg["weight"],
        yaxis="y2",
        marker_color="rgba(31, 119, 180, 0.4)",
        name=weight_label,
        hovertemplate="%{y:,.0f}<extra>" + weight_label.replace("<", "&lt;") + "</extra>",
    )
    bands = agg["band"].astype(str)
    fig.add_scatter(x=bands, y=agg["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(
        x=bands,
        y=agg["lo"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31,119,180,0.25)",
        name=f"{int(quantile*100)}%-{int((1-quantile)*100)}%",
    )
    fig.add_scatter(x=bands, y=agg["mean"], mode="lines+markers", name="mean SHAP relativity", line=dict(color=PALETTE[0]))
    fig.update_layout(
        template="plotly_white",
        title=f"SHAP: {feature}",
        xaxis_title=x_title or feature,
        yaxis_title="relativity" if model.link == "log" else "SHAP",
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False, title=weight_label, range=[0, float(agg["weight"].max()) * 3]
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def shap_heatmap_figure(
    d: pd.DataFrame,
    model,
    f1: str,
    f2: str,
    weight: str,
    b1: str,
    b2: str,
    rows_mask: np.ndarray,
    x_title: str | None = None,
    y_title: str | None = None,
    limits1=(None, None, None),
    limits2=(None, None, None),
):
    """2-D SHAP: mean combined SHAP relativity (f1 + f2 contributions) by band x band.

    ``limits1`` and ``limits2`` are (minimum, maximum, bins) for each feature, applied as in the 1-way chart."""
    go = _go()
    df = _shap_frame(d, model, f1, weight, rows_mask)
    df["shap2"] = model.SHAP[f"lgbm_SHAP_{f2}"].to_numpy()[rows_mask[model.SHAP_rows]]
    df = _positive_weight_frame(df)
    if df.empty:
        return _no_rows_figure("SHAP: no rows with a weight above 0 in this selection")
    df["b1"] = band_by_settings(df[f1], b1, *limits1)
    df["b2"] = band_by_settings(df[f2], b2, *limits2)
    tot = df["shap"] + df["shap2"]
    wm = np.average(tot, weights=df["_w"])
    df["val"] = np.exp(tot - wm) if model.link == "log" else tot - wm
    pv = df.pivot_table(index="b2", columns="b1", values="val", aggfunc="mean", observed=True)
    fig = go.Figure(
        go.Heatmap(
            z=pv.to_numpy(),
            x=[str(c) for c in pv.columns],
            y=[str(i) for i in pv.index],
            colorscale="RdBu_r",
            zmid=1.0 if model.link == "log" else 0.0,
            colorbar=dict(title="relativity"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"SHAP: {f1} x {f2}",
        xaxis_title=x_title or f1,
        yaxis_title=y_title or f2,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def stacked_shap_figure(
    d: pd.DataFrame,
    model,
    feature: str,
    weight: str,
    banding: str,
    rows_mask: np.ndarray,
    top_n: int = 5,
    x_title: str | None = None,
    minimum=None,
    maximum=None,
    bins=None,
):
    """What drives the prediction within each band of ``feature``: stacked mean SHAP of the top-n other features.
    ``minimum``, ``maximum`` and ``bins`` are the Charts settings, applied as in the 1-way chart."""
    go = _go()
    df = _shap_frame(d, model, feature, weight, rows_mask)
    df["band"] = band_by_settings(df[feature], banding, minimum, maximum, bins)
    cols = [c for c in model.SHAP.columns if c != "lgbm_SHAP_base_score"]
    S = model.SHAP[cols].iloc[np.flatnonzero(rows_mask[model.SHAP_rows])].reset_index(drop=True)
    S.columns = [c.replace("lgbm_SHAP_", "") for c in S.columns]
    S["band"] = pd.Series(df["band"].to_numpy(), dtype=df["band"].dtype)
    S["_w"] = df["_w"].to_numpy()
    keep = ((df["_w"] > 0) & np.isfinite(df["_w"])).to_numpy()
    if not keep.any():
        return _no_rows_figure("SHAP: no rows with a weight above 0 in this selection")
    S = S[keep].reset_index(drop=True)
    top = S.drop(columns=["band", "_w"]).abs().mean().sort_values(ascending=False).index[:top_n].tolist()
    agg = S.groupby("band", observed=True).apply(
        lambda t: pd.Series({c: np.average(t[c], weights=t["_w"]) for c in S.columns if c not in ("band", "_w")}),
        include_groups=False,
    )
    agg["other"] = agg.drop(columns=top).sum(axis=1)
    fig = go.Figure()
    for i, c in enumerate(top + ["other"]):
        fig.add_bar(x=agg.index.astype(str), y=agg[c], name=c)
    fig.update_layout(
        barmode="relative",
        template="plotly_white",
        title=f"Mean SHAP by {feature}",
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title=x_title or feature,
        yaxis_title="mean SHAP",
    )
    return fig


def evaluation_figure(model):
    """Learning curves (training and validation metrics by iteration) from a model's evaluation log."""
    go = _go()
    fig = go.Figure()
    for name, v in model.evaluation_log.items():
        fig.add_scatter(y=v, mode="lines", name=name)
    fig.add_vline(x=model.best_iteration - 1, line_dash="dot", annotation_text=f"best: {model.best_iteration}")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="tree",
        yaxis_title=model.params.get("metric", "metric"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


LIFT_BIN_CHOICES = (10, 20, 50, 100)
LIFT_DEFAULT_BINS = 20


def _split_arrays(
    d: pd.DataFrame,
    model,
    weight: str,
    rows_mask: np.ndarray | None = None,
):
    """Return (name, colour, actual, predicted, exposure) for train and test (or all rows), actual and predicted as totals."""
    positions = _prediction_positions(model.pred_rows, len(d))
    data = d.iloc[positions]
    actual = data[model.response].to_numpy(float)
    prediction = np.asarray(model.predictions, dtype=float)
    exposure = np.ones(len(data)) if weight == "N" else data[weight].to_numpy(float)
    keep = np.isfinite(actual) & np.isfinite(prediction) & np.isfinite(exposure) & (exposure > 0)
    if rows_mask is not None:
        keep &= np.asarray(rows_mask, dtype=bool)[positions]
    if model_weight_basis(model) == "average":
        # a response / prediction per unit of weight: express both as totals (like a ClaimCount response)
        actual, prediction = actual * exposure, prediction * exposure
    if "train_test" in data.columns:
        split = data["train_test"].to_numpy()
        splits = ((0, "Train", PALETTE[0]), (1, "Test", PALETTE[1]))
    else:
        split = None
        splits = ((None, "All", PALETTE[0]),)
    out = []
    for split_value, name, colour in splits:
        selected = keep if split_value is None else keep & (split == split_value)
        if selected.any() and exposure[selected].sum() > 0:
            out.append((name, colour, actual[selected], prediction[selected], exposure[selected]))
    return out


def lift_bins(
    actual: np.ndarray,
    predicted: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = LIFT_DEFAULT_BINS,
) -> pd.DataFrame:
    """Actual vs expected by equal-exposure bin, ranked by predicted rate (bin 1 = lowest predicted risk).

    ``actual`` and ``predicted`` are totals per row (e.g. claim counts); rates are divided by exposure.
    Lift is each bin's actual rate over the overall actual rate.
    """
    n_bins = max(1, int(n_bins))
    order = np.argsort(predicted / exposure, kind="mergesort")
    y, p, w = actual[order], predicted[order], exposure[order]
    midpoint = (np.cumsum(w) - w / 2.0) / w.sum()  # a row goes in the bin holding its exposure midpoint
    bin_index = np.minimum((midpoint * n_bins).astype(int), n_bins - 1)
    grouped = pd.DataFrame({"bin": bin_index, "w": w, "y": y, "p": p}).groupby("bin", sort=True).sum()
    overall_rate = y.sum() / w.sum()
    out = pd.DataFrame(
        {
            "Bin": grouped.index.to_numpy() + 1,
            "Exposure": grouped["w"].to_numpy(),
            "Actual": grouped["y"].to_numpy(),
            "Predicted": grouped["p"].to_numpy(),
        }
    )
    out["Actual rate"] = out["Actual"] / out["Exposure"]
    out["Predicted rate"] = out["Predicted"] / out["Exposure"]
    out["A/E"] = out["Actual"] / out["Predicted"]
    out["Lift"] = out["Actual rate"] / overall_rate if overall_rate > 0 else np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def lift_table_frame(
    d: pd.DataFrame,
    model,
    weight: str,
    rows_mask: np.ndarray | None = None,
    n_bins: int = LIFT_DEFAULT_BINS,
) -> pd.DataFrame:
    """A/E by bin for every dataset (Train / Test, or All), one row per dataset and bin."""
    frames = []
    for name, _, y, p, w in _split_arrays(d, model, weight, rows_mask):
        frame = lift_bins(y, p, w, n_bins)
        frame.insert(0, "Dataset", name)
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "Dataset",
                "Bin",
                "Exposure",
                "Actual",
                "Predicted",
                "Actual rate",
                "Predicted rate",
                "A/E",
                "Lift",
            ]
        )
    )


def lorenz_points(
    actual: np.ndarray,
    predicted: np.ndarray,
    exposure: np.ndarray,
):
    """Lorenz curve (cumulative exposure, cumulative actual; lowest predicted first) and its Gini, or None."""
    total = actual.sum()
    if not total > 0:
        return None
    order = np.argsort(predicted / exposure, kind="mergesort")
    cumulative_exposure = np.r_[0.0, np.cumsum(exposure[order]) / exposure.sum()]
    cumulative_actual = np.r_[0.0, np.cumsum(actual[order]) / total]
    area = np.sum((cumulative_actual[1:] + cumulative_actual[:-1]) * np.diff(cumulative_exposure) / 2.0)
    return cumulative_exposure, cumulative_actual, 1.0 - 2.0 * area


def lift_chart_figure(
    d: pd.DataFrame,
    model,
    weight: str,
    rows_mask: np.ndarray | None = None,
    n_bins: int = LIFT_DEFAULT_BINS,
):
    """Binned actual vs predicted by predicted-risk rank, train and test side by side.

    Bars are exposure per bin; lines are observed and predicted rate, with the portfolio mean as a reference
    and the top-bin lift (top-bin observed rate over the mean) in each panel title.
    """
    go = _go()
    from plotly.subplots import make_subplots

    n_bins = max(1, int(n_bins))
    panels = [
        (name, lift_bins(y, p, w, n_bins), (y.sum() / w.sum()))
        for name, _, y, p, w in _split_arrays(d, model, weight, rows_mask)
    ]
    if not panels:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Lift chart: no rows to chart")
        return fig

    count = len(panels)
    titles = []
    for name, table, mean in panels:
        top_lift = table["Lift"].iloc[-1]
        titles.append(f"{name}: top-bin lift {top_lift:.2f}x (mean {mean:.4g})" if np.isfinite(top_lift) else name)
    fig = make_subplots(
        rows=1, cols=count, specs=[[{"secondary_y": True}] * count], subplot_titles=titles, horizontal_spacing=0.16
    )
    # one rate scale for every panel so train and test read against each other
    rate_max = max(float(np.nanmax(t[["Actual rate", "Predicted rate"]].to_numpy())) for _, t, _ in panels)
    rate_max = max(rate_max, max(m for _, _, m in panels)) * 1.1
    marker_size = 7 if n_bins <= 20 else 4
    rate_title = f"{model.response} per {weight}" if weight != "N" else f"mean {model.response}"
    for col, (name, table, mean) in enumerate(panels, start=1):
        first = col == 1
        x = (table["Bin"].to_numpy() - 0.5) / n_bins
        fig.add_bar(
            x=x,
            y=table["Exposure"],
            width=np.full(len(x), 0.92 / n_bins),
            name="Exposure",
            marker_color="rgba(99, 110, 250, 0.35)",
            legendgroup="exposure",
            showlegend=first,
            secondary_y=False,
            row=1,
            col=col,
        )
        fig.add_scatter(
            x=[0, 1],
            y=[mean, mean],
            mode="lines",
            name="Portfolio mean",
            legendgroup="mean",
            line=dict(color="#777777", dash="dot"),
            showlegend=first,
            secondary_y=True,
            row=1,
            col=col,
        )
        fig.add_scatter(
            x=x,
            y=table["Actual rate"],
            mode="lines+markers",
            name="Observed",
            legendgroup="observed",
            line=dict(color=PALETTE[3]),
            marker=dict(size=marker_size),
            showlegend=first,
            secondary_y=True,
            row=1,
            col=col,
        )
        fig.add_scatter(
            x=x,
            y=table["Predicted rate"],
            mode="lines+markers",
            name="Predicted",
            legendgroup="predicted",
            line=dict(color=PALETTE[2]),
            marker=dict(size=marker_size),
            showlegend=first,
            secondary_y=True,
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Predicted risk rank (lowest to highest)", range=[0, 1], row=1, col=col)
        # rates on the left and drawn over the exposure bars, exposure on the right scaled to sit below the lines
        fig.update_yaxes(title_text=rate_title, side="left", range=[0, rate_max], secondary_y=True, row=1, col=col)
        fig.update_yaxes(
            title_text="Exposure",
            side="right",
            showgrid=False,
            range=[0, float(table["Exposure"].max()) * 3.0],
            secondary_y=False,
            row=1,
            col=col,
        )
    fig.update_layout(
        template="plotly_white",
        hovermode="x",
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    return fig


def lorenz_figure(
    d: pd.DataFrame,
    model,
    weight: str,
    rows_mask: np.ndarray | None = None,
):
    """Lorenz curves with the Gini of each dataset in the legend (lowest predicted risk first)."""
    go = _go()
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Equality", line=dict(color="#777777", dash="dot"))
    for name, colour, y, p, w in _split_arrays(d, model, weight, rows_mask):
        points = lorenz_points(y, p, w)
        if points is None:
            continue
        cumulative_exposure, cumulative_actual, gini = points
        # a few hundred points are enough; millions of rows would overwhelm the browser
        keep_points = np.unique(
            np.linspace(0, len(cumulative_exposure) - 1, min(len(cumulative_exposure), LIFT_MAX_POINTS)).astype(int)
        )
        fig.add_scatter(
            x=cumulative_exposure[keep_points],
            y=cumulative_actual[keep_points],
            mode="lines",
            name=f"{name} (Gini {gini:.3f})",
            line=dict(color=colour),
        )
    fig.update_layout(
        template="plotly_white",
        title="Lorenz curve",
        xaxis_title="Cumulative exposure share (lowest predicted first)",
        yaxis_title="Cumulative share of actual",
        margin=dict(l=50, r=20, t=50, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0),
    )
    return fig


def importance_height(rows: int) -> int:
    """Pixel height that gives every bar of an importance chart room for its label."""
    return max(260, 22 * int(rows) + 90)


def importance_figure(
    imp: pd.DataFrame,
    top: int | None = None,
    x_title: str = "gain %",
    value: str = "Gain",
):
    """Horizontal bars, most important feature at the top. Every feature is shown unless ``top`` limits them."""
    go = _go()
    t = (imp if top is None else imp.head(top)).iloc[::-1]
    fig = go.Figure(go.Bar(x=t[value], y=t["Feature"].astype(str), orientation="h", marker_color=PALETTE[0]))
    # no fixed height: the plot fills its container (callers set one with importance_height for long lists).
    # Every bar keeps its label (plotly otherwise skips some), and the left margin grows to fit long names.
    fig.update_layout(template="plotly_white", xaxis_title=x_title, margin=dict(l=100, r=20, t=30, b=40))
    fig.update_yaxes(type="category", tickmode="array", tickvals=list(t["Feature"].astype(str)), automargin=True)
    return fig


def ratebook_figure(
    table: pd.DataFrame,
    feature: str,
):
    """Step line of a ratebook table's relativity by level, with a reference line at 1.0."""
    go = _go()
    x = table["level"].astype(str) if not pd.api.types.is_numeric_dtype(table["level"]) else table["level"]
    fig = go.Figure(go.Scatter(x=x, y=table["relativity"], mode="lines+markers", line=dict(shape="hv")))
    fig.add_hline(y=1.0, line_dash="dot", line_color="grey")
    fig.update_layout(
        template="plotly_white",
        title=f"{feature}: relativity to base level",
        xaxis_title=feature,
        yaxis_title="relativity",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
