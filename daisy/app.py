"""Daisy: Shiny for Python insurance modelling app.

Run the demo:   shiny run app.py            (see the project-root app.py)
Use your data:  from daisy.app import make_app; app = make_app(my_dataframe, starting_response="price")
"""
from __future__ import annotations
import asyncio
import os
import re
import subprocess
from itertools import combinations
import io
import json
import pickle
import tempfile
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from shiny import App, reactive, render, req, ui
from shiny.types import SilentCancelOutputException, SilentException
from shinywidgets import output_widget, render_widget

from . import charts, report
from .ale import calculate_ale
from .banding import banding_guesser, bin_count, resolve_treatment, treatment_text, treatments_from_spec
from .data import DATA_DIR, demo_spec_paths, numerical_cols
from .filters import apply_filter, filter_text
from .gbm import (
    LGBM_OBJECTIVES,
    NonPositiveResponseError,
    build_gbm,
    parameter_help,
    compare_gbm_metrics,
    gbm_performance_metrics,
    DEVIANCE_OBJECTIVES,
    gbm_monotonicity_table,
    gbm_parameter_table,
    model_objective_power,
    model_overview,
    model_performance_metrics,
    model_weight_basis,
    normalise_objective,
    prediction_performance_metrics,
    score_gbm,
)
from .glm import GLM_OBJECTIVES, build_glm, formula_from_import, formula_to_text
from .filedialog import pick_file
from .glm_builder import builder_ui, feature_chips
from .ratebook import check_ratebook_link, tabulate_model, write_ratebook_excel
from .scoring import (
    iter_csv_chunks,
    model_objective,
    read_gbm_pickle,
    read_ratebook,
    score_metrics,
    score_ratebook,
    scored_frame,
)
from .specs import (
    REQUIRED,
    feature_monotonicity,
    set_feature_monotonicity,
    feature_label,
    feature_lookup,
    feature_treatment_spec,
    set_feature_treatment,
    filter_choices,
    filter_select_choices,
    load_specification,
    numeric_or_none,
    spec_chart_limits,
    specification_template,
)
from .summary import add_total_row, column_summary, one_way_summary
from .overlay import (
    Overlay,
    RULE as OVERLAY_RULE,
    apply_overlay,
    band_label,
    band_starts,
    band_table,
    adjustments_workbook,
    category_levels,
    bootstrap_intervals,
    check_relativities,
    evaluate as evaluate_overlay,
    suggest,
    apply_overlays,
)
from .weights import WEIGHT_BASES, WEIGHT_BASIS_HELP, to_total_scale

SPECIAL_COLS = {"train_test"}


def _show(
    df: pd.DataFrame,
    nd: int = 4,
) -> pd.DataFrame:
    """Make a DataFrame safe/pleasant for render.DataGrid."""
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype) or out[c].dtype == object:
            out[c] = out[c].astype(str)
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(nd)
    return out


def _prediction_export_frame(
    data: pd.DataFrame,
    response: str,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Export frame of the row index, train/test flag, response and prediction for the data."""
    out = {"row_index": np.arange(len(data))}
    if "train_test" in data.columns:
        out["train_test"] = data["train_test"].to_numpy()
    out[response] = data[response].to_numpy()
    out["prediction"] = np.asarray(predictions, dtype=float)
    return pd.DataFrame(out)


def _validate_editor_spec(
    frame: pd.DataFrame,
    spec_type: str,
    columns: list[str],
) -> pd.DataFrame:
    """Check an edited specification (feature, KPI or filter) for required columns, names and duplicates, and return it cleaned."""
    frame = frame.copy().fillna("")
    missing = [column for column in REQUIRED[spec_type] if column not in frame.columns]
    if missing:
        raise ValueError(f"{spec_type} specification is missing required columns: {', '.join(missing)}")
    if spec_type in {"feature", "kpi"} and frame.empty:
        raise ValueError(f"{spec_type} specification must contain at least one row")
    if spec_type == "feature":
        features = frame["feature"].astype(str).str.strip()
        if features.eq("").any():
            raise ValueError("feature specification contains an empty feature name")
        unknown = sorted(set(features) - set(columns))
        if unknown:
            raise ValueError(f"unknown feature columns: {', '.join(unknown)}")
        if features.duplicated().any():
            raise ValueError("feature specification contains duplicate feature names")
    if spec_type == "kpi" and frame["kpi_name"].astype(str).str.strip().eq("").any():
        raise ValueError("KPI names cannot be empty")
    return frame


def _parse_editor_spec(
    text: str,
    spec_type: str,
    columns: list[str],
) -> pd.DataFrame:
    """Parse a specification typed as CSV text (filters as one expression per line) and validate it."""
    if spec_type == "filter":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and lines[0].lower() == "filter":
            lines = lines[1:]
        frame = pd.DataFrame({"filter": lines or ["no filter"]})
    else:
        frame = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
    return _validate_editor_spec(frame, spec_type, columns)


def _read_editor_grid(
    renderer,
    current: pd.DataFrame,
    spec_type: str,
    columns: list[str],
) -> pd.DataFrame:
    """Read the rows of an editable specification grid, falling back to the current specification when the grid has not been edited, and validate them."""
    try:
        edited = renderer.data_patched()
    except (SilentException, SilentCancelOutputException):
        edited = current
    return _validate_editor_spec(edited, spec_type, columns)


def _gbm_features_from_spec(
    feature_spec: pd.DataFrame,
    columns: list[str],
    response: str,
    weight: str,
) -> list[str]:
    """Features the feature specification marks as modelling-valid, excluding the response, weight and special columns."""
    if "modelling_valid" not in feature_spec.columns:
        return []
    included = feature_spec["modelling_valid"].astype(str).str.strip().str.lower().eq("feature")
    excluded = {response, weight, *SPECIAL_COLS}
    return [
        feature
        for feature in feature_spec.loc[included, "feature"].astype(str)
        if feature in columns and feature not in excluded
    ]


def export_filename(
    base: str,
    kpi: str | None,
    extension: str,
) -> str:
    """``base - KPI name.extension`` for an export, with characters that are not allowed in file names removed."""
    name = f"{base} - {kpi}" if kpi else base
    name = re.sub(r'[\\/:*?"<>|]+', " ", name).strip()
    return name + extension


def _feature_step(select_id: str):
    """Prev and Next buttons that step through a feature drop-down (in the order it lists its features)."""
    script = (
        "function daisyStepSelect(id,step){var s=document.getElementById(id);if(!s){return;}"
        "var i=s.selectedIndex+step;if(i<0||i>=s.options.length){return;}s.selectedIndex=i;"
        "if(window.jQuery){window.jQuery(s).trigger('change');}"
        "else{s.dispatchEvent(new Event('change',{bubbles:true}));}}"
    )
    return ui.div(
        ui.tags.button(
            "\u2039 Prev",
            type="button",
            class_="btn btn-default btn-sm text-nowrap px-2",
            onclick=f"daisyStepSelect('{select_id}',-1)",
            title="Previous feature",
        ),
        ui.tags.button(
            "Next \u203a",
            type="button",
            class_="btn btn-default btn-sm text-nowrap px-2",
            onclick=f"daisyStepSelect('{select_id}',1)",
            title="Next feature",
        ),
        ui.tags.script(script),
        class_="d-flex gap-2",
        style="margin-top: -0.5rem;",
    )


def _process_memory_mb() -> float | None:
    """The memory the app process uses now, in MB, on Windows, macOS and Linux (None when it cannot be read).

    Uses psutil when it is installed; otherwise the Windows API on Windows and ``ps`` on macOS and Linux."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        pass
    except Exception:
        return None
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return None
            return counters.WorkingSetSize / 2**20
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) / 1024  # ps reports kilobytes on macOS and Linux
    except Exception:
        return None


_RANKING: dict[int, tuple[object, list[str]]] = {}
NOT_NON_MODELLED = ("train_test",)  # left out of the Non-modelled features too (with the response and weight)
TWO_WAY_FEATURES = 5  # default number of top features whose pairs get a two-way SHAP and ALE view in the report


def scoring_model_features(
    loaded: dict,
    data: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[str]:
    """The features the model being scored with uses, most important first. ``reference`` is the data the model was
    built on (its importance comes from that data). A ratebook keeps the order of its features."""
    model = loaded["model"]
    if loaded["kind"] == "ratebook":
        return list(model.features)
    if loaded["kind"] == "glm":
        return _ranked_features(reference, model, "glm", list(model.features(data.columns)))
    return _ranked_features(reference, model, "gbm", list(model.features))


def _ranked_features(
    data: pd.DataFrame,
    model,
    family: str,
    features: list[str],
) -> list[str]:
    """``features`` in the order of the model's feature importance, most important first. Features the importance
    does not list keep their order at the end. The importance of each model is worked out once."""
    cached = _RANKING.get(id(model))
    if cached is None or cached[0] is not model:
        try:
            if family == "glm":  # by feature: each feature's terms summed (the importance table lists terms)
                contributions = model.feature_contributions(data.iloc[model.pred_rows])
                mean_abs = contributions.abs().mean().sort_values(ascending=False)
                order = [c.replace("glm_LP_", "") for c in mean_abs.index]
            else:
                order = report.importance_frame(data, model, family)["Feature"].astype(str).tolist()
        except Exception:
            order = []
        cached = (model, order)
        _RANKING[id(model)] = cached
    ranked = [feature for feature in cached[1] if feature in features]
    return ranked + [feature for feature in features if feature not in ranked]


def _kpi_defaults(
    kpi_df: pd.DataFrame,
    num_cols: list[str],
    starting_response: str | None,
):
    """Launch defaults ``(kpi_name, response, weight)`` taken from a provided KPI specification.

    A KPI is usable when its numerator is a numeric column and its denominator is ``N`` or a numeric column.
    With a ``starting_response`` the first usable KPI for that response is chosen (its denominator becomes the
    weight); otherwise the first usable KPI is. Returns ``None`` when no KPI can be used.
    """
    usable = [
        row
        for row in kpi_df.to_dict("records")
        if row["kpi_numerator"] in num_cols and (row["kpi_denominator"] == "N" or row["kpi_denominator"] in num_cols)
    ]
    if starting_response in num_cols:
        for row in usable:
            if row["kpi_numerator"] == starting_response:
                return row["kpi_name"], starting_response, row["kpi_denominator"]
        return None, starting_response, "N"
    if usable:
        row = usable[0]
        return row["kpi_name"], row["kpi_numerator"], row["kpi_denominator"]
    return None


def _kpi_axis_title(
    kpi_df: pd.DataFrame,
    kpi_name: str | None,
    response: str,
    weight: str,
) -> str:
    """Primary y-axis title: the KPI name while it still describes the response / weight, else the response."""
    if kpi_name:
        row = kpi_df[kpi_df["kpi_name"].astype(str) == str(kpi_name)]
        if not row.empty and row.iloc[0]["kpi_numerator"] == response and row.iloc[0]["kpi_denominator"] == weight:
            return str(kpi_name)
    return response


def _kpi_name_for(
    kpi_df: pd.DataFrame,
    response: str,
    weight: str,
) -> str:
    """The KPI name for a response (its denominator matching the weight if possible), else the response column."""
    if kpi_df is None or kpi_df.empty or not response:
        return response
    rows = kpi_df[kpi_df["kpi_numerator"].astype(str) == str(response)]
    exact = rows[rows["kpi_denominator"].astype(str) == str(weight)]
    chosen = exact if not exact.empty else rows
    return str(chosen.iloc[0]["kpi_name"]) if not chosen.empty else response


def _label_response_with_kpi(
    summary: pd.DataFrame,
    response: str,
    title: str,
) -> pd.DataFrame:
    """Show the response rate column (response / weight) under the KPI name in a summary table."""
    if title == response or response not in summary.columns or title in summary.columns:
        return summary
    if summary.columns[0] == response:  # the response is the grouping column itself: leave it alone
        return summary
    return summary.rename(columns={response: title})


def _label_weight_column(
    summary: pd.DataFrame,
    weight: str,
) -> pd.DataFrame:
    """Show the weight column of a summary table under the actual weight name (``N`` = row counts)."""
    if weight == "weight" or "weight" not in summary.columns or weight in summary.columns:
        return summary
    return summary.rename(columns={"weight": weight})


def _first_modelled_feature(
    columns: list[str],
    modelled: list[str] | None,
) -> str | None:
    """The most important modelled feature, the first of the Modelled group (None without a model)."""
    return next((column for column in (modelled or []) if column in columns), None)


def _chart_feature_choices(
    columns: list[str],
    modelled: list[str] | None,
    excluded=(),
) -> dict:
    """Charts feature drop-down. With a model: Modelled (most important first) and Non-modelled (the other columns,
    without the ``excluded`` response and weight)."""
    if modelled is None:
        return {column: column for column in columns}
    modelled_set = set(modelled)
    choices = {}
    choices["Modelled"] = {column: column for column in modelled if column in columns}
    choices["Non-modelled"] = {
        column: column for column in columns if column not in modelled_set and column not in excluded
    }
    return choices


def _weight_basis_select(input_id: str):
    """The 'Response and weight' select: hints in the labels, hover text on each option, a note for the selection."""
    titles = json.dumps(WEIGHT_BASIS_HELP)
    script = (
        "(function(){var t="
        + titles
        + ";function apply(){document.querySelectorAll('#"
        + input_id
        + " option').forEach(function(o){if(t[o.value]){o.title=t[o.value];}});}apply();"
        "document.addEventListener('DOMContentLoaded',apply);"
        "if(window.jQuery){window.jQuery(document).on('shiny:connected',apply);}})();"
    )
    return ui.div(
        ui.input_select(input_id, "Response and weight", WEIGHT_BASES, selected="rate_weighted"),
        ui.div(ui.output_text(f"{input_id}_note"), class_="text-muted small mb-2"),
        ui.tags.script(script),
        title="How the weight is used: hover over an option for details",
    )


def _json_safe_figure(figure):
    """Convert a Plotly figure to a JSON-safe figure for the Shiny widget."""
    import plotly.io as pio

    return pio.from_json(pio.to_json(figure, validate=True, engine="json"))


_TAB_TITLES = ("Data", "Charts", "Maps", "GBMs", "GLMs", "Scoring")


def _positive_response_modal(
    count: int,
    total: int,
    model_kind: str,
    button_id: str,
):
    """The dialog offered when a gamma model has rows with a zero, negative or missing response or weight: build on the rest."""
    return ui.modal(
        f"{count:,} of {total:,} row(s) have a zero, negative or missing response or weight, which the Gamma objective "
        f"cannot fit. Build this {model_kind} without those rows? Your data is not changed: they are only left out of "
        "this model's training, test and metrics.",
        title="Gamma needs a response above 0",
        footer=ui.div(
            ui.modal_button("Cancel"),
            ui.input_action_button(button_id, "Build on response > 0", class_="btn-primary"),
        ),
        easy_close=True,
    )


def _notify_error(
    prefix: str,
    e: Exception,
):
    """Show an error notification with the prefix and the error message."""
    ui.notification_show(f"{prefix}: {e}", type="error", duration=None)


def metrics_comparison_table(
    comparison: pd.DataFrame,
    new_label: str,
    old_label: str,
):
    """The metrics comparison table (new against old, with the change and whether it improved), as used in the
    GLM and GBM tabs and the Overlays tab."""
    headers = ["Dataset", "Metric", new_label, old_label, "Change", "Result"]
    head = ui.tags.thead(ui.tags.tr(*[ui.tags.th(label) for label in headers]))
    body_rows = []
    for row in comparison.itertuples(index=False):
        style_class = f"daisy-metric-{row.Status.lower()}"
        body_rows.append(
            ui.tags.tr(
                ui.tags.td(row.Dataset),
                ui.tags.td(row.Metric),
                ui.tags.td(report.format_metric_value(row.New, row.Metric), class_=style_class),
                ui.tags.td(report.format_metric_value(row.Old, row.Metric)),
                ui.tags.td(report.format_metric_change(row.Change, row.Metric), class_=style_class),
                ui.tags.td(row.Status, class_=style_class),
            )
        )
    return ui.tags.div(
        ui.tags.table(head, ui.tags.tbody(*body_rows), class_="daisy-metrics-table"), style="overflow-x: auto;"
    )


def make_app(
    data: pd.DataFrame,
    *,
    dataset_name: str = "data",
    kpi_spec=None,
    filter_spec=None,
    feature_spec=None,
    starting_response: str | None = None,
    starting_tab: str = "Data",
    title: str | None = None,
    show_Data: bool = True,
    show_Map: bool = True,
    show_GBM: bool = True,
    show_GLM: bool = True,
    show_Scoring: bool = True,
    sidebar_width: int = 280,
    num_threads: int = -1,
    glm_formula: str | None = None,
) -> App:
    """Build the Daisy Shiny app for a dataset.

    data: the modelling data (a DataFrame with the response, weight and features).
    dataset_name: the name shown for the data.
    kpi_spec, filter_spec, feature_spec: specification tables or files (KPIs, row filters and features).
    starting_response: the response selected when the app opens.
    starting_tab: the tab shown first; one of the visible tab titles.
    title: the app title, or None for the default.
    show_Data, show_Map, show_GBM, show_GLM, show_Scoring: show or hide each tab.
    sidebar_width: sidebar width in pixels.
    num_threads: threads used by LightGBM (-1 uses all of them).
    glm_formula: the starting GLM formula, or None for the default built from the features.

    Returns the Shiny App.
    """

    if starting_tab not in _TAB_TITLES:
        raise ValueError(f"starting_tab must be one of {', '.join(_TAB_TITLES)}; got {starting_tab!r}")
    d0 = data.reset_index(drop=True).copy()
    for c in d0.columns:
        if d0[c].dtype == object or str(d0[c].dtype) in ("str", "string"):
            d0[c] = d0[c].astype("category")
    num_cols = numerical_cols(d0)
    all_cols = list(d0.columns)
    has_tt = "train_test" in d0.columns
    has_area = "PostcodeArea" in d0.columns
    kpi_df = load_specification(d0, kpi_spec, "kpi")
    filter_df = load_specification(d0, filter_spec, "filter")
    feat_df = load_specification(d0, feature_spec, "feature")
    # the csv the feature specification came from, so "Update specification" in Charts can write back to it
    feature_spec_path = (
        Path(str(feature_spec)) if isinstance(feature_spec, (str, Path)) and Path(str(feature_spec)).is_file() else None
    )
    # a provided KPI specification sets the launch KPI, response and weight
    starting_kpi, starting_weight = None, "N"
    if kpi_spec is not None and not kpi_df.equals(specification_template(d0, "kpi")):
        defaults = _kpi_defaults(kpi_df, num_cols, starting_response)
        if defaults:
            starting_kpi, starting_response, starting_weight = defaults
    if starting_response is None or starting_response not in num_cols:
        starting_response = "price" if "price" in num_cols else (num_cols[0] if num_cols else None)

    def default_features(resp: str) -> list[str]:
        """Default GBM features for a response: the feature spec's modelling-valid features."""
        return _gbm_features_from_spec(feat_df, all_cols, resp, starting_weight)

    if glm_formula is None:
        p = DATA_DIR / "glm_formula.csv"
        glm_formula = p.read_text() if dataset_name == "insurance" and p.exists() else ""
    interaction_groups = sorted(
        {g for g in feat_df.get("interaction_grouping", []) if g and g not in ("response", "train_test")}
    )
    monotone_features = (
        feat_df.loc[feat_df["monotonicity"].fillna("").astype(str).str.strip().ne(""), "feature"].tolist()
        if "monotonicity" in feat_df.columns
        else []
    )
    monotone_choices = {
        feature: f"{feature} ({feat_df.loc[feat_df['feature'] == feature, 'monotonicity'].iloc[0]})"
        for feature in monotone_features
    }

    # ------------------------------------------------------------------ UI
    sidebar = ui.sidebar(
        ui.tags.style(
            "body { background-color: #edf4ef; } "
            ".bslib-sidebar-layout > .sidebar { background-color: #dce9df; } "
            ".navbar.navbar-expand-md { background-color: #24543f; border-bottom: 3px solid #88b99a; "
            "box-shadow: 0 2px 8px rgba(25, 55, 39, 0.16); padding-block: 0.7rem; } "
            f".navbar.navbar-expand-md .navbar-brand {{ display: block; box-sizing: border-box; width: {sidebar_width}px; "
            f"max-width: 100%; margin-right: 0; white-space: normal; text-align: center; line-height: 1.2; "
            f"color: #f4faf5; font-weight: 700; }} "
            ".navbar.navbar-expand-md .daisy-title { display: block; max-width: 100%; overflow-wrap: anywhere; } "
            ".navbar.navbar-expand-md .daisy-flower { display: block; flex: none; } "
            ".navbar.navbar-expand-md .daisy-corner { display: flex; flex-direction: column; align-items: center; "
            "gap: 0.15rem; margin-left: auto; padding-left: 0.75rem; color: #f4faf5; font-weight: 700; "
            "font-size: 0.8rem; line-height: 1.1; } "
            ".navbar.navbar-expand-md .navbar-nav { align-items: center; } "
            ".navbar.navbar-expand-md .navbar-brand { padding-block: 0; } "
            ".navbar.navbar-expand-md .nav-link { color: #e2eee5; font-weight: 600; "
            "border-radius: 5px; margin-inline: 0.15rem; transition: background-color 140ms ease, color 140ms ease; } "
            ".navbar.navbar-expand-md .nav-link:hover, .navbar.navbar-expand-md .nav-link:focus-visible { "
            "background-color: rgba(244, 250, 245, 0.14); color: #fff; } "
            ".navbar.navbar-expand-md .nav-link.active, .navbar.navbar-expand-md .nav-link[aria-current='page'] { "
            "background-color: #e3f0e6; color: #204b38; } "
            ".daisy-model-controls { display: flex; justify-content: flex-end; margin-bottom: 0.75rem; } "
            ".daisy-model-controls-card { width: fit-content; max-width: 100%; padding: 0.35rem; } "
            ".daisy-model-controls-panel { display: flex; align-items: flex-end; flex-wrap: wrap; gap: 0.5rem; } "
            ".daisy-model-controls-panel .form-group { margin-bottom: 0; } "
            ".daisy-metrics-table { border-collapse: collapse; width: 100%; } "
            ".daisy-metrics-table th, .daisy-metrics-table td { border-bottom: 1px solid #d6dfd8; "
            "padding: 0.5rem 0.65rem; text-align: right; white-space: nowrap; } "
            ".daisy-metrics-table th:first-child, .daisy-metrics-table td:first-child, "
            ".daisy-metrics-table th:nth-child(2), .daisy-metrics-table td:nth-child(2) { text-align: left; } "
            ".daisy-metric-improved { background: #e5f3e8; color: #176b36; font-weight: 600; } "
            ".daisy-metric-deteriorated { background: #fbe9e7; color: #a12c24; font-weight: 600; } "
            ".daisy-metric-unchanged { color: #55645a; } "
            ".daisy-metrics-compare-controls { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); "
            "gap: 0.75rem; margin-bottom: 0.25rem; } "
            ".daisy-metrics-compare-controls .form-group { margin-bottom: 0; } "
            "#gbm_metrics_comparison { margin: 0 !important; padding: 0 !important; min-height: 0 !important; } "
            # Model Overview: tables start at the left edge, and the two lower tables sit next to each other
            ".daisy-overview shiny-data-frame .shiny-data-grid { margin-left: 0 !important; margin-right: 0 !important; } "
            ".daisy-table-row { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 0.25rem 3rem; } "
            ".daisy-table-row > div { flex: 0 1 auto; max-width: 100%; } "
            ".daisy-spec-editor { max-width: 1200px; } "
            ".daisy-spec-editor textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
            "font-size: 0.875rem; } "
            ".daisy-spec-actions { display: flex; align-items: center; gap: 1rem; margin-top: 1rem; }"
            # file inputs: the upload progress (and its 'Upload complete' text) shows in full
            ".shiny-input-container .progress { height: auto; min-height: 1.5rem; overflow: visible; } "
            ".shiny-input-container .progress-bar { white-space: normal; overflow: visible; line-height: 1.5rem; "
            "min-height: 1.5rem; } "
            ".shiny-file-input-progress { width: 100%; } "
        ),
        ui.input_select("kpi", "KPI", choices=list(kpi_df["kpi_name"]), selected=starting_kpi),
        ui.input_select("response", "Response", choices=num_cols, selected=starting_response),
        ui.input_select("weight", "Weight", choices=["N"] + num_cols, selected=starting_weight),
        ui.hr(),
        ui.input_select("filter_choice", "Filter", choices=filter_select_choices(filter_df)),
        ui.input_text("filter_custom", "Custom filter", placeholder="age<25 & licence=='Full'"),
        *(
            [
                ui.input_radio_buttons(
                    "train_test", "Rows", {"All": "All", "Train": "Train", "Test": "Test"}, inline=True
                )
            ]
            if has_tt
            else []
        ),
        ui.output_ui("row_count"),
        ui.output_ui("memory_usage"),  # the bottom of the sidebar
        # ui.output_text("filter_msg"),
        width=sidebar_width,
        title="Options",
    )

    x_default = "age" if "age" in all_cols else all_cols[0]
    chart_ui = ui.nav_panel(
        "Charts",
        ui.navset_tab(
            ui.nav_panel(
                "1-way line and bar",
                ui.output_ui("chart_model_selector"),
                ui.output_ui("x_monotone_label"),  # the feature's monotonicity, just above Banding
                # Enter in the Banding / Min / Max / Bins boxes moves focus out, which applies the value
                ui.tags.script(
                    "(function(){var ids={x_banding:1,x_min:1,x_max:1,x_bins:1};"
                    "document.addEventListener('keydown',function(e){if(e.key!=='Enter'||!ids[e.target.id]){return;}"
                    "e.target.blur();e.target.focus();});})();"
                ),
                ui.layout_columns(
                    ui.div(
                        ui.input_select("x_feature", "Feature", choices=all_cols, selected=x_default),
                        # Prev / Next step through the features in the order of the drop-down
                        ui.div(
                            ui.tags.button(
                                "\u2039 Prev",
                                type="button",
                                class_="btn btn-default btn-sm text-nowrap px-2",
                                onclick="daisyStepFeature(-1)",
                                title="Previous feature",
                            ),
                            ui.tags.button(
                                "Next \u203a",
                                type="button",
                                class_="btn btn-default btn-sm text-nowrap px-2",
                                onclick="daisyStepFeature(1)",
                                title="Next feature",
                            ),
                            class_="d-flex gap-2",
                            style="margin-top: -0.5rem;",
                        ),
                        ui.tags.script(
                            "function daisyStepFeature(step){var s=document.getElementById('x_feature');"
                            "if(!s){return;}var i=s.selectedIndex+step;if(i<0||i>=s.options.length){return;}"
                            "s.selectedIndex=i;if(window.jQuery){window.jQuery(s).trigger('change');}"
                            "else{s.dispatchEvent(new Event('change',{bubbles:true}));}}"
                        ),
                    ),
                    ui.input_text("x_banding", "Banding", value="1", update_on="blur"),
                    ui.tooltip(
                        ui.input_text("x_min", "Min", value="", placeholder="none", update_on="blur"),
                        "Values below this join the first band. Blank = no minimum.",
                    ),
                    ui.tooltip(
                        ui.input_text("x_max", "Max", value="", placeholder="none", update_on="blur"),
                        "Values above this join the last band. Blank = no maximum.",
                    ),
                    ui.tooltip(
                        ui.input_text("x_bins", "Bins", value="", placeholder="none", update_on="blur"),
                        "Split the feature into this many equal bins, e.g. 4 bins from 0 to 20: [0,5), [5,10), "
                        "[10,15), [15,Inf). Only available when Banding is blank or 0.",
                    ),
                    ui.tooltip(
                        ui.input_select(
                            "x_monotone",
                            "Monotonicity",
                            {"": "None", "increasing": "Increasing", "decreasing": "Decreasing"},
                        ),
                        "Monotonicity constraint for a GBM built from now on. Saved to the feature "
                        "specification with Update specification.",
                    ),
                    ui.input_select("low_exposure", "Group low exposure", {"0": "None", "1%": "1%", "0.1%": "0.1%"}),
                    ui.input_selectize("add_cols", "Add columns", choices=[], multiple=True),
                    ui.div(
                        ui.input_checkbox("show_bars", "Weight bars", True),
                        ui.input_checkbox("sigma_bars", "Sigma bars", False),
                    ),
                    col_widths=[2, 1, 1, 1, 1, 1, 2, 2, 1],
                ),
                ui.div(
                    ui.tooltip(
                        ui.input_numeric("two_way_n", "Two-way top N", TWO_WAY_FEATURES, min=2, max=10, width="150px"),
                        "The report's two-way SHAP and ALE views cover every pair of this many top features.",
                    ),
                    ui.div(  # far right, in line with Two-way top N: Update specification, then Export charts
                        ui.tooltip(
                            ui.input_action_button("x_apply_spec", "Update specification"),
                            "Save this feature's banding, minimum, maximum, bins and monotonicity to the feature "
                            "specification (and its csv file), and use them for the models built from now on.",
                        ),
                        ui.download_button("line_bar_export", "Export charts (.html)"),
                        class_="ms-auto d-flex flex-wrap align-items-end gap-2",
                    ),
                    class_="d-flex flex-wrap align-items-end gap-2 mb-3",
                ),
                ui.output_ui("x_spec_note"),
                ui.tags.script(
                    "(function(){function on(){if(!window.Shiny||!Shiny.addCustomMessageHandler){return false;}"
                    "Shiny.addCustomMessageHandler('daisy_disable',function(m){var e=document.getElementById(m.id);"
                    "if(e){e.disabled=!!m.disabled;}});return true;}"
                    "if(!on()){document.addEventListener('DOMContentLoaded',on);}})();"
                ),
                ui.card(output_widget("line_bar_plot", height="460px"), full_screen=True),
                ui.card(ui.output_data_frame("line_bar_table")),
            ),
            ui.nav_panel(
                "SHAP",
                ui.layout_columns(
                    ui.div(ui.input_select("shap_f1", "Feature 1", choices=[]), _feature_step("shap_f1")),
                    ui.input_text("shap_b1", "Banding 1", value="1"),
                    ui.input_select("shap_f2", "Feature 2", choices=["none"]),
                    ui.input_text("shap_b2", "Banding 2", value="1"),
                    ui.input_slider("shap_q", "Ribbon quantile", 0.01, 0.25, 0.05, step=0.01),
                    col_widths=[3, 2, 3, 2, 2],
                ),
                ui.card(output_widget("shap_plot", height="460px"), full_screen=True),
                ui.card(
                    ui.card_header("What drives predictions within each band (stacked mean SHAP)"),
                    output_widget("shap_stacked", height="380px"),
                ),
            ),
            ui.nav_panel(
                "ALE",
                ui.output_ui("ale_model_selector"),
                ui.layout_columns(
                    ui.div(ui.input_select("ale_feature", "Feature 1", choices=[]), _feature_step("ale_feature")),
                    ui.input_select("ale_feature2", "Feature 2", choices={"": "None"}),
                    ui.input_numeric("ale_bins", "Bins", 20, min=2, max=50),
                    ui.input_numeric("ale_sample_size", "Maximum rows", 10000, min=2, max=100000),
                    col_widths=[4, 4, 2, 2],
                ),
                ui.output_text("ale_status"),
                ui.card(output_widget("ale_plot", height="460px"), full_screen=True),
                ui.card(ui.output_data_frame("ale_table")),
            ),
        ),
    )

    map_ui = ui.nav_panel(
        "Maps",
        ui.input_select(
            "map_metric",
            "Colour by",
            {
                "rate": "response / weight",
                "weight": "weight",
                "count": "rows",
                "predicted": "predicted rate (model)",
                "ae": "A/E (model)",
            },
        ),
        ui.input_select("map_model", "Model for predictions", choices=[]),
        ui.card(output_widget("map_plot", height="680px"), full_screen=True),
        ui.output_text("map_note"),
    )

    data_ui = ui.nav_panel(
        "Data",
        ui.navset_tab(
            ui.nav_panel("Dataset", ui.output_data_frame("data_view")),
            ui.nav_panel("Column summary", ui.output_data_frame("col_summary")),
        ),
    )

    gbm_ui = ui.nav_panel(
        "GBMs",
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    ui.input_checkbox_group(
                        "gbm_features",
                        "Features",
                        choices={
                            c: c
                            for c in _gbm_features_from_spec(
                                feat_df, all_cols, starting_response or "", starting_weight
                            )
                        },
                        selected=default_features(starting_response or ""),
                    ),
                    style="max-height: 300px; overflow-y: auto;",
                ),
                ui.tooltip(
                    ui.input_select("gbm_objective", "Objective", list(LGBM_OBJECTIVES), selected="gamma"),
                    parameter_help("objective"),
                ),
                ui.tooltip(
                    ui.input_numeric("gbm_tweedie_p", "Tweedie variance power", 1.5, min=1.01, max=1.99, step=0.05),
                    parameter_help("tweedie_variance_power"),
                ),
                ui.tooltip(ui.input_numeric("gbm_rounds", "Max trees", 1000, min=1), parameter_help("num_iterations")),
                ui.tooltip(
                    ui.input_numeric("gbm_early", "Early stopping rounds", 50, min=0),
                    parameter_help("early_stopping_round"),
                ),
                ui.tooltip(
                    ui.input_numeric("gbm_lr", "Learning rate", 0.05, min=0.001, step=0.01),
                    parameter_help("learning_rate"),
                ),
                ui.tooltip(ui.input_numeric("gbm_leaves", "Num leaves", 5, min=2), parameter_help("num_leaves")),
                ui.tooltip(ui.input_numeric("gbm_depth", "Max depth (-1 = none)", -1), parameter_help("max_depth")),
                ui.tooltip(
                    ui.input_numeric("gbm_min_data", "Min data in leaf", 100, min=1), parameter_help("min_data_in_leaf")
                ),
                ui.tooltip(
                    ui.input_numeric("gbm_row_sample", "Row sample rate", 0.75, min=0.1, max=1, step=0.05),
                    parameter_help("bagging_fraction"),
                ),
                ui.tooltip(
                    ui.input_numeric("gbm_col_sample", "Column sample rate", 1.0, min=0.1, max=1, step=0.05),
                    parameter_help("feature_fraction"),
                ),
                ui.tooltip(ui.input_numeric("gbm_l1", "lambda_l1", 0.0, min=0), parameter_help("lambda_l1")),
                ui.tooltip(ui.input_numeric("gbm_l2", "lambda_l2", 0.0, min=0), parameter_help("lambda_l2")),
                ui.tooltip(
                    ui.input_select(
                        "gbm_offset", "Offset (expected total column)", ["no offset"] + num_cols
                    ),
                    parameter_help("offset (init_score)"),
                ),
                _weight_basis_select("gbm_weight_basis"),
                ui.div(
                    ui.input_checkbox_group(
                        "gbm_monotone", "Monotonicity constraints", choices=monotone_choices, selected=monotone_features
                    ),
                    ui.output_ui("gbm_monotone_note"),
                    style="max-height: 300px; overflow-y: auto;",
                ),
                ui.tooltip(
                    ui.input_selectize(
                        "gbm_groups",
                        "Interaction groups (blank = all may interact)",
                        choices=interaction_groups,
                        multiple=True,
                    ),
                    parameter_help("interaction_groups"),
                ),
                ui.tooltip(
                    ui.input_select("gbm_shap", "SHAP values", {"10k": "10k sample", "All": "All rows", "No": "None"}),
                    parameter_help("shap_sample"),
                ),
                width=300,
            ),
            ui.div(
                ui.card(
                    ui.div(
                        ui.input_select("gbm_model", "Model", choices=[]),
                        ui.download_button("gbm_download", "Ratebook (.xlsx)"),
                        ui.download_button("gbm_bundle_download", "Model bundle (.zip)"),
                        class_="daisy-model-controls-panel",
                    ),
                    class_="daisy-model-controls-card",
                ),
                class_="daisy-model-controls",
            ),
            ui.navset_card_tab(
                ui.nav_panel("Training", output_widget("gbm_eval_plot", height="380px"), ui.output_text("gbm_stats")),
                ui.nav_panel(
                    "Model Overview",
                    ui.div(
                        ui.output_data_frame("gbm_overview_table"),
                        ui.p(
                            "Total Weight is the sum of the weight column (row counts when the weight is N); Total Response is the sum of the response. "
                            "The weight is the one the model was built with. "
                            "Avg Actual - Weighted and Avg Predicted - Weighted are averages weighted by it. "
                            "The minimum, quartiles and maximum, and the mean, describe the unweighted row-level "
                            "predictions. Uses the current row filter (not the Rows selection).",
                            class_="text-muted small mt-2",
                        ),
                        ui.div(
                            ui.div(ui.h6("GBM parameters", class_="mt-3"), ui.output_ui("gbm_params_table")),
                            ui.div(
                                ui.h6("Monotonicity applied", class_="mt-3"), ui.output_data_frame("gbm_mono_table")
                            ),
                            class_="daisy-table-row",
                        ),
                        class_="daisy-overview",
                    ),
                ),
                ui.nav_panel(
                    "Feature importance",
                    output_widget("gbm_imp_plot", height="auto", fill=False),
                    ui.output_data_frame("gbm_imp_table"),
                ),
                ui.nav_panel(
                    "Lift Chart",
                    ui.input_select(
                        "gbm_lift_bins",
                        "Bins",
                        {"10": "10 (presentation)", "20": "20", "50": "50", "100": "100 (diagnostic)"},
                        selected=str(charts.LIFT_DEFAULT_BINS),
                        width="200px",
                    ),
                    output_widget("gbm_lift_plot", height="380px", fill=False),
                    ui.layout_columns(
                        output_widget("gbm_lorenz_plot", height="340px", fill=False),
                        ui.output_data_frame("gbm_lift_table"),
                        col_widths=[5, 7],
                        row_heights="auto",
                        fill=False,
                        fillable=False,
                    ),
                ),
                ui.nav_panel(
                    "Metrics",
                    ui.output_data_frame("gbm_metrics_table"),
                    ui.div(
                        ui.input_select(
                            "gbm_compare_source",
                            "Compare with",
                            {"gbm": "Old GBM model", "glm": "Old GLM model", "column": "Prediction column"},
                            selected="gbm",
                        ),
                        ui.output_ui("gbm_compare_target_ui"),
                        ui.input_select(
                            "gbm_compare_split", "Dataset", {"Train": "Train", "Test": "Test"}, selected="Test"
                        ),
                        class_="daisy-metrics-compare-controls",
                    ),
                    ui.output_ui("gbm_metrics_comparison"),
                ),
                ui.nav_panel(
                    "SHAP Relativities",
                    ui.output_text("gbm_shap_note"),
                    ui.layout_columns(
                        ui.input_select("gbm_shap_feature", "Feature", choices=[]),
                        ui.input_numeric("gbm_shap_banding", "Banding", 0, min=0),
                        col_widths=[6, 6],
                        row_heights="auto",
                        fill=False,
                        fillable=False,
                    ),
                    output_widget("gbm_shap_rel_plot", height="380px"),
                    ui.output_data_frame("gbm_shap_rel_table"),
                ),
                ui.nav_panel(
                    "Ratebook",
                    ui.input_select("gbm_rb_feature", "Feature", choices=[]),
                    output_widget("gbm_rb_plot", height="380px"),
                    ui.output_data_frame("gbm_rb_table"),
                ),
                ui.nav_spacer(),
                ui.nav_control(ui.input_action_button("gbm_build", "Build GBM", class_="btn-primary")),
            ),
        ),
    )

    glm_ui = ui.nav_panel(
        "GLMs",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("glm_objective", "Objective", list(GLM_OBJECTIVES), selected="gamma"),
                ui.input_numeric("glm_var_power", "Tweedie var.power", 1.5, min=1.01, max=1.99, step=0.05),
                _weight_basis_select("glm_weight_basis"),
                ui.input_radio_buttons(
                    "glm_data",
                    "Fit on",
                    {"All": "All rows", "Training": "Training rows"},
                    inline=True,
                    selected="Training" if has_tt else "All",
                ),
                width=280,
            ),
            ui.div(
                ui.card(
                    ui.div(
                        ui.input_select("glm_model", "Model", choices=[]),
                        ui.download_button("glm_download", "Ratebook (.xlsx)"),
                        ui.download_button("glm_predictions_download", "Predictions (.csv)"),
                        class_="daisy-model-controls-panel",
                    ),
                    class_="daisy-model-controls-card",
                ),
                class_="daisy-model-controls",
            ),
            ui.navset_card_tab(
                ui.nav_panel("Builder", builder_ui()),
                ui.nav_panel(
                    "Formula",
                    ui.div(
                        ui.tags.style(
                            ".glm-formula-tools { flex-wrap: nowrap !important; }"
                            ".glm-formula-tools .shiny-input-container { margin-bottom: 0; flex: 1 1 0; min-width: 0; }"
                            ".glm-formula-tools .btn { white-space: nowrap; }"
                        ),
                        ui.input_file(
                            "glm_formula_file",
                            "Import (.txt)",
                            accept=[".txt"],
                            multiple=False,
                        ),
                        ui.download_button("glm_formula_export", "Export (.txt)", class_="mb-1"),
                        class_="glm-formula-tools d-flex align-items-center gap-3 mb-2",
                    ),
                    ui.input_text_area(
                        "glm_formula",
                        "GLM formula terms (Patsy/Python RHS: C(make), age * C(make), I(age ** 2), bs(age, df=4) ...)",
                        value=glm_formula,
                        rows=20,
                        width="100%",
                    ),
                ),
                ui.nav_panel("Coefficients", ui.output_text("glm_stats"), ui.output_data_frame("glm_coefs")),
                ui.nav_panel(
                    "Feature importance",
                    ui.input_radio_buttons(
                        "glm_imp_view",
                        "Importance by",
                        {"term": "Term", "feature": "Feature"},
                        selected="term",
                        inline=True,
                    ),
                    ui.output_data_frame("glm_imp_table"),
                ),
                ui.nav_panel(
                    "Lift Chart",
                    ui.input_select(
                        "glm_lift_bins",
                        "Bins",
                        {"10": "10 (presentation)", "20": "20", "50": "50", "100": "100 (diagnostic)"},
                        selected=str(charts.LIFT_DEFAULT_BINS),
                        width="200px",
                    ),
                    output_widget("glm_lift_plot", height="380px", fill=False),
                    ui.layout_columns(
                        output_widget("glm_lorenz_plot", height="340px", fill=False),
                        ui.output_data_frame("glm_lift_table"),
                        col_widths=[5, 7],
                        row_heights="auto",
                        fill=False,
                        fillable=False,
                    ),
                ),
                ui.nav_panel(
                    "Metrics",
                    ui.output_data_frame("glm_metrics_table"),
                    ui.div(
                        ui.input_select(
                            "glm_compare_source",
                            "Compare with",
                            {"glm": "Old GLM model", "gbm": "Old GBM model", "column": "Prediction column"},
                            selected="glm",
                        ),
                        ui.output_ui("glm_compare_target_ui"),
                        ui.input_select(
                            "glm_compare_split",
                            "Dataset",
                            {"Train": "Train", "Test": "Test"} if has_tt else {"All": "All"},
                            selected="Test" if has_tt else "All",
                        ),
                        class_="daisy-metrics-compare-controls",
                    ),
                    ui.output_ui("glm_metrics_comparison"),
                ),
                ui.nav_panel(
                    "Ratebook",
                    ui.input_select("glm_rb_feature", "Feature", choices=[]),
                    output_widget("glm_rb_plot", height="380px"),
                    ui.output_data_frame("glm_rb_table"),
                ),
                ui.nav_spacer(),
                ui.nav_control(ui.input_action_button("glm_build", "Build GLM", class_="btn-primary")),
            ),
        ),
    )

    overlay_ui = ui.nav_panel(
        "Overlays",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h6("1. Data"),
                ui.input_text(
                    "overlay_path", "Path to cleaned CSV (In-sample data)", placeholder="/path/to/insample_data.csv"
                ),
                ui.div(
                    ui.input_action_button("overlay_browse", "Browse…"),
                    ui.input_action_button("overlay_load", "Load data", class_="btn-primary"),
                    class_="d-flex gap-2 mb-2",
                ),
                ui.output_text("overlay_data_status"),
                ui.input_text(
                    "overlay_test_path",
                    "Path to cleaned CSV (Out-of-sample test data)",
                    placeholder="/path/to/oos_test_data.csv",
                ),
                ui.div(
                    ui.input_action_button("overlay_test_browse", "Browse…"),
                    ui.input_action_button("overlay_test_load", "Load test data", class_="btn-primary"),
                    class_="d-flex gap-2 mb-2",
                ),
                ui.output_text("overlay_test_status"),
                ui.hr(),
                ui.h6("2. Read Model"),
                ui.input_file(
                    "overlay_model_file",
                    "GBM model.pkl (or Model bundle .zip) or GLM ratebook (.xlsx)",
                    accept=[".pkl", ".pickle", ".zip", ".xlsx"],
                ),
                ui.p(
                    "Files read in here appear in the Base model list. Only open model files you trust: loading a "
                    "pickle can run code.",
                    class_="text-muted small",
                ),
                ui.hr(),
                ui.h6("3. Model & Bootstrap Samples"),
                ui.input_select("overlay_model", "Base model", choices=[]),
                ui.input_numeric("overlay_samples", "Bootstrap samples", 500, min=100, max=2000),
                ui.hr(),
                ui.h6("4. Parameters"),
                ui.input_select(
                    "overlay_ratebook_objective",
                    "Ratebook objective (for PoDE)",
                    {
                        "poisson": "poisson",
                        "gamma": "gamma",
                        "tweedie": "tweedie",
                        "gaussian": "gaussian",
                        "binomial": "binomial",
                    },
                    selected="poisson",
                ),
                ui.input_numeric("overlay_ratebook_power", "Tweedie var.power", 1.5, min=1.01, max=1.99, step=0.05),
                ui.p("Used for a ratebook base model, which does not record its objective.", class_="text-muted small"),
                width=340,
            ),
            ui.layout_columns(
                ui.div(
                    ui.input_select("overlay_feature_plot", "Feature (shown in the plot and grid)", choices=[]),
                    _feature_step("overlay_feature_plot"),
                ),
                ui.div(
                    ui.input_action_button(
                        "overlay_calc", "Calculate suggestions for all features", class_="btn-primary"
                    ),
                    class_="d-flex justify-content-end align-items-end",
                ),
                col_widths=[5, 7],
                class_="mb-2",
            ),
            ui.output_ui("overlay_note"),
            ui.card(output_widget("overlay_range_plot", height="420px"), full_screen=True),
            ui.card(
                ui.card_header("Bands: suggested relativities (edit the Relativity column)"),
                ui.output_data_frame("overlay_grid"),
            ),
            ui.card(output_widget("overlay_adjusted_plot", height="380px"), full_screen=True),
            ui.layout_columns(
                ui.input_text("overlay_name", "Overlay name", placeholder="e.g. age overlay"),
                ui.input_action_button("overlay_check", "Check balance"),
                ui.input_action_button("overlay_save", "Save overlay", class_="btn-primary"),
                col_widths=[5, 2, 2],
            ),
            ui.output_ui("overlay_check_note"),
            ui.card(
                ui.card_header("Saved overlays"),
                ui.div(ui.download_button("overlay_export", "Export all adjustments (.xlsx)"), class_="mb-2"),
                ui.output_data_frame("overlay_list"),
            ),
            ui.layout_columns(
                ui.input_select("overlay_remove", "Remove an overlay", choices=[]),
                ui.div(ui.input_action_button("overlay_delete", "Remove", class_="btn-outline-danger")),
                col_widths=[4, 2],
            ),
            ui.hr(),
            ui.layout_columns(
                ui.div(
                    ui.input_select("overlay_eval", ui.tags.b("Overlays Plots on Test Data"), choices=[]),
                    _feature_step("overlay_eval"),
                ),
                ui.div(
                    ui.input_checkbox("overlay_errbars", "Error bars", True),
                    ui.download_button("overlay_report_download", "Export report (.html)", class_="text-nowrap"),
                    class_="d-flex gap-3 justify-content-end align-items-end",
                ),
                col_widths=[5, 7],
            ),
            ui.output_text("overlay_eval_note"),
            ui.card(output_widget("overlay_eval_plot", height="420px"), full_screen=True),
            ui.card(
                ui.card_header("Metrics on the test rows: base model and with the overlay"),
                ui.output_text("overlay_metric_note"),
                ui.output_ui("overlay_metric_table"),
            ),
        ),
    )

    scoring_ui = ui.nav_panel(
        "Scoring",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h6("1. Data"),
                ui.input_text(
                    "score_path", "Path to cleaned CSV (unseen data)", placeholder="/path/to/unseen_data.csv"
                ),
                ui.div(
                    ui.input_action_button("score_browse", "Browse…"),
                    ui.input_action_button("score_load", "Load data", class_="btn-primary"),
                    class_="d-flex gap-2 mb-2",
                ),
                ui.output_text("score_data_status"),
                ui.hr(),
                ui.h6("2. Model"),
                ui.input_radio_buttons(
                    "score_source",
                    "Score with",
                    {
                        "app": "Model selected in the app",
                        "gbm_pkl": "GBM: read in model.pkl",
                        "glm_rb": "GLM: read in ratebook (.xlsx)",
                    },
                    selected="gbm_pkl",
                ),
                ui.panel_conditional("input.score_source == 'app'", ui.output_ui("score_model_select")),
                ui.panel_conditional(
                    "input.score_source == 'gbm_pkl'",
                    ui.input_file(
                        "score_gbm_file", "GBM model.pkl (or Model bundle .zip)", accept=[".pkl", ".pickle", ".zip"]
                    ),
                    ui.p("Only open model files you trust: loading a pickle can run code.", class_="text-muted small"),
                ),
                ui.panel_conditional(
                    "input.score_source == 'glm_rb'",
                    ui.input_file("score_rb_file", "GLM ratebook (.xlsx)", accept=[".xlsx"]),
                    ui.p(
                        "Predictions are the base prediction times the relativity of each feature level.",
                        class_="text-muted small",
                    ),
                ),
                ui.output_ui("score_model_status"),
                ui.hr(),
                ui.h6("3. KPI & Model Objective"),
                ui.output_ui("score_columns_ui"),
                ui.output_ui("score_objective_ui"),
                ui.hr(),
                width=340,
            ),
            ui.navset_tab(
                ui.nav_panel("Column summary", ui.output_data_frame("score_col_summary")),
                ui.nav_panel("Predictions", ui.output_ui("score_notes"), ui.output_data_frame("score_preview")),
                ui.nav_panel(
                    "Metrics", ui.output_ui("score_metrics_note"), ui.output_data_frame("score_metrics_table")
                ),
                ui.nav_panel(
                    "1-way line and bar",
                    ui.layout_columns(
                        ui.div(ui.input_select("score_feature", "Feature", choices=[]), _feature_step("score_feature")),
                        ui.input_text("score_banding", "Banding", value="1"),
                        ui.input_text("score_min", "Min", value="", placeholder="none"),
                        ui.input_text("score_max", "Max", value="", placeholder="none"),
                        ui.input_text("score_bins", "Bins", value="", placeholder="none"),
                        ui.input_select(
                            "score_low_exposure", "Group low exposure", {"0": "None", "1%": "1%", "0.1%": "0.1%"}
                        ),
                        ui.div(
                            ui.input_checkbox("score_show_bars", "Weight bars", True),
                            ui.input_checkbox("score_sigma_bars", "Sigma bars", False),
                        ),
                        col_widths=[3, 1, 1, 1, 1, 2, 2],
                    ),
                    ui.output_ui("score_chart_note"),
                    ui.card(output_widget("score_line_bar_plot", height="460px"), full_screen=True),
                    ui.card(ui.output_data_frame("score_chart_table")),
                ),
                ui.nav_spacer(),
                # top right, in line with the tabs: Score data and the predictions download
                ui.nav_control(
                    ui.div(
                        ui.input_action_button("score_run", "Score data", class_="btn-primary text-nowrap"),
                        ui.download_button("score_download", "Predictions (.csv)", class_="text-nowrap"),
                        ui.download_button("score_export", "Export scoring report (.html)", class_="text-nowrap"),
                        class_="d-flex gap-2 align-items-center flex-shrink-0",
                    )
                ),
            ),
        ),
    )

    spec_editor = ui.nav_panel(
        "Specifications",
        ui.div(
            ui.h3("Specifications"),
            ui.navset_tab(
                ui.nav_panel(
                    "Features",
                    ui.input_radio_buttons(
                        "spec_feature_mode", "Editor", {"table": "Table", "csv": "CSV"}, selected="table", inline=True
                    ),
                    ui.output_ui("spec_feature_editor"),
                ),
                ui.nav_panel(
                    "KPIs",
                    ui.input_radio_buttons(
                        "spec_kpi_mode", "Editor", {"table": "Table", "csv": "CSV"}, selected="table", inline=True
                    ),
                    ui.output_ui("spec_kpi_editor"),
                ),
                ui.nav_panel(
                    "Filters",
                    ui.input_text_area(
                        "spec_filter_text",
                        "One filter per line; use lines starting with --- for section headings",
                        value="\n".join(filter_df["filter"].astype(str)),
                        rows=18,
                        width="100%",
                    ),
                ),
            ),
            ui.div(
                ui.input_action_button("spec_apply", "Apply specifications", class_="btn-primary"),
                ui.download_button("spec_download", "Download specifications (.zip)"),
                ui.output_text("spec_editor_status"),
                class_="daisy-spec-actions",
            ),
            class_="daisy-spec-editor",
        ),
    )

    panels = []
    if show_Data:
        panels.append(data_ui)
    panels.append(chart_ui)
    if show_Map and has_area:
        panels.append(map_ui)
    if show_GBM and has_tt:
        panels.append(gbm_ui)
    if show_GLM:
        panels.append(glm_ui)
        panels.append(overlay_ui)
    if show_Scoring:
        panels.append(scoring_ui)
    panels.append(spec_editor)
    brand = ui.span(ui.tags.span(title or "Daisy", class_="daisy-title"), class_="daisy-brand")
    panels.append(ui.nav_spacer())
    panels.append(
        ui.nav_control(
            ui.div(
                ui.tags.img(src="/images/daisy_flower.svg", alt="", width="44", height="44", class_="daisy-flower"),
                ui.span("Daisy", class_="daisy-corner-name"),
                class_="daisy-corner",
            )
        )
    )
    app_ui = ui.page_navbar(
        *panels, title=brand, id="tabs", selected=starting_tab, sidebar=sidebar, window_title="Daisy"
    )

    # ------------------------------------------------------------------ server
    def server(
        input,
        output,
        session,
    ):
        """Shiny server function: holds the app state and wires the outputs, reactive calculations and downloads of every tab."""
        nonlocal kpi_df, filter_df, feat_df, interaction_groups, monotone_features, monotone_choices
        base = reactive.value(d0)
        model_cols = reactive.value(pd.DataFrame(index=d0.index))
        glm_models = reactive.value([])
        gbm_models = reactive.value([])
        chart_default_model = reactive.value(None)
        overlays = reactive.value([])  # saved overlays: dict(overlay, family, model)
        overlay_candidate = reactive.value(None)  # the suggestion of the feature shown: dict(...)
        overlay_candidates = reactive.value({})  # the suggestions of every feature: feature -> dict(...)
        overlay_check_state = reactive.value(None)  # the last balance check: dict(...)
        overlay_uploads = reactive.value([])  # base models read in on the Overlays tab: dict(family, model, name)
        overlay_data = reactive.value(None)  # unseen data loaded on the Overlays tab (None: use the training rows)
        overlay_test_data = reactive.value(
            None
        )  # out-of-sample test data loaded on the Overlays tab (None: use the test rows)
        chart_features_model = reactive.value(None)  # the model the Charts feature drop-down was last set up for
        prediction_per_unit = reactive.value(frozenset())  # prediction columns that are per unit of weight, not totals
        spec_version = reactive.value(0)
        spec_status = reactive.value("Edits apply to this app session.")

        @render.ui
        def spec_feature_editor():
            """Feature specification editor: CSV text or an editable grid, depending on the chosen mode."""
            if input.spec_feature_mode() == "csv":
                return ui.input_text_area(
                    "spec_feature_text",
                    "Feature specification (CSV)",
                    value=feat_df.to_csv(index=False),
                    rows=18,
                    width="100%",
                )
            return ui.output_data_frame("spec_feature_grid")

        @render.data_frame
        def spec_feature_grid():
            """Editable grid of the feature specification, refreshed when the specifications are applied."""
            spec_version()
            return render.DataGrid(feat_df, editable=True, height="500px")

        @render.ui
        def spec_kpi_editor():
            """KPI specification editor: CSV text or an editable grid, depending on the chosen mode."""
            if input.spec_kpi_mode() == "csv":
                return ui.input_text_area(
                    "spec_kpi_text", "KPI specification (CSV)", value=kpi_df.to_csv(index=False), rows=12, width="100%"
                )
            return ui.output_data_frame("spec_kpi_grid")

        @render.data_frame
        def spec_kpi_grid():
            """Editable grid of the KPI specification, refreshed when the specifications are applied."""
            spec_version()
            return render.DataGrid(kpi_df, editable=True, height="360px")

        @reactive.effect
        @reactive.event(input.spec_apply)
        def _apply_specifications():
            """Read, check and apply the edited feature, KPI and filter specifications to this session. Errors are shown as the status."""
            nonlocal kpi_df, filter_df, feat_df, interaction_groups, monotone_features, monotone_choices
            try:
                if input.spec_feature_mode() == "csv":
                    new_features = _parse_editor_spec(input.spec_feature_text(), "feature", all_cols)
                else:
                    new_features = _read_editor_grid(spec_feature_grid, feat_df, "feature", all_cols)
                if input.spec_kpi_mode() == "csv":
                    new_kpis = _parse_editor_spec(input.spec_kpi_text(), "kpi", all_cols)
                else:
                    new_kpis = _read_editor_grid(spec_kpi_grid, kpi_df, "kpi", all_cols)
                new_filters = _parse_editor_spec(input.spec_filter_text(), "filter", all_cols)
            except Exception as error:
                reason = str(error) or type(error).__name__
                spec_status.set(f"Specifications not applied: {reason}")
                return

            kpi_df, filter_df, feat_df = new_kpis, new_filters, new_features
            interaction_groups = sorted(
                {
                    str(group).strip()
                    for group in feat_df.get("interaction_grouping", [])
                    if str(group).strip() and str(group).strip() not in ("response", "train_test")
                }
            )
            monotone_features = (
                feat_df.loc[feat_df["monotonicity"].fillna("").astype(str).str.strip().ne(""), "feature"].tolist()
                if "monotonicity" in feat_df.columns
                else []
            )
            monotone_choices = {
                feature: f"{feature} ({feat_df.loc[feat_df['feature'] == feature, 'monotonicity'].iloc[0]})"
                for feature in monotone_features
            }

            with reactive.isolate():
                selected_kpi = input.kpi()
                selected_filter = input.filter_choice()
            kpi_names = list(kpi_df["kpi_name"].astype(str))
            ui.update_select(
                "kpi",
                choices=kpi_names,
                selected=selected_kpi if selected_kpi in kpi_names else (kpi_names[0] if kpi_names else None),
            )
            available_filters = {value for values in filter_choices(filter_df).values() for value in values}
            ui.update_select(
                "filter_choice",
                choices=filter_select_choices(filter_df),
                selected=selected_filter if selected_filter in available_filters else "no filter",
            )
            if show_GBM and has_tt:
                with reactive.isolate():
                    selected_features = list(input.gbm_features() or [])
                    selected_monotone = list(input.gbm_monotone() or [])
                    selected_groups = list(input.gbm_groups() or [])
                feature_choices = _gbm_features_from_spec(feat_df, all_cols, response(), weight())
                ui.update_checkbox_group(
                    "gbm_features",
                    choices={feature: feature for feature in feature_choices},
                    selected=[feature for feature in selected_features if feature in feature_choices],
                )
                ui.update_checkbox_group(
                    "gbm_monotone",
                    choices=monotone_choices,
                    selected=[feature for feature in selected_monotone if feature in monotone_choices],
                )
                ui.update_selectize(
                    "gbm_groups",
                    choices=interaction_groups,
                    selected=[group for group in selected_groups if group in interaction_groups],
                )
            spec_version.set(spec_version() + 1)
            spec_status.set("Specifications applied for this session.")

        @render.ui
        def gbm_monotone_note():
            """Note shown when no feature has a monotonicity constraint for the GBM."""
            spec_version()
            if monotone_choices:
                return None
            return ui.p("No feature has a monotonicity in the feature specification.", class_="text-muted small")

        @render.text
        def spec_editor_status():
            """Status line of the specification editor."""
            return spec_status()

        @render.download_button(filename="daisy_specifications.zip")
        def spec_download():
            """Zip of the current feature, KPI and filter specifications."""
            archive_data = io.BytesIO()
            with zipfile.ZipFile(archive_data, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("feature_spec.csv", feat_df.to_csv(index=False))
                archive.writestr("kpi_spec.csv", kpi_df.to_csv(index=False))
                archive.writestr("filter_spec.csv", "\n".join(filter_df["filter"].astype(str)) + "\n")
            yield archive_data.getvalue()

        @reactive.calc
        def dfull():
            """The modelling data: the base data plus the model prediction columns when there are any."""
            frames = [base()]
            if model_cols().shape[1]:
                frames.append(model_cols())
            return pd.concat(frames, axis=1) if len(frames) > 1 else base()

        def chart_column(feature: str) -> pd.Series:
            """A column for the Charts feature: a column of the data (or of the model predictions)."""
            return dfull()[feature]

        @reactive.calc
        def response():
            """The response column chosen on the Data tab."""
            req(input.response())
            return input.response()

        @reactive.calc
        def weight():
            """The weight column chosen, or 'N' for no weight."""
            return input.weight() or "N"

        @render.text
        def gbm_weight_basis_note():
            """Help text for the GBM weight basis."""
            return WEIGHT_BASIS_HELP.get(input.gbm_weight_basis(), "")

        @render.text
        def glm_weight_basis_note():
            """Help text for the GLM weight basis."""
            return WEIGHT_BASIS_HELP.get(input.glm_weight_basis(), "")

        @reactive.calc
        def filt():
            """The row filter: the custom expression if given, otherwise the chosen filter, applied within the selected All / Train / Test rows."""
            custom = (input.filter_custom() or "").strip()
            expr = custom if custom else input.filter_choice()
            tt = input.train_test() if has_tt else "All"
            user, total, msg = apply_filter(dfull(), expr, tt)
            return dict(user=user, total=total, msg=msg)

        @render.ui
        def memory_usage():
            """Memory use of the app and the data, refreshed every 10 seconds and when the models change."""
            # refreshed when the models change and every 10 seconds (the process size moves as the app works)
            reactive.invalidate_later(10)
            glm_models(), gbm_models(), model_cols()
            data_mb = d0.memory_usage(deep=False).sum() / 1e6
            app_mb = _process_memory_mb()
            text = f"Data: {data_mb:,.0f} MB"
            if app_mb is not None:
                text = f"App memory: {app_mb:,.0f} MB \u00b7 " + text
            return ui.p(text, class_="text-muted small mb-0")

        @render.ui
        def row_count():
            """Rows in the selected split and filter, and the counts of data and modelling-valid columns."""
            # rows in the All / Train / Test selection, and how many of them the filter keeps when one is set
            f = filt()
            selected = int(np.sum(np.asarray(f["total"]) == 1))
            tt = input.train_test() if has_tt else "All"
            if tt == "All":
                in_split = len(d0)
            else:
                in_split = int((d0["train_test"] == (0 if tt == "Train" else 1)).sum())
            label = "Data rows" if tt == "All" else f"{tt} rows"
            text = f"{label}: {selected:,}"
            if selected != in_split:
                text += f" of {in_split:,} (filter applied)"
            # columns in the data, and those the feature specification marks modelling_valid = feature
            spec_version()
            modelling_valid = (
                feat_df.loc[
                    feat_df["modelling_valid"].fillna("").astype(str).str.strip().str.lower().eq("feature"), "feature"
                ]
                .astype(str)
                .isin(all_cols)
                .sum()
                if "modelling_valid" in feat_df.columns
                else 0
            )
            return ui.div(
                ui.p(text, class_="text-muted small mb-0"),
                ui.p(f"Data columns: {len(all_cols):,}", class_="text-muted small mb-0"),
                ui.p(f"Modelling valid columns: {int(modelling_valid):,}", class_="text-muted small mb-0"),
            )

        @render.text
        def filter_msg():
            """Message describing the active filter and the rows it keeps."""
            f = filt()
            return f"{filter_text(f['total'])} {f['msg'] if f['msg'] != 'no filter' else ''}".strip()

        # KPI -> response / weight
        @reactive.effect
        @reactive.event(input.kpi, ignore_init=True)
        def _kpi_changed():
            """When the KPI changes, set the response and weight to its numerator and denominator."""
            row = kpi_df[kpi_df["kpi_name"] == input.kpi()]
            if row.empty:
                return
            num, den = row.iloc[0]["kpi_numerator"], row.iloc[0]["kpi_denominator"]
            if num in num_cols:
                ui.update_select("response", selected=num)
            if den == "N" or den in num_cols:
                ui.update_select("weight", selected=den)

        # ---------------- model columns / bookkeeping
        def _put(
            prefix_cols_to_drop: list[str],
            new: dict[str, np.ndarray],
        ):
            """Store new model columns, replacing any existing columns that start with the given prefixes."""
            mc = model_cols().copy()
            mc = mc.drop(
                columns=[c for c in mc.columns if any(c.startswith(p) for p in prefix_cols_to_drop)], errors="ignore"
            )
            for k, v in new.items():
                mc[k] = v
            model_cols.set(mc)

        def _set_prediction_basis(
            column: str,
            model,
        ):
            """Record whether a model's prediction column is per unit of weight (Average response) or a total."""
            with reactive.isolate():
                columns = set(prediction_per_unit())
            (columns.add if model_weight_basis(model) == "average" else columns.discard)(column)
            prediction_per_unit.set(frozenset(columns))

        @reactive.calc
        def chart_per_unit_cols():
            """Response / prediction columns the 1-way charts must multiply by the weight (per unit, not totals)."""
            columns = set(prediction_per_unit())
            if any(
                model_weight_basis(m) == "average" and m.response == response() for m in glm_models() + gbm_models()
            ):
                columns.add(response())
            return columns

        def _full_length(
            positions_or_mask,
            values: np.ndarray,
        ) -> np.ndarray:
            """Array the length of the data (NaN elsewhere) with values at the given row positions."""
            out = np.full(len(d0), np.nan)
            out[positions_or_mask] = values
            return out

        def _show_glm(m):
            """Put a GLM's prediction and feature contribution columns into the model columns."""
            new = {"glm_prediction": _full_length(m.pred_rows, m.predictions)}
            sub = d0.iloc[m.pred_rows]
            fc = m.feature_contributions(sub)
            for c in fc.columns:
                new[c] = _full_length(m.pred_rows, fc[c].to_numpy())
            _put(["glm_prediction", "glm_LP_"], new)
            _set_prediction_basis("glm_prediction", m)

        def _show_gbm(m):
            """Put a GBM's prediction and SHAP columns into the model columns."""
            new = {"lgbm_prediction": _full_length(m.pred_rows, m.predictions)}
            if m.SHAP is not None:
                for c in m.SHAP.columns:
                    new[c] = _full_length(m.SHAP_rows, m.SHAP[c].to_numpy())
            _put(["lgbm_prediction", "lgbm_SHAP_"], new)
            _set_prediction_basis("lgbm_prediction", m)
            ui.update_select("gbm_shap_feature", choices=_ranked_features(d0, m, "gbm", list(m.features)))

        @reactive.effect
        @reactive.event(input.glm_model, ignore_init=True)
        def _glm_selected():
            """When a GLM is chosen on the GLMs tab, show its predictions and contributions in the data."""
            ms = glm_models()
            if not ms or not input.glm_model() or int(input.glm_model()) >= len(ms):
                return  # the selection can be stale for a moment, e.g. after the other models were removed
            _show_glm(ms[int(input.glm_model())])

        @reactive.effect
        @reactive.event(input.gbm_model, ignore_init=True)
        def _gbm_selected():
            """When a GBM is chosen on the GBMs tab, show its predictions and SHAP values in the data."""
            ms = gbm_models()
            if not ms or not input.gbm_model() or int(input.gbm_model()) >= len(ms):
                return
            _show_gbm(ms[int(input.gbm_model())])

        # ---------------- best model: keep one model and remove the rest from the session
        @reactive.effect
        @reactive.event(input.keep_best_model, ignore_init=True)
        def _ask_keep_best_model():
            """Ask for confirmation before keeping one model and removing the others."""
            key = input.chart_feature_model()
            req(key)
            family, index = key.split(":", 1)
            models = glm_models() if family == "glm" else gbm_models()
            req(index.isdigit(), int(index) < len(models))
            others = len(glm_models()) + len(gbm_models()) - 1
            ui.modal_show(
                ui.modal(
                    ui.p(f"Keep {models[int(index)].name} as the best model?"),
                    ui.p(
                        f"The other {others} model{'' if others == 1 else 's'} will be removed from this session, with "
                        "their predictions, SHAP values, ratebooks, metrics and comparisons, to save memory. This cannot "
                        "be undone. The report will then cover only this model.",
                        class_="text-muted",
                    ),
                    title="Keep the best model only",
                    footer=ui.div(
                        ui.modal_button("Cancel"),
                        ui.input_action_button(
                            "confirm_keep_best_model", "Remove the other models", class_="btn-danger"
                        ),
                    ),
                    easy_close=True,
                )
            )

        @reactive.effect
        @reactive.event(input.confirm_keep_best_model, ignore_init=True)
        def _keep_best_model():
            """Keep only the chosen model in the session, removing the others and their outputs to save memory."""
            key = input.chart_feature_model()
            req(key)
            family, index = key.split(":", 1)
            models = glm_models() if family == "glm" else gbm_models()
            req(index.isdigit(), int(index) < len(models))
            best = models[int(index)]
            ui.modal_remove()
            # only the chosen model stays; the lists below feed every selector, the report and the Scoring tab
            glm_models.set([best] if family == "glm" else [])
            gbm_models.set([best] if family == "gbm" else [])
            chart_default_model.set(f"{family}:0")
            overlays.set([o for o in overlays() if o["model"] is best])  # overlays of removed models go too
            model_cols.set(pd.DataFrame(index=d0.index))
            prediction_per_unit.set(frozenset())
            if family == "glm":
                ui.update_select("glm_model", choices={"0": best.name}, selected="0")
                ui.update_select("gbm_model", choices={}, selected=None)
                _show_glm(best)
            else:
                ui.update_select("gbm_model", choices={"0": best.name}, selected="0")
                ui.update_select("glm_model", choices={}, selected=None)
                _show_gbm(best)
            ui.notification_show(
                f"Kept {best.name}. The other models were removed from this session.", type="message", duration=8
            )

        @render.ui
        def gbm_compare_target_ui():
            """Choose the comparison for GBM metrics: a data column, or an older GLM or GBM with the same response."""
            source = input.gbm_compare_source()
            if source == "column":
                columns = [c for c in num_cols if c not in {response(), weight(), "train_test"}]
                return ui.input_select("gbm_compare_column", "Prediction column", choices=columns)
            models = glm_models() if source == "glm" else gbm_models()
            selected_new = input.gbm_model()
            new_model = gbm_models()[int(selected_new)] if selected_new else None
            choices = {
                str(i): model.name
                for i, model in enumerate(models)
                if model.response == (new_model.response if new_model else None)
                and (source != "gbm" or str(i) != selected_new)
            }
            selected_old = str(max(int(selected_new or 0) - 1, 0)) if source == "gbm" and choices else None
            if selected_old not in choices:
                selected_old = next(iter(choices), None)
            return ui.input_select("gbm_compare_model", "Old model", choices=choices, selected=selected_old)

        def _current(
            models,
            select_id,
        ):
            """The model chosen in a drop-down. The request stops when no valid model is chosen."""
            ms = models()
            req(ms, input[select_id]())
            req(int(input[select_id]()) < len(ms))
            return ms[int(input[select_id]())]

        def _column_metrics(
            model,
            column: str,
            rows_mask,
        ):
            """Metrics of a prediction column, measured like ``model`` (its response, weight and objective)."""
            objective, power = model_objective_power(model)
            return prediction_performance_metrics(
                d0,
                model.response,
                d0[column].to_numpy(dtype=float),
                model.weight,
                rows_mask=rows_mask,
                objective=objective,
                tweedie_power=power,
                weight_basis=model_weight_basis(model),
            )

        @render.ui
        def ale_model_selector():
            """Model drop-down for ALE: every GLM and GBM built so far."""
            glm_list, gbm_list = glm_models(), gbm_models()
            choices = {f"glm:{index}": model.name for index, model in enumerate(glm_list)}
            choices.update({f"gbm:{index}": model.name for index, model in enumerate(gbm_list)})
            if not choices:
                return None
            default = chart_default_model()
            return ui.input_select(
                "ale_model", "Model", choices=choices, selected=default if default in choices else next(iter(choices))
            )

        @reactive.calc
        def ale_selected_model():
            """The ALE model chosen, as a (family, model) pair."""
            selected = input.ale_model()
            req(selected)
            family, index = selected.split(":", 1)
            models = glm_models() if family == "glm" else gbm_models()
            req(index.isdigit(), int(index) < len(models))
            return family, models[int(index)]

        @reactive.effect
        def _ale_feature_choices():
            """Set the ALE feature choices to the model's features, most important first. The second feature must be numeric."""
            family, model = ale_selected_model()
            features = _ranked_features(
                d0, model, family, list(model.features(d0.columns) if family == "glm" else model.features)
            )
            numeric_features = [
                feature
                for feature in features
                if feature in d0.columns
                and pd.api.types.is_numeric_dtype(d0[feature])
                and not pd.api.types.is_bool_dtype(d0[feature])
            ]
            with reactive.isolate():
                current = input.ale_feature()
                current_second = input.ale_feature2()
            ui.update_select(
                "ale_feature",
                choices=features,
                selected=current if current in features else (features[0] if features else None),
            )
            ui.update_select(
                "ale_feature2",
                choices={"": "None", **{feature: feature for feature in numeric_features}},
                selected=current_second if current_second in numeric_features else "",
            )

        @reactive.calc
        def ale_result():
            """ALE calculation for the chosen model and feature(s), on the filtered rows with positive weight. Returns (result, message)."""
            family, model = ale_selected_model()
            feature = input.ale_feature()
            req(feature)
            features = [feature] + ([input.ale_feature2()] if input.ale_feature2() else [])
            selected = np.asarray(filt()["total"], dtype=bool)
            if model.weight != "N":
                model_weights = pd.to_numeric(d0[model.weight], errors="coerce").to_numpy(dtype=float)
                selected &= np.isfinite(model_weights) & (model_weights > 0)
            sample = d0.loc[selected].copy()
            limit = input.ale_sample_size()
            bins = input.ale_bins()
            if limit is None or bins is None or not 2 <= limit <= 100000 or not 2 <= bins <= 50:
                return None, "Choose 2-50 bins and a maximum row count between 2 and 100000."
            if len(sample) > int(limit):
                sample = sample.sample(n=int(limit), random_state=1)
            try:
                result = calculate_ale(sample, model, features, int(bins))
            except Exception as error:
                return None, f"ALE unavailable: {error}"
            scale = "offset-free predictions" if family == "gbm" else "model predictions"
            return (
                result,
                features,
            ), f"ALE uses {len(sample):,} filtered rows and {scale}; observations are equally weighted."

        @render.text
        def ale_status():
            """Status line for ALE: a prompt to build a model, or the calculation message."""
            if not glm_models() and not gbm_models():
                return "Build a GLM or GBM to view ALE."
            return ale_result()[1]

        @render_widget
        def ale_plot():
            """ALE plot for the chosen feature(s)."""
            result, _ = ale_result()
            req(result is not None)
            frame, features = result
            return _json_safe_figure(charts.ale_figure(frame, features, [feature_label(feat_df, f) for f in features]))

        @render.data_frame
        def ale_table():
            """Table of the ALE values for the chosen feature(s)."""
            result, _ = ale_result()
            req(result is not None)
            frame, features = result
            if len(features) == 1:
                table = frame.reset_index()
            else:
                table = frame.rename_axis(index=features[0], columns=features[1]).reset_index()
                table.columns = table.columns.map(str)
            return render.DataGrid(_show(table), height="320px")

        @render.ui
        def chart_model_selector():
            """Model drop-down for Charts, with a button to keep only the chosen model."""
            glm_list, gbm_list = glm_models(), gbm_models()
            choices = {f"glm:{index}": model.name for index, model in enumerate(glm_list)}
            choices.update({f"gbm:{index}": model.name for index, model in enumerate(gbm_list)})
            if not choices:
                return None
            default = chart_default_model()
            return ui.div(
                ui.input_select(
                    "chart_feature_model",
                    "Model",
                    choices=choices,
                    selected=default if default in choices else next(iter(choices)),
                ),
                ui.tooltip(
                    ui.input_action_button(
                        "keep_best_model", "Keep only this model", class_="btn-outline-danger btn-sm"
                    ),
                    "Choose this model as the best one: every other GLM and GBM is removed from the session to save "
                    "memory. The report then covers only this model.",
                ),
                class_="d-flex flex-wrap align-items-end gap-2",
            )

        @reactive.effect
        def _group_chart_features():
            """Set the Charts feature choices: data columns grouped as Modelled and Non-modelled."""
            glm_list, gbm_list = glm_models(), gbm_models()
            modelled = None
            if glm_list or gbm_list:
                selected_model = input.chart_feature_model()
                req(selected_model)
                family, index = selected_model.split(":", 1)
                models = glm_list if family == "glm" else gbm_list
                if not index.isdigit() or int(index) >= len(models):
                    return
                model = models[int(index)]
                modelled = _ranked_features(
                    d0, model, family, list(model.features(d0.columns) if family == "glm" else model.features)
                )
            choices = _chart_feature_choices(all_cols, modelled, excluded={response(), weight(), *NOT_NON_MODELLED})
            with reactive.isolate():
                selected_feature = input.x_feature()
                previous_model = chart_features_model()
            current_model = input.chart_feature_model() if (glm_list or gbm_list) else None
            if current_model != previous_model:
                # a model was built, or another one chosen: show its first modelled feature (the banding follows)
                with reactive.isolate():
                    chart_features_model.set(current_model)
                first_modelled = _first_modelled_feature(all_cols, modelled)
                if first_modelled is not None:
                    selected_feature = first_modelled
            offered = set(all_cols)
            ui.update_select(
                "x_feature", choices=choices, selected=selected_feature if selected_feature in offered else None
            )

        # choosing a model in Charts selects it in its GLM / GBM tab too, which refreshes its prediction column
        @reactive.effect
        @reactive.event(input.chart_feature_model, ignore_init=True)
        def _chart_model_selected():
            """Choosing a model in Charts also chooses it on its GLM or GBM tab."""
            selected_model = input.chart_feature_model()
            if not selected_model:
                return
            family, index = selected_model.split(":", 1)
            models = glm_models() if family == "glm" else gbm_models()
            if not index.isdigit() or int(index) >= len(models):
                return
            tab_input = "glm_model" if family == "glm" else "gbm_model"
            with reactive.isolate():
                current = input[tab_input]()
            if current != index:
                ui.update_select(tab_input, selected=index)

        # keep the Charts "add columns" choices in step with model predictions
        @reactive.effect
        def _refresh_add_cols():
            """Keep the Charts 'add columns' choices in step with the model predictions."""
            mc = model_cols()
            preds = [c for c in mc.columns if c.endswith("_prediction")]
            others = [c for c in num_cols if c != input.response()]
            with reactive.isolate():
                cur = list(input.add_cols() or [])
            ui.update_selectize("add_cols", choices=preds + others, selected=list(dict.fromkeys(cur + preds)))

        @reactive.effect
        @reactive.event(input.x_feature)
        def _default_banding():
            """Set the banding, bins, minimum, maximum and monotonicity for a newly chosen Charts feature from the feature specification."""
            f = input.x_feature()
            s = chart_column(f)
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                spec = numeric_or_none(feature_lookup(feat_df, f).get("banding"))
                b = spec if spec else banding_guesser(s)
            else:
                b = 0
            # the minimum, maximum and bins start from what the specification applies to modelling
            applied = feature_treatment_spec(feat_df, f) or {}
            minimum, maximum = spec_chart_limits(feat_df, f)
            ui.update_text("x_banding", value="0" if applied.get("bins") else f"{b:g}")  # bins replace the banding
            ui.update_text("x_bins", value=str(applied["bins"]) if applied.get("bins") else "")
            ui.update_text("x_min", value="" if minimum is None else f"{minimum:g}")
            ui.update_text("x_max", value="" if maximum is None else f"{maximum:g}")
            ui.update_select("x_monotone", selected=feature_monotonicity(feat_df, f))

        def _bins_available() -> bool:
            """Bins can be used for a numeric feature whose Banding is blank or 0."""
            f = input.x_feature()
            if (
                not f
                or not pd.api.types.is_numeric_dtype(chart_column(f))
                or pd.api.types.is_bool_dtype(chart_column(f))
            ):
                return False
            banding = (input.x_banding() or "").strip()
            return banding == "" or numeric_or_none(banding) == 0

        @reactive.effect
        async def _grey_out_bins():
            """Grey out the Bins box unless bins apply to the current feature."""
            await session.send_custom_message("daisy_disable", {"id": "x_bins", "disabled": not _bins_available()})

        def _chart_limits():
            """(minimum, maximum, bins) typed in Charts; blank or non-numeric limits are None, and the bins are
            None while a banding is set (the Bins box is greyed out then)."""
            bins = (input.x_bins() or "").strip() if _bins_available() else ""
            return (
                numeric_or_none((input.x_min() or "").strip()),
                numeric_or_none((input.x_max() or "").strip()),
                bin_count(bins),
            )
        def _chart_settings_for(feature: str, own_banding):
            """(banding, minimum, maximum, bins) for a SHAP feature: the Charts settings when it is the feature chosen
            in Charts; otherwise its own banding box with the minimum, maximum and bins from its specification, as the
            1-way chart uses for a feature not chosen in Charts."""
            if feature and feature == input.x_feature():
                banding = (input.x_banding() or "").strip()
                minimum, maximum, bins = _chart_limits()
                return (banding if numeric_or_none(banding) is not None else "0"), minimum, maximum, bins
            applied = feature_treatment_spec(feat_df, feature) or {} if feature else {}
            minimum, maximum = spec_chart_limits(feat_df, feature) if feature else (None, None)
            return (own_banding or "0"), minimum, maximum, applied.get("bins")

        @render.ui
        def x_monotone_label():
            """Monotonicity of the chosen Charts feature."""
            spec_version()
            f = input.x_feature()
            req(f)
            mono = feature_monotonicity(feat_df, f) or "none"
            return ui.div(f"Monotonicity: {mono}", class_="text-end text-muted small")

        @render.ui
        def x_spec_note():
            """Note on how the chosen Charts feature is modelled, from its specification."""
            spec_version()
            f = input.x_feature()
            req(f)
            try:
                applied = treatments_from_spec(d0, feat_df, [f]).get(f)
            except Exception:
                applied = None
            if not applied:
                return None
            return ui.p(f"Used for modelling: {f} with {treatment_text(applied)}.", class_="text-muted small mb-0")

        @reactive.effect
        @reactive.event(input.x_apply_spec)
        def _update_specification_from_chart():
            """Save the banding, minimum, maximum, bins and monotonicity from Charts to the feature specification, and to its CSV file when one was given."""
            nonlocal feat_df, monotone_features, monotone_choices
            f = input.x_feature()
            req(f)
            if f not in all_cols:
                ui.notification_show(f"{f} is a model column, so it has no specification to save.", type="warning")
                return
            s = d0[f]
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                ui.notification_show(
                    f"{f} is not numeric, so it has no banding, minimum or maximum to save.", type="warning"
                )
                return
            banding = numeric_or_none((input.x_banding() or "").strip())
            try:
                minimum, maximum, bins = _chart_limits()
                if banding is not None and banding < 0:
                    raise ValueError("the banding cannot be negative")
                treatment = resolve_treatment(s, minimum, maximum, bins)  # checks the settings
                updated = set_feature_monotonicity(
                    set_feature_treatment(feat_df, f, minimum, maximum, banding, bins), f, input.x_monotone()
                )
            except Exception as error:
                _notify_error("Specification not updated", error)
                return
            feat_df = updated
            monotone_features = feat_df.loc[
                feat_df["monotonicity"].fillna("").astype(str).str.strip().ne(""), "feature"
            ].tolist()
            monotone_choices = {
                feature: f"{feature} ({feat_df.loc[feat_df['feature'] == feature, 'monotonicity'].iloc[0]})"
                for feature in monotone_features
            }
            ui.update_checkbox_group("gbm_monotone", choices=monotone_choices, selected=monotone_features)
            saved = "for this session (the app was not given a specification csv file to write to)"
            if feature_spec_path is not None:
                try:
                    feat_df.to_csv(feature_spec_path, index=False)
                    saved = f"and saved to {feature_spec_path.name}"
                except Exception as error:
                    saved = f"for this session only: {feature_spec_path.name} could not be written ({error})"
            spec_version.set(spec_version() + 1)
            used = f"{f} is modelled with {treatment_text(treatment)}" if treatment else f"{f} is modelled as it is"
            if feature_monotonicity(feat_df, f):
                used += f", {feature_monotonicity(feat_df, f)}"
            rebuild = " Rebuild existing models to use it." if (glm_models() or gbm_models()) else ""
            ui.notification_show(
                f"Feature specification updated {saved}. From now on {used}.{rebuild}", type="message", duration=10
            )

        # ---------------- Charts: 1-way line and bar
        @reactive.calc
        def lb_summary():
            """One-way summary for the chosen Charts feature, with the banding, limits, filter and per-unit columns applied."""
            f, b = input.x_feature(), (input.x_banding() or "0").strip()
            req(f)
            if pd.api.types.is_numeric_dtype(chart_column(f)):
                try:
                    float(b)
                except ValueError:
                    b = "0"
            try:
                minimum, maximum, bins = _chart_limits()
                return one_way_summary(
                    dfull(),
                    response(),
                    weight(),
                    f,
                    list(input.add_cols() or []),
                    banding=b,
                    group_low_exposure=input.low_exposure(),
                    total_filter=filt()["total"],
                    per_unit_cols=chart_per_unit_cols(),
                    minimum=minimum,
                    maximum=maximum,
                    bins=bins,
                )
            except Exception as e:  # pragma: no cover
                _notify_error("Chart error", e)
                req(False)

        def y_title():
            """Y-axis title for the 1-way charts, from the KPI."""
            return _kpi_axis_title(kpi_df, input.kpi(), response(), weight())

        @render_widget
        def line_bar_plot():
            """1-way line and bar chart for the chosen Charts feature."""
            s = lb_summary()
            return _json_safe_figure(
                charts.line_bar_figure(
                    s,
                    input.x_feature(),
                    response(),
                    list(input.add_cols() or []),
                    show_bars=input.show_bars(),
                    sigma_bars=input.sigma_bars(),
                    y_title=y_title(),
                    weight_label=weight(),
                    x_title=feature_label(feat_df, input.x_feature()),
                )
            )

        @render.data_frame
        def line_bar_table():
            """Table behind the 1-way chart, with a total row and the response labelled by its KPI."""
            s = add_total_row(lb_summary(), response(), list(input.add_cols() or []))
            s = _label_response_with_kpi(s, response(), y_title())
            s = _label_weight_column(s, weight())
            return render.DataGrid(_show(s), height="320px")

        def two_way_top() -> int:
            """How many top features the report's two-way SHAP and ALE views cover (2 to 10)."""
            value = input.two_way_n()
            return min(max(int(value), 2), 10) if value is not None else TWO_WAY_FEATURES

        @render.download_button(filename=lambda: export_filename("daisy_charts", input.kpi(), ".html"))
        def line_bar_export():
            """Download the Charts views as an HTML report: 1-way charts for every column, with the chosen model's overview, metrics, ALE, lift and importance when a model is built."""
            glm_list, gbm_list = glm_models(), gbm_models()
            modelled, extra_sections, leading_sections, model_name = None, [], [], None
            full = dfull()
            user_rows = np.asarray(filt()["user"])

            # Rows views: All / Train / Test share the user filter; the Rows radio only sets the opening view
            view_masks = {"All": user_rows}
            if has_tt:
                tt = pd.to_numeric(full["train_test"], errors="coerce").fillna(0).to_numpy()
                for name, mask in (("Train", user_rows * (1 - tt)), ("Test", user_rows * tt)):
                    mask = np.asarray(mask).astype(np.int8)
                    if mask.sum() > 0:
                        view_masks[name] = mask
            default_view = input.train_test() if has_tt and input.train_test() in view_masks else "All"

            if glm_list or gbm_list:
                selected_model = input.chart_feature_model()
                req(selected_model)
                family, index = selected_model.split(":", 1)
                models = glm_list if family == "glm" else gbm_list
                req(index.isdigit(), int(index) < len(models))
                model = models[int(index)]
                model_name = model.name
                modelled = _ranked_features(
                    d0, model, family, list(model.features(d0.columns) if family == "glm" else model.features)
                )

                # Model overview: first tab, Train and Test side by side (follows the user filter only)
                overview = model_overview(
                    d0,
                    model.response,
                    model.predictions,
                    model.weight,
                    model.pred_rows,
                    user_rows,
                    weight_basis=model_weight_basis(model),
                )
                weight_text = "row counts (weight N)" if model.weight == "N" else f"the {model.weight} column"
                gbm_tables = (
                    []
                    if family != "gbm"
                    else [
                        ("GBM parameters", gbm_parameter_table(model)),
                        ("Monotonicity applied", gbm_monotonicity_table(model)),
                    ]
                )
                leading_sections.append(
                    report.overview_section(
                        overview,
                        tables=gbm_tables,
                        note=(
                            f"{model.name}. Total Weight is the sum of {weight_text}; Total Response is the sum of the response. Avg Actual - Weighted and "
                            "Avg Predicted - Weighted are the actual and predicted per unit of weight. The minimum, quartiles, mean and "
                            "maximum describe the unweighted row-level predictions. Uses the current row filter."
                        ),
                    )
                )

                # Metrics: the selected model, train / test, with the current filters (before the 1-way plots)
                def metrics_for(mask):
                    """Metrics section for the chosen model on the rows of one view (All, Train or Test)."""
                    metrics = model_performance_metrics(d0, model, rows_mask=mask)
                    return report.metrics_section(
                        metrics,
                        note=(
                            f"{model.name}. Metrics use the current row filters, the model's weight and, for PoDE, "
                            f"the {normalise_objective(model_objective_power(model)[0])} deviance."
                        ),
                    )

                leading_sections.append(metrics_for)

                # SHAP: GBMs only - the selected GBM, otherwise the most recently built one
                shap_model = model if family == "gbm" else (gbm_list[-1] if gbm_list else None)
                if shap_model is not None:
                    shap_note = (
                        None
                        if shap_model is model
                        else (f"SHAP values come from {shap_model.name}, the latest GBM built.")
                    )
                    quantile = input.shap_q() or 0.05
                    shap_features = _ranked_features(d0, shap_model, "gbm", list(shap_model.features))
                    # two-way SHAP heatmaps for every pair of the top features
                    shap_pairs = list(combinations(shap_features[: two_way_top()], 2))
                    extra_sections.append(
                        lambda mask, m=shap_model, n=shap_note, q=quantile, fs=shap_features, pr=shap_pairs: report.shap_section(
                            full,
                            m,
                            fs,
                            feat_df,
                            weight(),
                            mask,
                            quantile=q,
                            note=n,
                            pairs=pr,
                            selected_feature=input.x_feature(),
                            selected_banding=input.x_banding() or "0",
                            selected_limits=_chart_limits(),
                        )
                    )

                # ALE: any fitted GLM or GBM
                bins, limit = input.ale_bins(), input.ale_sample_size()
                bins = int(bins) if bins is not None and 2 <= bins <= 50 else 20
                limit = int(limit) if limit is not None and 2 <= limit <= 100000 else 10000
                # two-way ALE plots for every pair of the top features (modelled is already most important first)
                ale_pairs = list(combinations(modelled[: two_way_top()], 2)) if modelled else []
                extra_sections.append(
                    lambda mask, pr=ale_pairs: report.ale_section(
                        d0, model, family, modelled, mask, bins, limit, feature_spec=feat_df, pairs=pr
                    )
                )

                # Lift Chart: the selected model, train / test, with the current filters (after ALE)
                try:
                    lift_bins = int(input[f"{family}_lift_bins"]())
                except Exception:  # the control has not been used: the default bin count
                    lift_bins = charts.LIFT_DEFAULT_BINS
                extra_sections.append(
                    lambda mask, bins=lift_bins: report.lift_section(
                        d0,
                        model,
                        model.weight,
                        mask,
                        bins,
                        note=f"{model.name}. The Lift Chart uses the current row filters.",
                    )
                )

                # Feature importance: the selected model, after the Lift Chart (does not follow the row filter)
                extra_sections.append(
                    report.importance_section(
                        d0, model, family, note=f"{model.name}. Feature importance does not follow the row filters."
                    )
                )
            # the 1-way plots: the Modelled features most important first, then the others (response and weight
            # left out); with no model every column is shown in its order
            chart_order = (modelled or []) + [c for c in all_cols if c not in set(modelled or [])]
            yield report.export_line_bar_plots(
                full,
                chart_order,
                modelled,
                feat_df,
                response(),
                weight(),
                list(input.add_cols() or []),
                user_rows,
                input.low_exposure(),
                input.show_bars(),
                input.sigma_bars(),
                selected_feature=input.x_feature(),
                selected_banding=input.x_banding() or "0",
                selected_limits=_chart_limits(),
                valid_features=[c for c in all_cols if c not in {response(), weight(), *NOT_NON_MODELLED}],
                extra_sections=extra_sections,
                model_name=model_name,
                view_masks=view_masks,
                default_view=default_view,
                y_title=y_title(),
                leading_sections=leading_sections,
                app_title=f"{input.kpi() or 'Daisy'} - {'Model Report' if model_name else 'Report'}",
                per_unit_cols=chart_per_unit_cols(),
            )

        # ---------------- Charts: SHAP
        @reactive.effect
        def _shap_choices():
            """Set the Charts SHAP feature choices to the GBM's features, most important first (the chosen GBM, or the latest one)."""
            ms = gbm_models()
            if not ms:
                return
            index = int(input.gbm_model()) if input.gbm_model() else len(ms) - 1
            chosen = ms[index] if index < len(ms) else ms[-1]
            feats = _ranked_features(d0, chosen, "gbm", list(chosen.features))
            ui.update_select("shap_f1", choices=feats)
            ui.update_select("shap_f2", choices=["none"] + feats)

        def _shap_banding(
            f: str,
            current: str,
        ):
            """Banding for a SHAP feature: its specification's banding when numeric (0 when it has bins), else a guess
            from its values; 0 for a non-numeric feature."""
            s = d0[f]
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                return 0
            if (feature_treatment_spec(feat_df, f) or {}).get("bins"):
                return 0
            configured = numeric_or_none(feature_lookup(feat_df, f).get("banding"))
            return configured if configured else banding_guesser(s)

        @reactive.effect
        @reactive.event(input.shap_f1)
        def _shap_b1():
            """Fill the first SHAP feature's banding with its guess."""
            if input.shap_f1():
                ui.update_text("shap_b1", value=f"{_shap_banding(input.shap_f1(), ''):g}")

        @reactive.effect
        @reactive.event(input.shap_f2)
        def _shap_b2():
            """Fill the second SHAP feature's banding with its guess."""
            if input.shap_f2() and input.shap_f2() != "none":
                ui.update_text("shap_b2", value=f"{_shap_banding(input.shap_f2(), ''):g}")

        @render_widget
        def shap_plot():
            """SHAP chart for the chosen GBM: a flame chart for one feature, or a heatmap when a second feature is chosen."""
            m = _current(gbm_models, "gbm_model")
            req(input.shap_f1())
            mask = filt()["total"] == 1
            try:
                banding1, minimum1, maximum1, bins1 = _chart_settings_for(input.shap_f1(), input.shap_b1())
                if input.shap_f2() in ("none", "", None):
                    figure = charts.shap_flame_figure(
                        dfull(),
                        m,
                        input.shap_f1(),
                        weight(),
                        banding1,
                        mask,
                        quantile=input.shap_q(),
                        weight_label=weight(),
                        x_title=feature_label(feat_df, input.shap_f1()),
                        minimum=minimum1,
                        maximum=maximum1,
                        bins=bins1,
                    )
                else:
                    banding2, minimum2, maximum2, bins2 = _chart_settings_for(input.shap_f2(), input.shap_b2())
                    figure = charts.shap_heatmap_figure(
                        dfull(),
                        m,
                        input.shap_f1(),
                        input.shap_f2(),
                        weight(),
                        banding1,
                        banding2,
                        mask,
                        x_title=feature_label(feat_df, input.shap_f1()),
                        y_title=feature_label(feat_df, input.shap_f2()),
                        limits1=(minimum1, maximum1, bins1),
                        limits2=(minimum2, maximum2, bins2),
                    )
                return _json_safe_figure(figure)
            except Exception as e:
                _notify_error("SHAP chart", e)
                req(False)

        @render_widget
        def shap_stacked():
            """Stacked SHAP chart for the chosen feature."""
            m = _current(gbm_models, "gbm_model")
            req(input.shap_f1())
            try:
                banding1, minimum1, maximum1, bins1 = _chart_settings_for(input.shap_f1(), input.shap_b1())
                figure = charts.stacked_shap_figure(
                    dfull(),
                    m,
                    input.shap_f1(),
                    weight(),
                    banding1,
                    filt()["total"] == 1,
                    x_title=feature_label(feat_df, input.shap_f1()),
                    minimum=minimum1,
                    maximum=maximum1,
                    bins=bins1,
                )
                return _json_safe_figure(figure)
            except Exception as e:
                _notify_error("Stacked SHAP", e)
                req(False)

        # ---------------- Maps
        @reactive.effect
        def _map_model_choices():
            """Set the Maps model drop-down to the models built so far."""
            choices = overlay_models()
            with reactive.isolate():
                current = input.map_model()
            ui.update_select(
                "map_model", choices=choices, selected=current if current in choices else next(iter(choices), None)
            )

        @render_widget
        def map_plot():
            """Postcode-area map of the chosen measure. The predicted and A/E measures use the chosen model."""
            from . import maps

            metric = input.map_metric()
            d = dfull()
            expected = None
            if metric in ("predicted", "ae"):
                key = input.map_model()
                if not key or not overlay_models():
                    ui.notification_show("Build a model first: the predictions come from a GLM or GBM.", type="warning")
                    req(False)
                family, model = model_by_key(key)
                expected = np.asarray(expected_for(family, model, d), dtype=float)
                if family != "ratebook" and model_weight_basis(model) == "average":
                    expected = expected * (np.ones(len(d)) if weight() == "N" else d[weight()].to_numpy(float))
            try:
                s = maps.postcode_area_summary(d, response(), weight(), filt()["total"], expected=expected)
                return _json_safe_figure(maps.area_choropleth(s, metric))
            except ImportError as e:
                _notify_error("Maps needs geopandas and plotly", e)
                req(False)

        @render.text
        def map_note():
            """Note on the map's geography: the sector-level geometry is not in the repo."""
            return "Postcode-area resolution (the sector-level geometry is not included in the repo)."

        # ---------------- Data
        @render.data_frame
        def data_view():
            """Filterable grid of the first 2,000 rows of the data."""
            return render.DataGrid(_show(dfull().head(2000)), filters=True, height="650px")

        @render.data_frame
        def col_summary():
            """Summary table of every data column."""
            return render.DataGrid(_show(column_summary(dfull())), height="650px")

        # ---------------- GLMs
        @render.ui
        def glm_builder_chips():
            """The features the GLM builder offers: the modelling-valid features of the feature specification (the
            same list the GBM defaults and the Non-modelled features use), without the response and weight."""
            features = []
            for c in _gbm_features_from_spec(feat_df, all_cols, response(), weight()):
                s = d0[c]
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    features.append((c, feature_label(feat_df, c), "numeric"))
                else:  # a categorical feature: its levels, for the reference and level pickers (first 300 shown)
                    levels = sorted(pd.unique(s.dropna()), key=str)[:300]
                    features.append((c, feature_label(feat_df, c), "categorical", levels))
            return feature_chips(features)

        def _fit_glm(
            positive_only: bool,
        ):
            """Build a GLM from the GLMs tab settings and add it to the session. With ``positive_only`` the rows whose
            response or weight is not above 0 are left out of this model."""
            with ui.Progress(min=0, max=1) as prog:
                prog.set(0.3, message="GLM", detail="building model")
                try:
                    # base levels from the feature spec apply to bare categorical terms
                    bl = {
                        r["feature"]: r["base_level"]
                        for r in feat_df.to_dict("records")
                        if r.get("base_level", "") != ""
                        and r["feature"] in d0.columns
                        and not pd.api.types.is_numeric_dtype(d0[r["feature"]])
                    }
                    m = build_glm(
                        d0,
                        response(),
                        weight(),
                        input.glm_formula(),
                        input.glm_objective(),
                        input.glm_var_power(),
                        input.glm_data(),
                        base_levels=bl,
                        weight_basis=input.glm_weight_basis(),
                        treatments=treatments_from_spec(d0, feat_df, all_cols),
                        positive_response=positive_only,
                    )
                except NonPositiveResponseError as e:
                    ui.modal_show(_positive_response_modal(e.count, e.total, "GLM", "glm_positive_ok"))
                    return
                except Exception as e:
                    _notify_error("GLM build error", e)
                    return
                suffix = " (response > 0)" if positive_only else ""
                m.name = f"GLM {len(glm_models()) + 1}: {m.objective} ~ {y_title()}{suffix}"
                ms = glm_models() + [m]
                glm_models.set(ms)
                chart_default_model.set(f"glm:{len(ms) - 1}")
                prog.set(0.9, detail="done")
            ui.update_select("glm_model", choices={str(i): x.name for i, x in enumerate(ms)}, selected=str(len(ms) - 1))
            ui.update_select(
                "glm_rb_feature", choices=_ranked_features(d0, m, "glm", list(m.features(d0.columns)))
            )  # most important first

        @reactive.effect
        @reactive.event(input.glm_build)
        def _build_glm():
            """Build a GLM from the GLMs tab settings, using every row."""
            _fit_glm(False)

        @reactive.effect
        @reactive.event(input.glm_positive_ok)
        def _build_glm_positive():
            """After the user agrees, build the GLM on the rows with a response and a weight above 0. The data is not changed."""
            ui.modal_remove()
            _fit_glm(True)

        @render.text
        def glm_stats():
            """One-line summary of the chosen GLM: terms, deviance, AIC, dispersion and convergence."""
            m = _current(glm_models, "glm_model")
            f = m.fit
            return (
                f"{m.name} | terms {len(f.columns) - 1} | deviance {f.deviance:,.2f} | AIC {f.aic:,.1f} | "
                f"dispersion {f.dispersion:.4f} | iterations {f.iterations}{'' if f.converged else ' (not converged)'}"
            )

        @render.data_frame
        def glm_coefs():
            """Coefficient table of the chosen GLM."""
            m = _current(glm_models, "glm_model")
            return render.DataGrid(_show(m.coefficients(), 5), height="600px")

        @render.data_frame
        def glm_imp_table():
            """Importance of the chosen GLM: the mean absolute linear-predictor contribution, scaled by 100. By term (the
            default), each term is its own row, so an interaction is shown separately. By feature, each feature's terms
            are summed, so an interaction counts towards each feature it uses."""
            m = _current(glm_models, "glm_model")
            if input.glm_imp_view() == "feature":
                fc = m.feature_contributions(d0.iloc[m.pred_rows])
                imp = fc.abs().mean().sort_values(ascending=False) * 100
                frame = pd.DataFrame(
                    {"feature": [c.replace("glm_LP_", "") for c in imp.index], "mean |LP| x 100": imp.values}
                )
            else:
                imp = m.term_importance(d0.iloc[m.pred_rows])
                frame = pd.DataFrame({"term": imp.index, "mean |LP| x 100": imp.values})
            return render.DataGrid(_show(frame))

        @render_widget
        def glm_lift_plot():
            """Lift chart of the chosen GLM for the current filter."""
            model = _current(glm_models, "glm_model")
            try:
                figure = charts.lift_chart_figure(d0, model, model.weight, filt()["total"], int(input.glm_lift_bins()))
            except Exception as e:  # show the reason instead of failing the session
                _notify_error("Lift chart error", e)
                req(False)
            return _json_safe_figure(figure)

        @render_widget
        def glm_lorenz_plot():
            """Lorenz curve of the chosen GLM for the current filter."""
            model = _current(glm_models, "glm_model")
            try:
                figure = charts.lorenz_figure(d0, model, model.weight, filt()["total"])
            except Exception as e:
                _notify_error("Lorenz curve error", e)
                req(False)
            return _json_safe_figure(figure)

        @render.data_frame
        def glm_lift_table():
            """A/E by bin table of the chosen GLM."""
            model = _current(glm_models, "glm_model")
            try:
                table = charts.lift_table_frame(d0, model, model.weight, filt()["total"], int(input.glm_lift_bins()))
            except Exception as e:
                _notify_error("A/E by bin error", e)
                req(False)
            return render.DataGrid(_show(table), height="340px")

        @render.data_frame
        def glm_metrics_table():
            """Performance metrics of the chosen GLM for the current filter."""
            model = _current(glm_models, "glm_model")
            metrics = model_performance_metrics(d0, model, rows_mask=filt()["total"])
            return render.DataGrid(_show(metrics), height="200px")

        @render.data_frame
        def glm_builder_metrics():
            """The Builder's metrics: the same performance metrics as the Metrics tab, for the chosen GLM."""
            model = _current(glm_models, "glm_model")
            metrics = model_performance_metrics(d0, model, rows_mask=filt()["total"])
            return render.DataGrid(_show(metrics), height="200px")

        @render.ui
        def glm_compare_target_ui():
            """Comparison for GLM metrics: a data column, or an older GLM or GBM with the same response."""
            source = input.glm_compare_source()
            if source == "column":
                columns = [c for c in num_cols if c not in {response(), weight(), "train_test"}]
                return ui.input_select("glm_compare_column", "Prediction column", choices=columns)
            models = gbm_models() if source == "gbm" else glm_models()
            selected_new = input.glm_model()
            new_model = glm_models()[int(selected_new)] if selected_new else None
            choices = {
                str(i): model.name
                for i, model in enumerate(models)
                if model.response == (new_model.response if new_model else None)
                and (source != "glm" or str(i) != selected_new)
            }
            selected_old = str(max(int(selected_new or 0) - 1, 0)) if source == "glm" and choices else None
            if selected_old not in choices:
                selected_old = next(iter(choices), None)
            return ui.input_select("glm_compare_model", "Old model", choices=choices, selected=selected_old)

        @render.ui
        def glm_metrics_comparison():
            """Metrics comparison of the chosen GLM against a column or an older model, for the chosen split."""
            models = glm_models()
            selected_new = input.glm_model()
            if not models or not selected_new:
                return ui.p("Build a GLM model to compare performance metrics.")

            new_model = models[int(selected_new)]
            active_rows = filt()["total"]
            new_metrics = model_performance_metrics(d0, new_model, rows_mask=active_rows)
            compare_source = input.glm_compare_source()
            if compare_source == "column":
                column = input.glm_compare_column()
                if not column or column not in d0.columns:
                    return ui.p("Choose a numeric prediction column to compare.")
                old_name = column
                old_metrics = _column_metrics(new_model, column, active_rows)
            else:
                selected_old = input.glm_compare_model()
                if not selected_old or (compare_source == "glm" and selected_old == selected_new):
                    if compare_source == "glm":
                        return ui.p("Choose a different old GLM model to compare.")
                    return ui.p("Choose an old GBM model to compare.")
                old_models = gbm_models() if compare_source == "gbm" else models
                old_model = old_models[int(selected_old)]
                if old_model.response != new_model.response:
                    return ui.p("Choose a model fitted to the same response.")
                old_name = old_model.name
                old_metrics = model_performance_metrics(d0, old_model, rows_mask=active_rows)

            comparison = compare_gbm_metrics(new_metrics, old_metrics)
            comparison = comparison[comparison["Dataset"] == input.glm_compare_split()]
            return metrics_comparison_table(comparison, f"New: {new_model.name}", f"Old: {old_name}")

        @reactive.calc
        def glm_tabs():
            """Ratebook tables, one per feature, for the chosen GLM."""
            spec_version()
            m = _current(glm_models, "glm_model")
            return tabulate_model(d0, lambda x: m.predict(x), m.features(d0.columns), feat_df, link=m.link)

        @render_widget
        def glm_rb_plot():
            """Ratebook chart of relativities for the chosen GLM feature."""
            t = glm_tabs()
            f = input.glm_rb_feature()
            req(f, f in t)
            return _json_safe_figure(charts.ratebook_figure(t[f], f))

        @render.data_frame
        def glm_rb_table():
            """Ratebook table of relativities for the chosen GLM feature."""
            t = glm_tabs()
            f = input.glm_rb_feature()
            req(f, f in t)
            return render.DataGrid(_show(t[f]), height="300px")

        @render.download_button(filename="glm_ratebook.xlsx")
        def glm_download():
            """Download the chosen GLM's ratebook as an Excel workbook."""
            try:
                check_ratebook_link(_current(glm_models, "glm_model").link)
            except ValueError as e:
                _notify_error("Ratebook", e)
                req(False)
            t = glm_tabs()
            with tempfile.TemporaryDirectory() as tmp:
                glm = _current(glm_models, "glm_model")
                path = write_ratebook_excel(
                    t, str(Path(tmp) / "rb.xlsx"), glm.name, weight_basis=glm.weight_basis, weight=glm.weight
                )
                yield Path(path).read_bytes()

        @render.download_button(filename="glm_formula.txt")
        def glm_formula_export():
            """Download the GLM formula as text: one term per line."""
            yield formula_to_text(input.glm_formula() or "")

        @reactive.effect
        @reactive.event(input.glm_formula_file)
        def _import_glm_formula():
            """Load a formula from a text file (one term per line) into the Formula box."""
            files = input.glm_formula_file()
            if not files:
                return
            try:
                text = Path(files[0]["datapath"]).read_text(encoding="utf-8-sig")
                formula = formula_from_import(text)
            except (ValueError, UnicodeDecodeError) as e:
                ui.notification_show(f"Could not import the formula: {e}", type="warning")
                return
            ui.update_text_area("glm_formula", value=formula)
            ui.notification_show(
                "Formula imported into the Formula tab. The Builder does not read imported text, so its "
                "terms are not changed.",
                type="message",
            )

        @render.download_button(filename="daisy_glm_predictions.csv")
        def glm_predictions_download():
            """Download the chosen GLM's predictions for every row as CSV."""
            model = _current(glm_models, "glm_model")
            predictions = model.score(d0)  # totals for a total-response model, as the GBM export
            prediction_csv = _prediction_export_frame(d0, model.response, predictions).to_csv(index=False)
            yield prediction_csv

        # ---------------- GBMs
        def _fit_gbm(
            positive_only: bool,
        ):
            """Build a GBM from the GBMs tab settings and add it to the session. With ``positive_only`` the rows whose
            response is zero, negative or missing are left out of this model."""
            available_features = _gbm_features_from_spec(feat_df, all_cols, response(), weight())
            feats = [c for c in (input.gbm_features() or []) if c in available_features]
            if not feats:
                ui.notification_show("Select at least one feature", type="warning")
                return
            params = dict(
                objective=input.gbm_objective(),
                num_iterations=int(input.gbm_rounds()),
                early_stopping_round=int(input.gbm_early()),
                learning_rate=input.gbm_lr(),
                num_leaves=int(input.gbm_leaves()),
                max_depth=int(input.gbm_depth()),
                min_data_in_leaf=int(input.gbm_min_data()),
                bagging_fraction=input.gbm_row_sample(),
                feature_fraction=input.gbm_col_sample(),
                lambda_l1=input.gbm_l1(),
                lambda_l2=input.gbm_l2(),
            )
            gbm_feature_spec = feat_df.copy()
            if "monotonicity" in gbm_feature_spec.columns:
                selected_monotone = set(input.gbm_monotone() or [])
                gbm_feature_spec.loc[~gbm_feature_spec["feature"].isin(selected_monotone), "monotonicity"] = ""
            if input.gbm_objective() == "tweedie":
                params["tweedie_variance_power"] = input.gbm_tweedie_p()
            with ui.Progress(min=0, max=1) as prog:
                prog.set(0.2, message="GBM", detail="training")
                try:
                    m = build_gbm(
                        d0,
                        response(),
                        weight(),
                        feats,
                        params,
                        init_score=input.gbm_offset(),
                        feature_table=gbm_feature_spec,
                        interaction_groups=list(input.gbm_groups() or []) or None,
                        shap_sample=input.gbm_shap(),
                        num_threads=num_threads,
                        weight_basis=input.gbm_weight_basis(),
                        treatments=treatments_from_spec(d0, feat_df, feats),
                        positive_response=positive_only,
                    )
                except NonPositiveResponseError as e:
                    ui.modal_show(_positive_response_modal(e.count, e.total, "GBM", "gbm_positive_ok"))
                    return
                except Exception as e:
                    _notify_error("GBM build error", e)
                    return
                suffix = " (response > 0)" if positive_only else ""
                m.name = f"GBM {len(gbm_models()) + 1}: {input.gbm_objective()} ~ {y_title()}{suffix}"
                ms = gbm_models() + [m]
                gbm_models.set(ms)
                chart_default_model.set(f"gbm:{len(ms) - 1}")
                prog.set(0.95, detail="done")
            ui.update_select("gbm_model", choices={str(i): x.name for i, x in enumerate(ms)}, selected=str(len(ms) - 1))
            ranked = _ranked_features(d0, m, "gbm", list(m.features))  # most important first
            ui.update_select("gbm_rb_feature", choices=ranked)
            ui.update_select("gbm_shap_feature", choices=ranked)

        @reactive.effect
        @reactive.event(input.gbm_build)
        def _build_gbm():
            """Build a GBM from the GBMs tab settings, using every row."""
            _fit_gbm(False)

        @reactive.effect
        @reactive.event(input.gbm_positive_ok)
        def _build_gbm_positive():
            """After the user agrees, build the GBM on the rows with a response above 0. The data itself is not changed."""
            ui.modal_remove()
            _fit_gbm(True)

        @render_widget
        def gbm_eval_plot():
            """Learning curves of the chosen GBM."""
            return _json_safe_figure(charts.evaluation_figure(_current(gbm_models, "gbm_model")))

        @render.text
        def gbm_stats():
            """Summary of the chosen GBM: best iteration, last metric values and run time."""
            m = _current(gbm_models, "gbm_model")
            last = {k: v[m.best_iteration - 1] for k, v in m.evaluation_log.items() if len(v) >= m.best_iteration}
            return (
                f"{m.name} | best iteration {m.best_iteration} | "
                + " | ".join(f"{k} {m.params.get('metric')} {v:.5f}" for k, v in last.items())
                + f" | {m.run_time:.1f}s"
            )

        @render.data_frame
        def gbm_overview_table():
            """Overview statistics of the chosen GBM for the current row filter."""
            model = _current(gbm_models, "gbm_model")
            overview = model_overview(
                d0,
                model.response,
                model.predictions,
                model.weight,
                model.pred_rows,
                filt()["user"],
                weight_basis=model_weight_basis(model),
            )
            return render.DataGrid(_show(overview), height=None)  # all eight statistics, no scrolling

        @render.ui
        def gbm_params_table():
            """Table of the GBM's parameters, with an explanation on hover."""
            # an HTML table, so each parameter shows its explanation when hovered over
            table = gbm_parameter_table(_current(gbm_models, "gbm_model"))
            html = report.text_table_html(table, hover=parameter_help).replace(
                'class="metrics text"', 'class="table table-sm table-striped w-auto"'
            )
            return ui.HTML(html)

        @render.data_frame
        def gbm_mono_table():
            """Monotonicity constraints applied in the chosen GBM."""
            return render.DataGrid(gbm_monotonicity_table(_current(gbm_models, "gbm_model")), height="380px")

        @render_widget
        def gbm_imp_plot():
            """Feature importance chart of the chosen GBM, sized to its number of features."""
            # every feature, with the height set by the number of bars (the container is height="auto")
            importances = _current(gbm_models, "gbm_model").importances
            figure = charts.importance_figure(importances)
            figure.update_layout(height=charts.importance_height(len(importances)))
            return _json_safe_figure(figure)

        @render.data_frame
        def gbm_imp_table():
            """Feature importance table of the chosen GBM."""
            return render.DataGrid(_show(_current(gbm_models, "gbm_model").importances), height="320px")

        @render_widget
        def gbm_lift_plot():
            """Lift chart of the chosen GBM for the current filter."""
            model = _current(gbm_models, "gbm_model")
            try:
                figure = charts.lift_chart_figure(d0, model, model.weight, filt()["total"], int(input.gbm_lift_bins()))
            except Exception as e:  # show the reason instead of failing the session
                _notify_error("Lift chart error", e)
                req(False)
            return _json_safe_figure(figure)

        @render_widget
        def gbm_lorenz_plot():
            """Lorenz curve of the chosen GBM for the current filter."""
            model = _current(gbm_models, "gbm_model")
            try:
                figure = charts.lorenz_figure(d0, model, model.weight, filt()["total"])
            except Exception as e:
                _notify_error("Lorenz curve error", e)
                req(False)
            return _json_safe_figure(figure)

        @render.data_frame
        def gbm_lift_table():
            """A/E by bin table of the chosen GBM."""
            model = _current(gbm_models, "gbm_model")
            try:
                table = charts.lift_table_frame(d0, model, model.weight, filt()["total"], int(input.gbm_lift_bins()))
            except Exception as e:
                _notify_error("A/E by bin error", e)
                req(False)
            return render.DataGrid(_show(table), height="340px")

        @render.data_frame
        def gbm_metrics_table():
            """Performance metrics of the chosen GBM for the current filter. Shows empty headings before a GBM is built."""
            if not gbm_models() or not input.gbm_model():
                empty_metrics = pd.DataFrame(
                    columns=["Dataset", "GINI", "PoDE", "RMSE", "Within 5% (%)", "Within 10% (%)", "Within 20% (%)"]
                )
                return render.DataGrid(_show(empty_metrics), height="200px")
            metrics = model_performance_metrics(d0, _current(gbm_models, "gbm_model"), rows_mask=filt()["total"])
            return render.DataGrid(_show(metrics), height="200px")

        @render.ui
        def gbm_metrics_comparison():
            """Metrics comparison of the chosen GBM against a column or an older model, for the chosen split."""
            models = gbm_models()
            selected_new = input.gbm_model()
            if not models or not selected_new:
                return ui.p("Build a GBM model to compare performance metrics.")
            new_model = models[int(selected_new)]
            active_rows = filt()["total"]
            new_metrics = model_performance_metrics(d0, new_model, rows_mask=active_rows)
            compare_source = input.gbm_compare_source()
            if compare_source == "column":
                column = input.gbm_compare_column()
                if not column or column not in d0.columns:
                    return ui.p("Choose a numeric prediction column to compare.")
                old_name = column
                old_metrics = _column_metrics(new_model, column, active_rows)
            else:
                selected_old = input.gbm_compare_model()
                if not selected_old or (compare_source == "gbm" and selected_old == selected_new):
                    if compare_source == "gbm":
                        return ui.p("Choose a different old GBM model to compare.")
                    return ui.p("Choose an old GLM model to compare.")
                old_models = glm_models() if compare_source == "glm" else models
                old_model = old_models[int(selected_old)]
                if old_model.response != new_model.response:
                    return ui.p("Choose a model fitted to the same response.")
                old_name = old_model.name
                old_metrics = model_performance_metrics(d0, old_model, rows_mask=active_rows)
            comparison = compare_gbm_metrics(new_metrics, old_metrics)
            comparison = comparison[comparison["Dataset"] == input.gbm_compare_split()]
            return metrics_comparison_table(comparison, f"New: {new_model.name}", f"Old: {old_name}")

        @reactive.calc
        def gbm_tabs():
            """Ratebook tables, one per feature, for the chosen GBM."""
            spec_version()
            m = _current(gbm_models, "gbm_model")
            return tabulate_model(d0, lambda x: m.predict(x), m.features, feat_df, link=m.link)

        @reactive.effect
        @reactive.event(input.gbm_shap_feature)
        def _gbm_shap_banding():
            """Set the GBM SHAP banding for the chosen feature: from the specification, otherwise a guess from its values."""
            feature = input.gbm_shap_feature()
            if not feature or feature not in d0.columns:
                return
            series = d0[feature]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                banding = numeric_or_none(feature_lookup(feat_df, feature).get("banding"))
                ui.update_numeric(
                    "gbm_shap_banding", value=banding if banding and banding > 0 else banding_guesser(series)
                )
            else:
                ui.update_numeric("gbm_shap_banding", value=0)

        @render.text
        def gbm_shap_note():
            """Note under the GBM SHAP chart: whether SHAP values exist and how they are scaled."""
            models = gbm_models()
            selected = input.gbm_model()
            if not models or not selected:
                return "Build a GBM to view SHAP values."
            model = models[int(selected)]
            if model.SHAP is None:
                return "SHAP values are unavailable. Build a GBM with SHAP values enabled."
            if model.link != "log":
                return "Mean centered SHAP contributions are shown on the model's link scale."
            return "Mean SHAP relativities are centered on the filtered SHAP sample."

        @reactive.calc
        def gbm_shap_summary():
            """Mean SHAP relativity by band for the chosen GBM feature, on the filtered rows."""
            model = _current(gbm_models, "gbm_model")
            feature = input.gbm_shap_feature()
            req(model.SHAP is not None, feature, feature in model.features)
            return charts.mean_shap_summary(
                d0, model, feature, weight(), float(input.gbm_shap_banding() or 0), filt()["total"]
            )

        @render_widget
        def gbm_shap_rel_plot():
            """Mean SHAP chart for the chosen GBM feature."""
            model = _current(gbm_models, "gbm_model")
            summary = gbm_shap_summary()
            req(not summary.empty)
            return _json_safe_figure(
                charts.mean_shap_figure(
                    summary,
                    input.gbm_shap_feature(),
                    model.link,
                    x_title=feature_label(feat_df, input.gbm_shap_feature()),
                )
            )

        @render.data_frame
        def gbm_shap_rel_table():
            """Table behind the GBM mean SHAP chart."""
            return render.DataGrid(_show(_label_weight_column(gbm_shap_summary(), weight())), height="300px")

        @render_widget
        def gbm_rb_plot():
            """Ratebook chart of relativities for the chosen GBM feature."""
            t = gbm_tabs()
            f = input.gbm_rb_feature()
            req(f, f in t)
            return _json_safe_figure(charts.ratebook_figure(t[f], f))

        @render.data_frame
        def gbm_rb_table():
            """Ratebook table of relativities for the chosen GBM feature."""
            t = gbm_tabs()
            f = input.gbm_rb_feature()
            req(f, f in t)
            return render.DataGrid(_show(t[f]), height="300px")

        @render.download_button(filename="gbm_ratebook.xlsx")
        def gbm_download():
            """Download the chosen GBM's ratebook as an Excel workbook."""
            try:
                check_ratebook_link(_current(gbm_models, "gbm_model").link)
            except ValueError as e:
                _notify_error("Ratebook", e)
                req(False)
            t = gbm_tabs()
            with tempfile.TemporaryDirectory() as tmp:
                gbm = _current(gbm_models, "gbm_model")
                # a GBM built with its own offset column is written without the weight scaling
                path = write_ratebook_excel(
                    t,
                    str(Path(tmp) / "rb.xlsx"),
                    gbm.name,
                    weight_basis=gbm.weight_basis if gbm.init_score == "no offset" else None,
                    weight=gbm.weight,
                )
                yield Path(path).read_bytes()

        @render.download_button(filename="daisy_gbm_bundle.zip")
        def gbm_bundle_download():
            """Download the chosen GBM as a zip: the model (pickle and LightGBM text), its metadata and settings, and its predictions."""
            model = _current(gbm_models, "gbm_model")
            feature_spec = model.feature_table
            if feature_spec is not None:
                feature_spec = feature_spec[feature_spec["feature"].isin(model.features)]
                feature_spec = feature_spec.astype(object).where(pd.notna(feature_spec), None).to_dict(orient="records")
            monotonicity = {}
            if feature_spec:
                monotonicity = {
                    str(item["feature"]): str(item["monotonicity"])
                    for item in feature_spec
                    if item.get("monotonicity") is not None
                    and str(item["monotonicity"]).strip()
                    and str(item["monotonicity"]).lower() != "nan"
                }
            metadata = {
                "app": "Daisy",
                "model_name": model.name,
                "response": model.response,
                "weight": model.weight,
                "init_score": model.init_score,
                "link": model.link,
                "weight_basis": model.weight_basis,
                # added to every raw score of model.txt (with the offset) before the link is inverted
                "offset_base": getattr(model, "offset_base", 0.0),
                "features": model.features,
                "categorical_levels": model.categories,
                # minimum / maximum caps and bins applied to the features before model.txt sees them
                "feature_treatments": getattr(model, "treatments", {}),
                "hyperparameters": model.params,
                "training_config": model.training_config,
                "monotonicity": monotonicity,
                "interaction_groups": model.interaction_groups,
                "interaction_constraints": model.params.get("interaction_constraints"),
                "feature_specification": feature_spec,
                "best_iteration": model.best_iteration,
                "run_time_seconds": model.run_time,
                "evaluation_log": model.evaluation_log,
            }
            predictions = score_gbm(model, d0)
            prediction_csv = _prediction_export_frame(d0, model.response, predictions).to_csv(index=False)
            archive_data = io.BytesIO()
            with zipfile.ZipFile(archive_data, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("model.pkl", pickle.dumps(model))
                archive.writestr(
                    "model.txt",
                    model.booster.model_to_string(num_iteration=model.best_iteration),
                )
                archive.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
                archive.writestr("predictions.csv", prediction_csv)
            yield archive_data.getvalue()

        # ---------------- Overlays: band relativities layered on a base model (adjustment overlays)
        def overlay_models() -> dict:
            """Choices for the Overlays base model: GLMs, GBMs and uploaded model files."""
            choices = {f"glm:{i}": m.name for i, m in enumerate(glm_models())}
            choices.update({f"gbm:{i}": m.name for i, m in enumerate(gbm_models())})
            choices.update({f"file:{i}": f"{item['name']} (file)" for i, item in enumerate(overlay_uploads())})
            return choices

        def model_by_key(key: str):
            """The model for a drop-down key (family:index), as a (family, model) pair."""
            family, index = key.split(":", 1)
            if family == "file":
                item = overlay_uploads()[int(index)]
                return item["family"], item["model"]
            models = glm_models() if family == "glm" else gbm_models()
            return family, models[int(index)]

        def response_for(
            family: str,
            model,
        ) -> str:
            """The response of a base model; a ratebook has none, so the app's response is used."""
            return response() if family == "ratebook" else model.response

        def expected_for(
            family: str,
            model,
            frame: pd.DataFrame,
        ) -> np.ndarray:
            """Expected (predicted) values of a model for a frame, scored with the model's own method."""
            if family == "ratebook":
                return score_ratebook(model, frame, model.weight)[0]
            return model.score(frame) if family == "glm" else score_gbm(model, frame)

        @reactive.effect
        @reactive.event(input.overlay_model_file, ignore_init=True)
        def _overlay_load_file():
            """Read a model file uploaded on the Overlays tab: a ratebook (.xlsx) or a GBM pickle."""
            files = input.overlay_model_file()
            if not files:
                return
            f = files[0]
            try:
                if f["name"].lower().endswith(".xlsx"):
                    ratebook = read_ratebook(f["datapath"])
                    item = {"family": "ratebook", "model": ratebook, "name": ratebook.model_name or f["name"]}
                else:
                    gbm = read_gbm_pickle(f["datapath"])
                    item = {"family": "gbm", "model": gbm, "name": gbm.name}
            except Exception as e:
                ui.notification_show(f"Could not read the model file: {e}", type="error", duration=10)
                return
            overlay_uploads.set(overlay_uploads() + [item])
            ui.notification_show(f"Read in '{item['name']}' as a base model for the overlays.", type="message")

        def overlay_frames():
            """(training rows, test rows). Without a train_test column the training rows are all rows.

            An out-of-sample test file, when loaded, is the whole test set (every row is out of sample)."""
            if has_tt:
                train = d0[d0["train_test"] == 0].reset_index(drop=True)
                test = d0[d0["train_test"] == 1].reset_index(drop=True)
            else:
                train, test = d0.reset_index(drop=True), d0.iloc[0:0].reset_index(drop=True)
            if overlay_data() is not None:
                # Unseen data, when loaded, is used first: its test rows (train_test = 1) are the test set and
                # the rest are the rows the suggestions are calculated on.
                unseen = overlay_data().reset_index(drop=True)
                if "train_test" in unseen.columns:
                    unseen_test = unseen[unseen["train_test"] == 1].reset_index(drop=True)
                    train = unseen[unseen["train_test"] != 1].reset_index(drop=True)
                    if not unseen_test.empty:
                        test = unseen_test
                else:
                    train = unseen
            if overlay_test_data() is not None:
                test = overlay_test_data().reset_index(drop=True)
            return train, test

        @reactive.effect
        @reactive.event(input.overlay_browse, ignore_init=True)
        async def _overlay_browse_data():
            """Pick the in-sample CSV with a file dialog and fill in its path."""
            await _browse_csv_into("overlay_path", "Choose the in-sample CSV")

        @reactive.effect
        @reactive.event(input.overlay_test_browse, ignore_init=True)
        async def _overlay_browse_test_data():
            """Pick the out-of-sample test CSV with a file dialog and fill in its path."""
            await _browse_csv_into("overlay_test_path", "Choose the out-of-sample test CSV")

        @reactive.effect
        @reactive.event(input.overlay_test_load, ignore_init=True)
        async def _overlay_load_test_data():
            """Load the out-of-sample test CSV named on the Overlays tab."""
            path = (input.overlay_test_path() or "").strip()
            if not path:
                ui.notification_show("Enter the path to an out-of-sample test CSV file first.", type="warning")
                return
            try:
                data = await read_csv_path(path)
            except Exception as e:
                _notify_error("Could not load the test data", e)
                return
            overlay_test_data.set(data)
            overlay_check_state.set(None)
            ui.notification_show(f"Loaded {len(data):,} out-of-sample test rows for the overlays.", type="message")

        @render.text
        def overlay_test_status():
            """Status line for the out-of-sample test data."""
            data = overlay_test_data()
            if data is None:
                return "No out-of-sample test data loaded: the overlays are tested on the test rows above."
            return f"Out-of-sample test data: {len(data):,} rows x {data.shape[1]} columns (the overlays are tested on these rows)."

        @reactive.effect
        @reactive.event(input.overlay_load, ignore_init=True)
        async def _overlay_load_data():
            """Load the unseen-data CSV named on the Overlays tab, used for the suggestions."""
            path = (input.overlay_path() or "").strip()
            if not path:
                ui.notification_show("Enter the path to a CSV file first.", type="warning")
                return
            try:
                data = await read_csv_path(path)
            except Exception as e:
                _notify_error("Could not load the data", e)
                return
            overlay_data.set(data)
            overlay_candidate.set(None)
            overlay_check_state.set(None)
            ui.notification_show(f"Loaded {len(data):,} unseen rows for the overlays.", type="message")

        @render.text
        def overlay_data_status():
            """Status line for the unseen data used for the suggestions."""
            data = overlay_data()
            if data is None:
                return "No in-sample data loaded: the suggestions use the training rows."
            return f"In-sample data: {len(data):,} rows x {data.shape[1]} columns (the suggestions use these rows)."

        @reactive.effect
        def _overlay_model_choices():
            """Set the Overlays base-model drop-down to the available models."""
            choices = overlay_models()
            with reactive.isolate():
                current = input.overlay_model()
            ui.update_select(
                "overlay_model", choices=choices, selected=current if current in choices else next(iter(choices), None)
            )

        def overlay_is_categorical(feature: str) -> bool:
            """Categorical features (text, category, boolean) use one band per level, not numeric bands."""
            return not (pd.api.types.is_numeric_dtype(d0[feature]) and not pd.api.types.is_bool_dtype(d0[feature]))

        def model_features(
            family: str,
            model,
        ) -> list:
            """The features of a base model that exist in the data."""
            features = list(model.features(d0.columns)) if family == "glm" else list(model.features)
            return [f for f in features if f in d0.columns]

        def feature_banding(feature: str) -> float:
            """The banding a numeric feature starts with: its specification banding, or a guessed one."""
            spec_banding = numeric_or_none(feature_lookup(feat_df, feature).get("banding"))
            return spec_banding if spec_banding else banding_guesser(d0[feature])

        def overlay_feature_groups(
            family: str,
            model,
        ):
            """(modelled features, non-modelled features): the base model's features, then the other columns of the
            data (without the response, weight and train_test)."""
            modelled = _ranked_features(d0, model, family, model_features(family, model))  # most important first
            excluded = {response(), weight(), *NOT_NON_MODELLED}
            others = [c for c in d0.columns if c not in set(modelled) and c not in excluded]
            return modelled, others

        @reactive.effect
        def _overlay_feature_choices():
            """The features of the chosen base model in two groups; a new base model clears the suggestions of the old."""
            key = input.overlay_model()
            if not key or not overlay_models():
                return
            family, model = model_by_key(key)
            modelled, others = overlay_feature_groups(family, model)
            choices = _chart_feature_choices(
                list(d0.columns), modelled, excluded={response(), weight(), *NOT_NON_MODELLED}
            )
            flat = modelled + others
            with reactive.isolate():
                current = input.overlay_feature_plot()
            overlay_candidates.set({})
            overlay_candidate.set(None)
            overlay_check_state.set(None)
            ui.update_select(
                "overlay_feature_plot",
                choices=choices,
                selected=current if current in flat else (flat[0] if flat else None),
            )

        @reactive.effect
        @reactive.event(input.overlay_feature_plot, ignore_init=True)
        def _overlay_show_feature():
            """Show the suggestions of the chosen feature in the plot, grid and save."""
            overlay_candidate.set(overlay_candidates().get(input.overlay_feature_plot()))
            overlay_check_state.set(None)

        @reactive.effect
        @reactive.event(input.overlay_calc, ignore_init=True)
        async def _overlay_calculate():
            """Calculate suggested band relativities for every feature of the base model from the training rows, with bootstrap ranges."""
            key = input.overlay_model()
            if not key:
                ui.notification_show("Choose a base model first.", type="warning")
                return
            family, model = model_by_key(key)
            train, _ = overlay_frames()
            if train.empty:
                ui.notification_show("There are no training rows to calculate the suggestions from.", type="warning")
                return
            modelled, others = overlay_feature_groups(family, model)
            features = modelled + others
            if not features:
                ui.notification_show("The base model has no features in this data.", type="warning")
                return
            response_name = response_for(family, model)
            try:
                expected = expected_for(family, model, train)
                actual = train[response_name].to_numpy(dtype=float)
            except Exception as e:
                _notify_error("Could not score the data with the base model", e)
                return
            samples = int(input.overlay_samples() or 500)
            source = "unseen" if overlay_data() is not None else "training"
            candidates, skipped = {}, []
            with ui.Progress(min=0, max=len(features)) as progress:
                for number, feature in enumerate(features, start=1):
                    progress.set(
                        number - 1, message="Calculating suggestions", detail=f"{feature} ({number} of {len(features)})"
                    )
                    await asyncio.sleep(0)  # lets the progress bar draw before this feature's bootstrap runs
                    try:
                        categorical = overlay_is_categorical(feature)
                        banding = 0.0 if categorical else feature_banding(feature)
                        levels = category_levels(train[feature]) if categorical else None
                        minimum, maximum = (None, None) if categorical else spec_chart_limits(feat_df, feature)
                        keys = band_starts(train[feature], banding, levels, minimum, maximum)
                        suggestion = suggest(
                            band_table(actual, expected, keys),
                            bootstrap_intervals(actual, expected, keys, samples=samples),
                        )
                    except Exception:
                        skipped.append(feature)
                        progress.set(number, message="Calculating suggestions", detail=f"{feature} skipped")
                        continue
                    candidates[feature] = {
                        "family": family,
                        "model": model,
                        "feature": feature,
                        "banding": banding,
                        "minimum": minimum,
                        "maximum": maximum,
                        "levels": levels,
                        "suggestion": suggestion,
                        "training_rows": len(train),
                        "response": response_name,
                        "source": source,
                    }
                    progress.set(number, message="Calculating suggestions", detail=f"{feature} done")
                    await asyncio.sleep(0)
            if not candidates:
                ui.notification_show("No feature could be calculated for this base model.", type="error")
                return
            overlay_candidates.set(candidates)
            selected = input.overlay_feature_plot()
            if selected not in candidates:
                selected = next(iter(candidates))
            ui.update_select(
                "overlay_feature_plot",
                choices=_chart_feature_choices(
                    list(d0.columns), modelled, excluded={response(), weight(), *NOT_NON_MODELLED}
                ),
                selected=selected,
            )
            overlay_candidate.set(candidates[selected])
            overlay_check_state.set(None)
            message = f"Calculated suggestions for {len(candidates)} features."
            if skipped:
                message += f" Skipped: {', '.join(skipped)}."
            ui.notification_show(message, type="message", duration=10)

        @reactive.calc
        def overlay_rows():
            """The band table shown in the grid (None before the suggestions are calculated)."""
            cand = overlay_candidate()
            if cand is None:
                return None
            s = cand["suggestion"]
            return pd.DataFrame(
                {
                    "Band": [band_label(start, cand["banding"], cand["levels"]) for start in s.index],
                    "Start": s.index.to_numpy(dtype=float),
                    "Rows": s["n"].to_numpy(),
                    "Actual": s["A"].round(2).to_numpy(),
                    "Expected": s["E"].round(2).to_numpy(),
                    "A/E": s["AE"].round(4).to_numpy(),
                    "99% range": [f"{lo:.3f} to {hi:.3f}" for lo, hi in zip(s["lo99"], s["hi99"])],
                    "95% range": [f"{lo:.3f} to {hi:.3f}" for lo, hi in zip(s["lo95"], s["hi95"])],
                    "90% range": [f"{lo:.3f} to {hi:.3f}" for lo, hi in zip(s["lo90"], s["hi90"])],
                    "Status": s["status"].to_numpy(),
                    "Relativity": s["relativity"]
                    .round(4)
                    .astype(object)
                    .to_numpy(),  # object: typed edits (text) must fit
                }
            )

        @render.ui
        def overlay_note():
            """Note under the Overlays grid: the chosen feature's base model, banding and source rows, with the rule used."""
            cand = overlay_candidate()
            if cand is None:
                return ui.p(
                    "Choose a base model, then press Calculate suggestions to calculate every feature. The suggestions use the "
                    "in-sample data if loaded, otherwise the training rows. " + OVERLAY_RULE,
                    class_="text-muted small",
                )
            if cand["levels"] is not None:
                desc = f"categorical, {len(cand['levels'])} levels"
            else:
                desc = f"banding {cand['banding']:g}"
            return ui.p(
                f"{cand['model'].name}, feature {cand['feature']}, {desc}, from "
                f"{cand['training_rows']:,} {cand['source']} rows. {OVERLAY_RULE} Edit the Relativity column to "
                "change any band; bands you don't change keep the suggested value.",
                class_="text-muted small",
            )

        def ae_figure_for(item):
            """The A/E-by-band figures of one saved overlay on the training rows: before, then after its relativities."""
            overlay, family, model = item["overlay"], item["family"], item["model"]
            train, _ = overlay_frames()
            expected = expected_for(family, model, train)
            actual = train[item["response"]].to_numpy(dtype=float)
            keys = overlay.band_keys(train[overlay.feature])
            table = band_table(actual, expected, keys).join(
                bootstrap_intervals(actual, expected, keys, samples=500), how="left"
            )
            rel = np.array(
                [
                    1.0 if overlay.relativities.get(float(start)) is None else float(overlay.relativities[start])
                    for start in table.index
                ],
                dtype=float,
            )
            labels = [band_label(start, overlay.banding, overlay.levels or None) for start in table.index]
            return [
                build_range_figure(table, labels, overlay.feature, overlay.base_name),
                build_ae_figure(table, rel, labels, overlay.feature),
            ]

        def build_ae_figure(
            s,
            rel,
            labels,
            feature: str,
        ):
            """A/E by band before and after the relativities (rel), with the shaded 99%, 95% and 90% ranges of the
            adjusted A/E (the base ranges divided by the relativity) and the share of rows per band."""
            actual, expected = s["A"].to_numpy(dtype=float), s["E"].to_numpy(dtype=float)
            ae_base = actual / expected
            ae_adjusted = actual / (expected * rel)
            go_ = charts._go()
            fig = go_.Figure()
            # shaded 99%, 95% and 90% ranges of the adjusted A/E (the base ranges divided by the relativity)
            for tag, colour in (
                ("99", "rgba(148,103,189,0.16)"),
                ("95", "rgba(255,127,14,0.24)"),
                ("90", "rgba(44,160,44,0.34)"),
            ):
                lo = s[f"lo{tag}"].to_numpy(dtype=float) / rel
                hi = s[f"hi{tag}"].to_numpy(dtype=float) / rel
                fig.add_trace(
                    go_.Scatter(
                        x=labels + labels[::-1],
                        y=list(hi) + list(lo[::-1]),
                        fill="toself",
                        fillcolor=colour,
                        line=dict(width=0),
                        name=f"{tag}% range",
                        hoverinfo="skip",
                    )
                )
            share = s["n"].to_numpy(dtype=float) / max(float(s["n"].sum()), 1.0)
            fig.add_trace(
                go_.Bar(
                    x=labels,
                    y=share,
                    name="Prop. of weight",
                    yaxis="y2",
                    marker_color="rgba(31, 119, 180, 0.4)",
                    hovertemplate="%{y:.1%} of rows<extra>Prop. of weight</extra>",
                )
            )
            fig.add_trace(
                go_.Scatter(
                    x=labels,
                    y=ae_base,
                    mode="lines+markers",
                    name="A/E base",
                    line=dict(color=charts.PALETTE[1], dash="dash"),
                    marker=dict(size=7, color=charts.PALETTE[1]),
                )
            )
            fig.add_trace(
                go_.Scatter(
                    x=labels,
                    y=ae_adjusted,
                    mode="lines+markers",
                    name="A/E with relativities",
                    line=dict(color=charts.PALETTE[0]),
                    marker=dict(size=9, color=charts.PALETTE[0], line=dict(color="white", width=1)),
                    customdata=np.column_stack([rel, s["n"].to_numpy()]),
                    hovertemplate=(
                        "%{x}<br>A/E with relativities %{y:.4f}<br>relativity "
                        "%{customdata[0]:.4f}<br>rows %{customdata[1]:,.0f}<extra></extra>"
                    ),
                )
            )
            fig.add_hline(y=1, line_dash="dot", line_color="grey")
            fig.update_layout(
                title=f"A/E by band with relativities applied: {feature}",
                template="plotly_white",
                xaxis=dict(title=feature, type="category"),
                yaxis=dict(title="A/E"),
                yaxis2=dict(
                    title="Prop. of weight",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    tickformat=".0%",
                    rangemode="tozero",
                ),
                legend=dict(orientation="h", x=1.0, xanchor="right", y=1.02, yanchor="bottom"),
                margin=dict(l=50, r=50, t=90, b=50),
            )
            return fig

        @render_widget
        def overlay_adjusted_plot():
            """A/E by band before and after the relativities in the grid (the training rows of the suggestions)."""
            cand = overlay_candidate()
            req(cand is not None)
            s = cand["suggestion"]
            rel_by_start = edited_relativities()
            rel = np.array([rel_by_start.get(float(start)) or 1.0 for start in s.index], dtype=float)
            labels = [band_label(start, cand["banding"], cand["levels"]) for start in s.index]
            return _json_safe_figure(build_ae_figure(s, rel, labels, cand["feature"]))

        def build_range_figure(
            s,
            labels,
            feature: str,
            model_name: str,
        ):
            """A/E by band with its 99% (thin), 95% and 90% (thicker) ranges, as on the Overlays tab."""
            go_ = charts._go()
            ae = s["AE"].to_numpy(dtype=float)
            fig = go_.Figure()
            # shaded areas for the 99%, 95% and 90% ranges (the 90% area is the darkest, inside the others)
            for tag, colour in (
                ("99", "rgba(148,103,189,0.16)"),
                ("95", "rgba(255,127,14,0.24)"),
                ("90", "rgba(44,160,44,0.34)"),
            ):
                lo, hi = s[f"lo{tag}"].to_numpy(dtype=float), s[f"hi{tag}"].to_numpy(dtype=float)
                fig.add_trace(
                    go_.Scatter(
                        x=labels + labels[::-1],
                        y=list(hi) + list(lo[::-1]),
                        fill="toself",
                        fillcolor=colour,
                        line=dict(width=0),
                        name=f"{tag}% range",
                        hoverinfo="skip",
                    )
                )
            share = s["n"].to_numpy(dtype=float) / max(float(s["n"].sum()), 1.0)
            fig.add_trace(
                go_.Bar(
                    x=labels,
                    y=share,
                    name="Prop. of weight",
                    yaxis="y2",
                    marker_color="rgba(31, 119, 180, 0.4)",
                    hovertemplate="%{y:.1%} of rows<extra>Prop. of weight</extra>",
                )
            )
            fig.add_trace(
                go_.Scatter(
                    x=labels,
                    y=ae,
                    mode="lines+markers",
                    name="A/E",
                    line=dict(color=charts.PALETTE[0]),
                    marker=dict(size=9, color=charts.PALETTE[0], line=dict(color="white", width=1)),
                    customdata=np.column_stack(
                        [s["n"].to_numpy(), s["lo99"], s["hi99"], s["lo95"], s["hi95"], s["lo90"], s["hi90"]]
                    ),
                    hovertemplate=(
                        "%{x}<br>A/E %{y:.4f}<br>rows %{customdata[0]:,.0f}<br>"
                        "99%: %{customdata[1]:.3f} to %{customdata[2]:.3f}<br>"
                        "95%: %{customdata[3]:.3f} to %{customdata[4]:.3f}<br>"
                        "90%: %{customdata[5]:.3f} to %{customdata[6]:.3f}<extra></extra>"
                    ),
                )
            )
            fig.add_hline(y=1, line_dash="dot", line_color="grey")
            fig.update_layout(
                title=f"A/E by band, training rows: {model_name}, {feature}",
                template="plotly_white",
                xaxis=dict(title=feature, type="category"),
                yaxis=dict(title="A/E"),
                yaxis2=dict(
                    title="Prop. of weight",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    tickformat=".0%",
                    rangemode="tozero",
                ),
                legend=dict(orientation="h", x=1.0, xanchor="right", y=1.02, yanchor="bottom"),
                margin=dict(l=50, r=50, t=90, b=50),
            )
            return fig

        @render_widget
        def overlay_range_plot():
            """A/E by band with its 99% (thin), 95% and 90% (thicker) ranges, coloured by the rule's status."""
            cand = overlay_candidate()
            req(cand is not None)
            s = cand["suggestion"]
            labels = [band_label(start, cand["banding"], cand["levels"]) for start in s.index]
            return _json_safe_figure(build_range_figure(s, labels, cand["feature"], cand["model"].name))

        @render.data_frame
        def overlay_grid():
            """Editable grid of the band relativities for the chosen feature."""
            rows = overlay_rows()
            req(rows is not None)
            return render.DataGrid(rows, editable=True, height="420px")

        def edited_relativities() -> dict:
            """The relativity of each band (keyed by band start) as it is in the grid, including the edits."""
            rows = overlay_rows()
            try:
                edited = overlay_grid.data_patched()
            except (SilentException, SilentCancelOutputException):
                edited = rows
            values = pd.to_numeric(edited["Relativity"], errors="coerce")
            return {
                float(start): (float(value) if np.isfinite(value) else None)
                for start, value in zip(edited["Start"], values)
            }

        @reactive.effect
        @reactive.event(input.overlay_check, ignore_init=True)
        def _overlay_check():
            """Check the edited relativities: warn about values outside the 99% range and 'Adjust' bands left at 1, and show the balanced A/E."""
            cand = overlay_candidate()
            if cand is None:
                ui.notification_show("Calculate the suggestions first.", type="warning")
                return
            relativities = edited_relativities()
            s = cand["suggestion"]
            rel = np.array([relativities.get(float(start)) or 1.0 for start in s.index])
            adjusted_expected = s["E"].to_numpy() * rel
            warnings = check_relativities(relativities)
            for start, r, lo, hi, status in zip(s.index, rel, s["lo99"], s["hi99"], s["status"]):
                label = band_label(start, cand["banding"], cand["levels"])
                if r != 1.0 and (r < lo or r > hi):
                    warnings.append(f"{label}: relativity {r:.3f} is outside its 99% range ({lo:.3f} to {hi:.3f}).")
                if status == "Adjust" and r == 1.0:
                    warnings.append(f"{label}: the rule says adjust, but the relativity is 1.")
            overlay_check_state.set({"total": float(s["A"].sum() / adjusted_expected.sum()), "warnings": warnings})

        @render.ui
        def overlay_check_note():
            """Results of the overlay check: the training A/E after the overlay, and any warnings."""
            state = overlay_check_state()
            if state is None:
                return None
            items = [
                ui.p(
                    f"After the overlay, the training A/E is {state['total']:.4f} (1.0000 is balanced).",
                    class_="small mb-1",
                )
            ]
            items += [ui.p(w, class_="text-warning small mb-0") for w in state["warnings"]]
            return ui.div(*items)

        @reactive.effect
        @reactive.event(input.overlay_save, ignore_init=True)
        def _overlay_save():
            """Save the edited overlay under a unique name, if its relativities pass the checks."""
            cand = overlay_candidate()
            if cand is None:
                ui.notification_show("Calculate the suggestions first.", type="warning")
                return
            relativities = edited_relativities()
            problems = [p for p in check_relativities(relativities)]
            if problems:
                ui.notification_show("Not saved: " + "; ".join(problems), type="warning", duration=10)
                return
            base_name = (input.overlay_name() or "").strip() or f"{cand['feature']} overlay"
            taken = {i["overlay"].name for i in overlays()}
            name, k = base_name, 2
            while name in taken:  # names identify overlays in the lists, so keep them unique
                name, k = f"{base_name} ({k})", k + 1
            overlay = Overlay(
                name,
                cand["model"].name,
                cand["feature"],
                cand["banding"],
                {start: value for start, value in relativities.items()},
                cand["training_rows"],
                levels=cand["levels"] or [],
                minimum=cand.get("minimum"),
                maximum=cand.get("maximum"),
            )
            overlays.set(
                overlays()
                + [{"overlay": overlay, "family": cand["family"], "model": cand["model"], "response": cand["response"]}]
            )
            ui.notification_show(f"Saved the overlay '{name}'.", type="message")

        @reactive.calc
        def overlay_table_rows():
            """Table of the saved overlays: name, base model, feature, banding, changed bands and training rows."""
            rows = []
            for item in overlays():
                o = item["overlay"]
                rows.append(
                    {
                        "Overlay": o.name,
                        "Base model": o.base_name,
                        "Feature": o.feature,
                        "Banding": "categorical" if o.levels else o.banding,
                        "Bands changed": sum(
                            1 for v in o.relativities.values() if v is not None and abs(v - 1) > 1e-12
                        ),
                        "Training rows": o.training_rows,
                    }
                )
            return pd.DataFrame(
                rows, columns=["Overlay", "Base model", "Feature", "Banding", "Bands changed", "Training rows"]
            )

        @render.data_frame
        def overlay_list():
            """Grid listing the saved overlays."""
            return render.DataGrid(overlay_table_rows(), height="220px")

        @render.download_button(filename="daisy_overlays.xlsx")
        def overlay_export():
            """Download the saved overlays as an Excel workbook."""
            items = overlays()
            if not items:
                ui.notification_show("Save an overlay first.", type="warning", duration=8)
                req(False)
            yield adjustments_workbook([(item["overlay"], item["response"]) for item in items])

        @reactive.effect
        def _overlay_selectors():
            """Set the overlay remove and evaluate drop-downs to the saved overlay names."""
            names = [item["overlay"].name for item in overlays()]
            with reactive.isolate():
                remove_current, eval_current = input.overlay_remove(), input.overlay_eval()
            ui.update_select(
                "overlay_remove",
                choices=names,
                selected=remove_current if remove_current in names else (names[-1] if names else None),
            )
            ui.update_select(
                "overlay_eval",
                choices=names,
                selected=eval_current if eval_current in names else (names[-1] if names else None),
            )

        @reactive.effect
        @reactive.event(input.overlay_delete, ignore_init=True)
        def _overlay_delete():
            """Remove the chosen overlay from the saved list."""
            name = input.overlay_remove()
            overlays.set([item for item in overlays() if item["overlay"].name != name])

        def overlay_eval_for(item):
            """(band table, summary, message) of one saved overlay on the test rows."""
            _, test = overlay_frames()
            if test.empty:
                return None, None, "This data has no test rows (train_test = 1), so the overlay cannot be tested."
            overlay, model = item["overlay"], item["model"]
            missing = [c for c in [item["response"], overlay.feature] if c not in test.columns]
            if missing:
                return None, None, f"The test data has no column {', '.join(missing)}, so the overlay cannot be tested."
            try:
                expected = expected_for(item["family"], model, test)
            except Exception as e:
                return None, None, f"Could not score the test data with the base model: {e}"
            actual = test[item["response"]].to_numpy(dtype=float)
            table, summary = evaluate_overlay(overlay, test, actual, expected)
            # standard error of each band's actual per row (its mean), for the error bars on the plot
            frame = pd.DataFrame({"band": overlay.band_keys(test[overlay.feature]), "y": actual}).dropna()
            grouped = frame.groupby("band")["y"]
            table["SE"] = (grouped.std(ddof=1) / np.sqrt(grouped.size())).reindex(table.index).to_numpy()
            table.index = [band_label(start, overlay.banding, overlay.levels or None) for start in table.index]
            return table, summary, None

        def overlay_eval_result():
            """(band table, summary, message) of the chosen overlay on the test rows."""
            name = input.overlay_eval()
            item = next((i for i in overlays() if i["overlay"].name == name), None)
            if item is None:
                return None, None, "Save an overlay to test it on the test rows."
            return overlay_eval_for(item)

        def overlays_on_base(model) -> list:
            """The saved overlays built on this base model, in the order they were saved."""
            return [i for i in overlays() if i["model"] is model]

        def overlay_metric_for(item):
            """(comparison of the test-row metrics of the base model with all its saved overlays applied together;
            with overlays as New, base as Old; message)."""
            _, test = overlay_frames()
            if test.empty:
                return None, "This data has no test rows (train_test = 1), so the metrics cannot be compared."
            model, family = item["model"], item["family"]
            group = overlays_on_base(model)  # every saved overlay on this base model, applied together
            test = test.reset_index(drop=True)
            missing = [c for c in [item["response"], model.weight] if c not in test.columns and c != "N"]
            if missing:
                return None, f"The test data has no column {', '.join(missing)}, so the metrics cannot be compared."
            try:
                base = expected_for(family, model, test)
            except Exception as e:
                return None, f"Could not score the test data with the base model: {e}"
            adjusted = apply_overlays([i["overlay"] for i in group], test, base)
            if family == "ratebook":  # ratebook predictions are per unit of weight; the objective is chosen by the user
                objective = input.overlay_ratebook_objective() or "poisson"
                power = float(input.overlay_ratebook_power() or 1.5)
                basis = "average"
            else:
                objective, power = model_objective_power(model)
                basis = model_weight_basis(model)
            kwargs = dict(
                weight=model.weight,
                objective=objective,
                tweedie_power=power,
                dataset_label="Test rows",
                weight_basis=basis,
            )
            base_metrics = prediction_performance_metrics(test, item["response"], base, **kwargs)
            adjusted_metrics = prediction_performance_metrics(test, item["response"], adjusted, **kwargs)
            return compare_gbm_metrics(adjusted_metrics, base_metrics), None

        @reactive.calc
        def overlay_metric_result():
            """(comparison of the chosen overlay's base model with all its saved overlays; message)."""
            name = input.overlay_eval()
            item = next((i for i in overlays() if i["overlay"].name == name), None)
            if item is None:
                return None, "Save an overlay to compare its metrics on the test rows."
            return overlay_metric_for(item)

        @render.text
        def overlay_metric_note():
            """Note above the overlay metrics table: how the change columns are read."""
            comparison, message = overlay_metric_result()
            if message:
                return message
            return (
                "Change is the with-overlay value minus the base value; GINI and RMSE changes are in %. "
                "Higher is better for GINI, PoDE and Within; lower is better for RMSE."
            )

        @render.ui
        def overlay_metric_table():
            """Metrics of the chosen overlay's base model against the base model plus its overlays, on the test rows."""
            comparison, message = overlay_metric_result()
            req(comparison is not None, message is None)
            item = next(i for i in overlays() if i["overlay"].name == input.overlay_eval())
            kind = "GBM" if item["family"] == "gbm" else "GLM"
            return metrics_comparison_table(comparison, f"{kind} + Overlay", kind)

        @render.text
        def overlay_eval_note():
            """Summary of the chosen overlay on the test rows: A/E and RMSE before and after the overlay."""
            _, summary, message = overlay_eval_result()
            if message:
                return message
            return (
                f"Test rows ({summary['rows']:,}): A/E total {summary['A/E total base']:.4f} base, "
                f"{summary['A/E total adjusted']:.4f} adjusted. RMSE {summary['RMSE base']:,.2f} base, "
                f"{summary['RMSE adjusted']:,.2f} adjusted."
            )

        def build_test_figure(
            item,
            table,
            errbars: bool = True,
        ):
            """The test-rows plot of one saved overlay (base prediction, adjusted prediction and actual per row)."""
            go_ = charts._go()
            n = table["n"].replace(0, np.nan)
            fig = go_.Figure()
            fig.add_trace(
                go_.Scatter(
                    x=table.index,
                    y=table["A"] / n,
                    name="Actual (per row)",
                    mode="lines+markers",
                    line=dict(color=charts.PALETTE[0]),
                    **(
                        {
                            "error_y": dict(
                                type="data", array=(2 * table["SE"]).to_numpy(dtype=float), visible=True, thickness=1
                            )
                        }
                        if errbars
                        else {}
                    ),
                )
            )
            fig.add_trace(
                go_.Scatter(
                    x=table.index,
                    y=table["E base"] / n,
                    name="Base prediction",
                    mode="lines+markers",
                    line=dict(color=charts.PALETTE[1]),
                )
            )
            fig.add_trace(
                go_.Scatter(
                    x=table.index,
                    y=table["E adjusted"] / n,
                    name="Base + overlay",
                    mode="lines+markers",
                    line=dict(color=charts.PALETTE[2], dash="dash"),
                )
            )
            share = table["n"].to_numpy(dtype=float) / max(float(table["n"].sum()), 1.0)
            share_label = f"Prop. of {weight()}"
            fig.add_trace(
                go_.Bar(
                    x=table.index,
                    y=share,
                    name=share_label,
                    yaxis="y2",
                    marker_color="rgba(31, 119, 180, 0.4)",
                    hovertemplate="%{y:.1%}<extra>" + share_label + "</extra>",
                )
            )
            fig.update_layout(
                title="Test rows by band: actual, base and adjusted",
                template="plotly_white",
                yaxis=dict(title=_kpi_axis_title(kpi_df, input.kpi(), response(), weight())),
                yaxis2=dict(
                    title=share_label,
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    tickformat=".1%",
                    rangemode="tozero",
                ),
                xaxis=dict(title=feature_label(feat_df, item["overlay"].feature)),
                legend=dict(orientation="h", x=1.0, xanchor="right", y=1.02, yanchor="bottom"),
                margin=dict(l=50, r=50, t=90, b=50),
            )
            return fig

        def overlay_test_figure():
            """The test-rows plot of the chosen overlay."""
            table, _, message = overlay_eval_result()
            req(table is not None, message is None)
            item = next(i for i in overlays() if i["overlay"].name == input.overlay_eval())
            return build_test_figure(item, table, errbars=bool(input.overlay_errbars()))

        @render_widget
        def overlay_eval_plot():
            """Test-rows plot of the chosen overlay."""
            return _json_safe_figure(overlay_test_figure())

        @render.download_button(filename="daisy_overlay_report - Overlays.html")
        def overlay_report_download():
            """Download an HTML report of every saved overlay, tested on the test rows."""
            items = overlays()
            if not items:
                ui.notification_show("Save an overlay first.", type="warning", duration=8)
                req(False)
            report_items, groups = [], []
            for item in items:
                table, summary, message = overlay_eval_for(item)
                if table is None:
                    ui.notification_show(
                        f"'{item['overlay'].name}' cannot be tested on the test rows: {message}",
                        type="warning",
                        duration=10,
                    )
                    req(False)
                overlay, family = item["overlay"], item["family"]
                starts = sorted(overlay.relativities)
                relativities = pd.DataFrame(
                    {
                        "Band": [band_label(start, overlay.banding, overlay.levels or None) for start in starts],
                        "Relativity": [
                            np.nan if overlay.relativities[start] is None else overlay.relativities[start]
                            for start in starts
                        ],
                    }
                )
                info_rows = [
                    ("Overlay", overlay.name),
                    ("Base model", overlay.base_name),
                    ("Feature", overlay.feature),
                    ("Feature type", "categorical" if overlay.levels else "numeric"),
                    ("Banding", "one band per level" if overlay.levels else f"{overlay.banding:g}"),
                    ("Response", item["response"]),
                    ("Training rows", f"{overlay.training_rows:,}"),
                    ("Test rows", f"{summary['rows']:,}"),
                    ("A/E total, base", f"{summary['A/E total base']:.4f}"),
                    ("A/E total, adjusted", f"{summary['A/E total adjusted']:.4f}"),
                    (
                        "Bands changed",
                        f"{sum(1 for v in overlay.relativities.values() if v is not None and abs(v - 1) > 1e-12)}",
                    ),
                ]
                report_items.append(
                    {
                        "name": overlay.name,
                        "base_name": overlay.base_name,
                        "feature": overlay.feature,
                        "response": item["response"],
                        "info_rows": info_rows,
                        "relativities": relativities,
                        "figure": build_test_figure(item, table, errbars=bool(input.overlay_errbars())),
                        "ae_figures": ae_figure_for(item),
                        "plot_note": (
                            f"Test rows ({summary['rows']:,}): A/E total {summary['A/E total base']:.4f} base, "
                            f"{summary['A/E total adjusted']:.4f} adjusted."
                        ),
                    }
                )
                if not any(g["model"] is item["model"] for g in groups):
                    comparison, metric_message = overlay_metric_for(item)
                    if comparison is None:
                        ui.notification_show(metric_message, type="warning", duration=10)
                        req(False)
                    kind = "GBM" if family == "gbm" else "GLM"
                    groups.append(
                        {
                            "model": item["model"],
                            "base_name": overlay.base_name,
                            "overlays": [i["overlay"].name for i in overlays_on_base(item["model"])],
                            "comparison": comparison,
                            "new_label": f"{kind} + Overlays",
                            "old_label": kind,
                        }
                    )
            for group in groups:
                group.pop("model")
            train, _ = overlay_frames()
            yield report.overlay_report_html(
                report_items,
                groups,
                weight=weight(),
                subtitle=f"{len(report_items)} saved overlay(s)",
                app_title=f"{_kpi_name_for(kpi_df, report_items[0]['response'] if report_items else response(), weight())} - Overlays Report",
            )

        # ---------------- Scoring
        score_data = reactive.value(None)
        score_result = reactive.value(None)
        score_model_count = reactive.value(0)

        @reactive.effect
        def _score_source_follows_models():
            """When models are built or removed, switch the Scoring source to the app models, or to a GBM file when none are left."""
            count = len(glm_models()) + len(gbm_models())
            with reactive.isolate():
                previous, current = score_model_count(), input.score_source()
            score_model_count.set(count)
            if count > previous:
                ui.update_radio_buttons("score_source", selected="app")
            elif count == 0 and current == "app":
                ui.update_radio_buttons("score_source", selected="gbm_pkl")

        async def _browse_csv_into(input_id: str, title: str):
            """Open a file dialog on the host computer and put the chosen path in the text box ``input_id``.

            The dialog blocks until it is closed, so it runs in a worker thread and other sessions keep responding.
            """
            path = await asyncio.to_thread(pick_file, title)
            if path:
                ui.update_text(input_id, value=path)

        @reactive.effect
        @reactive.event(input.score_browse, ignore_init=True)
        async def _score_browse_data():
            """Pick the scoring CSV with a file dialog and fill in its path."""
            await _browse_csv_into("score_path", "Choose the unseen-data CSV")

        async def read_csv_path(path: str) -> pd.DataFrame:
            """Read a cleaned CSV in chunks, showing progress; raises if the file is empty or unreadable."""
            chunks = []
            with ui.Progress(min=0, max=1) as progress:
                progress.set(0, message="Loading data", detail="Opening the file")
                await asyncio.sleep(0)
                for chunk, fraction, rows in iter_csv_chunks(path):
                    chunks.append(chunk)
                    progress.set(fraction, message="Loading data", detail=f"{rows:,} rows read")
                    await asyncio.sleep(0)
                progress.set(0.99, message="Loading data", detail="Combining the chunks")
                await asyncio.sleep(0)
                data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                if data.empty:
                    raise ValueError("the file has no rows")
                progress.set(1, message="Loading data", detail=f"{len(data):,} rows loaded")
            return data

        @reactive.effect
        @reactive.event(input.score_load)
        async def _score_load():
            """Load the scoring CSV named on the Scoring tab."""
            path = (input.score_path() or "").strip()
            if not path:
                ui.notification_show("Enter the path to a CSV file first.", type="warning")
                return
            try:
                data = await read_csv_path(path)
            except Exception as e:
                _notify_error("Could not load the data", e)
                return
            score_data.set(data)
            score_result.set(None)
            ui.notification_show(f"Loaded {len(data):,} rows and {data.shape[1]} columns.", type="message")

        @render.text
        def score_data_status():
            """Status line: rows and columns of the scoring data loaded."""
            data = score_data()
            return "No data loaded yet." if data is None else f"{len(data):,} rows x {data.shape[1]} columns loaded."

        @render.ui
        def score_model_select():
            """Model drop-down for Scoring: the GLMs and GBMs built in this session."""
            glm_list, gbm_list = glm_models(), gbm_models()
            choices = {f"glm:{index}": model.name for index, model in enumerate(glm_list)}
            choices.update({f"gbm:{index}": model.name for index, model in enumerate(gbm_list)})
            if not choices:
                return ui.p(
                    "No model has been built yet. Build a GLM or GBM, or read in a model file.",
                    class_="text-muted small",
                )
            default = chart_default_model()
            return ui.input_select(
                "score_model", "Model", choices=choices, selected=default if default in choices else list(choices)[-1]
            )

        @reactive.calc
        def score_model_loaded():
            """The model to score with: ``dict(kind, model, name, objective, power, response, weight)`` or ``dict(error=...)``."""
            source = input.score_source()
            if source == "app":
                try:
                    selected = input.score_model()
                except SilentException:
                    selected = None
                if not selected:
                    return {"error": "No model has been built yet. Build a GLM or GBM, or read in a model file."}
                family, index = selected.split(":", 1)
                models = glm_models() if family == "glm" else gbm_models()
                if not index.isdigit() or int(index) >= len(models):
                    return {"error": "Select a model."}
                model = models[int(index)]
                objective, power = model_objective(model)
                return {
                    "kind": family,
                    "model": model,
                    "name": model.name,
                    "objective": objective,
                    "power": power,
                    "response": model.response,
                    "weight": model.weight,
                }
            if source == "gbm_pkl":
                files = input.score_gbm_file()
                if not files:
                    return {"error": "Read in a GBM model.pkl file."}
                try:
                    model = read_gbm_pickle(files[0]["datapath"])
                except Exception as e:
                    return {"error": f"Could not read the model: {e}"}
                objective, power = model_objective(model)
                return {
                    "kind": "gbm",
                    "model": model,
                    "name": model.name,
                    "objective": objective,
                    "power": power,
                    "response": model.response,
                    "weight": model.weight,
                }
            files = input.score_rb_file()
            if not files:
                return {"error": "Read in a GLM ratebook (.xlsx)."}
            try:
                ratebook = read_ratebook(files[0]["datapath"])
            except Exception as e:
                return {"error": f"Could not read the ratebook: {e}"}
            return {
                "kind": "ratebook",
                "model": ratebook,
                "name": ratebook.model_name or files[0]["name"],
                "objective": None,
                "power": None,
                "response": None,
                "weight": ratebook.weight if ratebook.weight != "N" else None,
            }

        @render.ui
        def score_model_status():
            """Status line describing the model being scored with."""
            loaded = score_model_loaded()
            if loaded.get("error"):
                return ui.p(loaded["error"], class_="text-muted small")
            detail = {"glm": "GLM built in the app", "gbm": "GBM", "ratebook": "GLM ratebook"}[loaded["kind"]]
            if loaded["kind"] == "ratebook":
                detail += f", {len(loaded['model'].features)} features, base prediction {loaded['model'].base_prediction:,.4g}"
            elif loaded["kind"] == "gbm":
                detail += f", {len(loaded['model'].features)} features"
            return ui.p(f"{loaded['name']} ({detail})", class_="small")

        @render.ui
        def score_columns_ui():
            """Response and weight drop-downs for the scoring metrics, defaulting to the model's columns."""
            data = score_data()
            if data is None:
                return ui.p(
                    "Load data to choose the response and weight columns for the metrics.", class_="text-muted small"
                )
            numeric = [
                column
                for column in data.columns
                if pd.api.types.is_numeric_dtype(data[column]) and not pd.api.types.is_bool_dtype(data[column])
            ]
            loaded = score_model_loaded()
            candidates = [loaded.get("response"), response()]
            default_response = next((c for c in candidates if c in numeric), "")
            default_weight = next((c for c in (loaded.get("weight"), weight()) if c == "N" or c in numeric), "N")
            return ui.TagList(
                ui.input_select(
                    "score_response",
                    "Response column",
                    {"": "None (skip metrics)", **{c: c for c in numeric}},
                    selected=default_response,
                ),
                ui.input_select("score_weight", "Weight column", ["N"] + numeric, selected=default_weight),
            )

        @render.ui
        def score_objective_ui():
            """Objective for PoDE: taken from the model, or chosen by the user when the model has none."""
            loaded = score_model_loaded()
            if loaded.get("objective"):
                family = normalise_objective(loaded["objective"])
                power = f", variance power {loaded['power']:g}" if family == "tweedie" and loaded.get("power") else ""
                return ui.p(f"Objective for PoDE: {family}{power} (from the model).", class_="small")
            return ui.TagList(
                ui.input_select("score_objective", "Model objective (for PoDE)", DEVIANCE_OBJECTIVES, selected="gamma"),
                ui.panel_conditional(
                    "input.score_objective == 'tweedie'",
                    ui.input_numeric("score_tweedie_p", "Tweedie variance power", 1.5, min=1.01, max=1.99, step=0.05),
                ),
            )

        @reactive.effect
        @reactive.event(input.score_run)
        async def _score_run():
            """Score the loaded data with the chosen model, and calculate metrics when a response column is chosen."""
            data = score_data()
            if data is None:
                ui.notification_show("Load the scoring data first.", type="warning")
                return
            loaded = score_model_loaded()
            if loaded.get("error"):
                ui.notification_show(loaded["error"], type="warning")
                return
            notes = []
            try:
                with ui.Progress(min=0, max=3) as progress:
                    progress.set(1, message="Scoring", detail=f"{len(data):,} rows on {loaded['name']}")
                    await asyncio.sleep(0)
                    if loaded["kind"] == "ratebook":
                        predictions, notes = score_ratebook(loaded["model"], data, input.score_weight() or "N")
                    elif loaded["kind"] == "gbm":
                        predictions = score_gbm(loaded["model"], data)
                    else:
                        predictions = loaded["model"].score(data)
                    progress.set(2, message="Scoring", detail="Calculating metrics")
                    await asyncio.sleep(0)
                    objective, power = loaded.get("objective"), loaded.get("power")
                    if not objective:
                        objective = input.score_objective() if input.score_objective() else "poisson"
                        power = input.score_tweedie_p() if objective == "tweedie" else None
                    metrics, metrics_error = None, None
                    response_column = input.score_response()
                    if response_column:
                        try:
                            metrics = score_metrics(
                                data,
                                response_column,
                                predictions,
                                input.score_weight() or "N",
                                objective,
                                power or 1.5,
                                weight_basis=model_weight_basis(loaded["model"]),
                            )
                        except Exception as e:
                            metrics_error = str(e)
                    else:
                        metrics_error = "No response column selected, so metrics were skipped."
                    progress.set(3, message="Scoring", detail="Done")
            except Exception as e:
                _notify_error("Could not score the data", e)
                return
            per_unit = loaded["kind"] != "ratebook" and model_weight_basis(loaded["model"]) == "average"
            score_result.set(
                {
                    "predictions": np.asarray(predictions, dtype=float),
                    "notes": notes,
                    "metrics": metrics,
                    "metrics_error": metrics_error,
                    "name": loaded["name"],
                    "objective": normalise_objective(objective),
                    "power": power,
                    "response": response_column or None,
                    "weight": input.score_weight() or "N",
                    "per_unit": per_unit,
                    "path": (input.score_path() or "").strip(),
                    "modelled": scoring_model_features(loaded, data, d0),
                    "rows": len(data),
                    "columns": data.shape[1],
                    "kind": loaded["kind"],
                }
            )
            ui.notification_show(f"Scored {len(data):,} rows.", type="message")

        @render.data_frame
        def score_col_summary():
            """Summary table of every column of the scoring data."""
            data = score_data()
            req(data is not None)
            return render.DataGrid(_show(column_summary(data)), height="650px")

        @render.ui
        def score_notes():
            """Notes after scoring: rows scored, mean prediction, missing predictions and any model notes."""
            result, data = score_result(), score_data()
            if result is None or data is None:
                return ui.p("Load data, choose a model and press Score data.", class_="text-muted")
            predictions = result["predictions"]
            summary = (
                f"Scored {len(predictions):,} rows with {result['name']}. "
                f"Mean prediction {np.nanmean(predictions):,.4f}; "
                f"{int(np.sum(~np.isfinite(predictions))):,} missing predictions."
            )
            items = [ui.p(summary)] + [ui.tags.div(note, class_="text-warning small") for note in result["notes"]]
            return ui.TagList(*items)

        @render.data_frame
        def score_preview():
            """Preview of the scored data: the first 1,000 rows with the predictions."""
            result, data = score_result(), score_data()
            req(result is not None, data is not None)
            return render.DataGrid(_show(scored_frame(data, result["predictions"]).head(1000)), height="560px")

        @render.ui
        def score_metrics_note():
            """Note above the scoring metrics: the response, weight and PoDE objective used."""
            result = score_result()
            if result is None:
                return ui.p("Score the data to see the metrics.", class_="text-muted")
            if result["metrics_error"]:
                return ui.p(result["metrics_error"], class_="text-warning")
            power = (
                f", variance power {result['power']:g}" if result["objective"] == "tweedie" and result["power"] else ""
            )
            return ui.p(
                f"Response: {result['response']}; weight: {result['weight']}; "
                f"PoDE objective: {result['objective']}{power}.",
                class_="text-muted small",
            )

        @render.data_frame
        def score_metrics_table():
            """Metrics of the scored data against the response column."""
            result = score_result()
            req(result is not None, result["metrics"] is not None)
            return render.DataGrid(_show(result["metrics"]), height="200px")

        @reactive.effect
        def _score_feature_choices():
            """Modelled features (the scoring model's features, most important first) and Non-modelled features (the
            other columns, without the response, weight and prediction)."""
            data, result = score_data(), score_result()
            if data is None:
                return
            excluded = {(result or {}).get("response"), (result or {}).get("weight"), "prediction", *NOT_NON_MODELLED}
            columns = [c for c in data.columns if c not in excluded]
            modelled = (result or {}).get("modelled")
            choices = _chart_feature_choices(columns, modelled, excluded=excluded) if modelled is not None else columns
            flat = [c for group in choices.values() for c in group] if isinstance(choices, dict) else choices
            with reactive.isolate():
                current = input.score_feature()
            # after a scoring run the first Modelled feature (the most important) is chosen; otherwise keep the choice
            first_modelled = next((c for c in (modelled or []) if c in columns), None)
            selected = (
                first_modelled
                if first_modelled is not None
                else (current if current in flat else (flat[0] if flat else None))
            )
            ui.update_select("score_feature", choices=choices, selected=selected)

        @reactive.effect
        @reactive.event(input.score_feature, ignore_init=True)
        def _score_feature_defaults():
            """Banding, Min, Max and Bins for the scored plots start from the feature specification of the model data
            (the same starting values as the Charts tab)."""
            data, f = score_data(), input.score_feature()
            if data is None or not f or f not in data.columns:
                return
            series = data[f]
            applied = feature_treatment_spec(feat_df, f) or {}
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                spec_banding = numeric_or_none(feature_lookup(feat_df, f).get("banding"))
                banding = spec_banding if spec_banding else banding_guesser(series)
            else:
                banding = 0
            ui.update_text("score_banding", value="0" if applied.get("bins") else f"{banding:g}")
            ui.update_text("score_bins", value=str(applied["bins"]) if applied.get("bins") else "")
            minimum, maximum = spec_chart_limits(feat_df, f)
            ui.update_text("score_min", value="" if minimum is None else f"{minimum:g}")
            ui.update_text("score_max", value="" if maximum is None else f"{maximum:g}")

        @reactive.calc
        def score_chart_summary():
            """The 1-way summary of the scored data for the Scoring tab: (summary, message)."""
            data, result = score_data(), score_result()
            req(data is not None, result is not None)
            if not result["response"]:
                return (
                    None,
                    "Choose a response column in the Metrics settings and press Score data to plot the 1-way chart.",
                )
            feature = input.score_feature()
            req(feature)
            try:
                banding = (input.score_banding() or "0").strip()
                minimum = numeric_or_none((input.score_min() or "").strip())
                maximum = numeric_or_none((input.score_max() or "").strip())
                bins = bin_count(input.score_bins() or "") if banding in ("", "0") else None
                frame = scored_frame(data, result["predictions"], "prediction")
                summary = one_way_summary(
                    frame,
                    result["response"],
                    result["weight"],
                    feature,
                    ["prediction"],
                    banding=banding or "0",
                    group_low_exposure=input.score_low_exposure(),
                    per_unit_cols=["prediction"] if result["per_unit"] else [],
                    minimum=minimum,
                    maximum=maximum,
                    bins=bins,
                )
            except Exception as error:
                return None, f"Chart unavailable: {error}"
            return summary, None

        @render.ui
        def score_chart_note():
            """Warning under the Scoring chart when the chart cannot be built."""
            _, message = score_chart_summary()
            return ui.p(message, class_="text-warning small") if message else None

        @render_widget
        def score_line_bar_plot():
            """1-way chart of the scored predictions against the response, for the chosen feature."""
            summary, message = score_chart_summary()
            req(summary is not None)
            result, feature = score_result(), input.score_feature()
            figure = charts.line_bar_figure(
                summary,
                feature,
                result["response"],
                ["prediction"],
                show_bars=input.score_show_bars(),
                sigma_bars=input.score_sigma_bars(),
                title=feature,
                y_title=result["response"],
                weight_label=result["weight"],
                x_title=feature_label(feat_df, feature),
            )
            return _json_safe_figure(figure)

        @render.data_frame
        def score_chart_table():
            """Table behind the Scoring 1-way chart."""
            summary, _ = score_chart_summary()
            req(summary is not None)
            return render.DataGrid(_show(summary), height="320px")

        def scoring_details(
            result,
            data,
        ) -> list[tuple[str, str]]:
            """The scoring data and the model used, for the scoring report."""
            predictions = result["predictions"]
            rows = [
                ("Scoring data file", result.get("path") or "not recorded"),
                ("Rows scored", f"{result['rows']:,}"),
                ("Columns in the data", f"{result['columns']:,}"),
                ("Model", result["name"]),
                (
                    "Objective (PoDE)",
                    result["objective"]
                    + (
                        f", variance power {result['power']:g}"
                        if result["objective"] == "tweedie" and result["power"]
                        else ""
                    ),
                ),
                ("Response", result["response"] or "none (metrics skipped)"),
                ("Weight", result["weight"]),
                ("Mean prediction", f"{np.nanmean(predictions):,.4f}"),
                ("Missing predictions", f"{int(np.sum(~np.isfinite(predictions))):,}"),
            ]
            if result["response"] and result["response"] in data.columns:
                rows.append(
                    (
                        "Mean response (unweighted)",
                        f"{pd.to_numeric(data[result['response']], errors='coerce').mean():,.4f}",
                    )
                )
            if result["metrics_error"]:
                rows.append(("Metrics", result["metrics_error"]))
            rows += [("Notes", note) for note in result["notes"]]
            return rows

        @render.download_button(
            filename=lambda: export_filename("daisy_scoring", (score_result() or {}).get("name"), ".html")
        )
        def score_export():
            """Download the scoring report as HTML: data and model details, metrics and 1-way charts."""
            result, data = score_result(), score_data()
            if result is None or data is None:
                ui.notification_show("Load the data and press Score data before exporting.", type="warning", duration=8)
                req(False)
            if not result["response"]:
                ui.notification_show(
                    "Choose a response column and press Score data before exporting.", type="warning", duration=8
                )
                req(False)
            frame = scored_frame(data, result["predictions"], "prediction")
            response, weight = result["response"], result["weight"]
            features = [c for c in data.columns if c not in {response, weight, "prediction"}]
            modelled = result.get("modelled")
            if modelled is not None:  # Modelled features first, then the Non-modelled ones
                modelled = [c for c in modelled if c in features]
                features = modelled + [c for c in features if c not in modelled]
            leading = [report.info_section(scoring_details(result, data))]
            if result["metrics"] is not None:
                leading.append(
                    report.metrics_section(
                        result["metrics"],
                        note=(f"{result['name']}. Metrics use the response and weight chosen for scoring."),
                    )
                )
            feature = input.score_feature() or (features[0] if features else None)
            yield report.export_line_bar_plots(
                frame,
                features,
                modelled,
                feat_df,
                response,
                weight,
                ["prediction"],
                np.ones(len(frame), dtype=np.int8),
                input.score_low_exposure(),
                input.score_show_bars(),
                input.score_sigma_bars(),
                selected_feature=feature,
                selected_banding=input.score_banding() or "0",
                selected_limits=(
                    numeric_or_none((input.score_min() or "").strip()),
                    numeric_or_none((input.score_max() or "").strip()),
                    bin_count(input.score_bins() or "")
                    if (input.score_banding() or "0").strip() in ("", "0")
                    else None,
                ),
                leading_sections=leading,
                app_title=f"{_kpi_name_for(kpi_df, response, weight)} - Scoring Report",
                model_name=result["name"],
                per_unit_cols=["prediction"] if result["per_unit"] else [],
                y_title=response,
            )

        @render.download_button(filename="daisy_predictions.csv")
        def score_download():
            """Download the predictions as a CSV file."""
            result, data = score_result(), score_data()
            if result is None or data is None:
                ui.notification_show("Load the data and press Score data first.", type="warning", duration=8)
                req(False)
            yield pd.DataFrame({"prediction": np.asarray(result["predictions"], dtype=float)}).to_csv(index=False)

    return App(app_ui, server, static_assets={"/images": str(Path(__file__).parent / "images")})


def demo_app() -> App:
    """The app on the bundled insurance demo data, opening on the Charts tab."""
    from .data import load_demo_dataset

    p = demo_spec_paths()
    return make_app(
        load_demo_dataset(),
        dataset_name="insurance",
        kpi_spec=p["kpi"],
        filter_spec=p["filter"],
        feature_spec=p["feature"],
        starting_response="total_premium",
        starting_tab="Charts",
        title="Motor Insurance Pricing",
        num_threads=-1,
    )
