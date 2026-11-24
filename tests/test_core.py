import io, json
import pickle
from types import SimpleNamespace
from daisy.gbm import GBMModel, normalise_objective, unit_deviance
from daisy.scoring import (
    DATASET_LABEL,
    iter_csv_chunks,
    model_objective,
    read_csv_with_progress,
    read_gbm_pickle,
    read_ratebook,
    score_metrics,
    score_ratebook,
    scored_frame,
)
import numpy as np, pandas as pd, pytest
from daisy.data import load_demo_dataset, demo_spec_paths, DATA_DIR
from daisy.specs import load_specification, filter_choices
from daisy.banding import banding_guesser, band_numeric
from daisy.filters import apply_filter, combine_filters
from daisy.summary import one_way_summary, add_total_row
from daisy.glm import build_glm, Design
from daisy.ratebook import tabulate_model, write_ratebook_excel
from scipy import stats
from daisy.gbm import (
    build_gbm,
    model_objective_power,
    model_performance_metrics,
    prediction_performance_metrics,
    score_gbm,
    unit_deviance,
)
from daisy.ratebook import _grid, check_ratebook_link
from daisy.weights import zero_weight_predictions


@pytest.fixture(scope="module")
def d():
    return load_demo_dataset()


def test_demo_shape(d):
    assert d.shape == (283258, 48) and d["train_test"].mean() == pytest.approx(0.2, abs=0.01)


def test_chart_feature_choices_are_grouped_only_with_a_model():
    from daisy.app import _chart_feature_choices

    columns = ["driver_age", "vehicle_brand", "total_claims", "train_test"]
    assert _chart_feature_choices(columns, None) == {column: column for column in columns}
    assert _chart_feature_choices(columns, ["driver_age", "vehicle_brand"]) == {
        "Modelled": {"driver_age": "driver_age", "vehicle_brand": "vehicle_brand"},
        "Non-modelled": {"total_claims": "total_claims", "train_test": "train_test"},
    }
    assert _chart_feature_choices(columns, []) == {
        "Modelled": {},
        "Non-modelled": {column: column for column in columns},
    }


def _export_payload(html: str) -> dict:
    import json

    marker = '<script id="daisy-data" type="application/json">'
    start = html.index(marker) + len(marker)
    return json.loads(html[start : html.index("</script>", start)])


def _sections(html: str) -> dict:
    return {section["id"]: section for section in _export_payload(html)["sections"]}


def test_export_line_plots_single_html_with_modelled_and_non_modelled_tabs():
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame(
        {
            "driver_age": [20, 30, 40],
            "total_premium": [100.0, 200.0, 300.0],
            "weight": [1.0, 2.0, 1.0],
            "a/b": ["x", "y", "z"],
        }
    )
    spec = pd.DataFrame({"feature": ["driver_age"], "banding": ["10"]})
    result = export_line_bar_plots(
        data,
        list(data.columns),
        ["driver_age", "a/b"],
        spec,
        "total_premium",
        "weight",
        [],
        [True, True, False],
        "0",
        True,
        False,
        selected_feature="driver_age",
        selected_banding="5",
    )
    assert isinstance(result, bytes)
    html = result.decode()
    assert html.count("<!DOCTYPE html>") == 1 and '<select id="feature">' in html
    assert 'src="plotly.min.js"' not in html  # plotly is embedded, so one file is enough
    sections = _sections(html)
    assert list(sections) == ["lines"]
    groups = {g["id"]: g for g in sections["lines"]["groups"]}
    assert [f["name"] for f in groups["modelled"]["features"]] == ["driver_age", "a/b"]
    assert [f["name"] for f in groups["non-modelled"]["features"]] == ["total_premium", "weight"]
    assert groups["modelled"]["features"][0]["badge"] == "Banding: 5 · Monotonicity: none"
    assert groups["modelled"]["title"] == "Modelled" and groups["non-modelled"]["title"] == "Non-modelled"
    assert all(f["figs"][0]["fig"].startswith("{") for g in groups.values() for f in g["features"])


def test_export_line_plots_non_modelled_tab_keeps_only_modelling_valid_features():
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame(
        {
            "driver_age": [20, 30, 40],
            "total_premium": [100.0, 200.0, 300.0],
            "weight": [1.0, 2.0, 1.0],
            "vehicle_brand": ["x", "y", "z"],
            "train_test": [0, 0, 1],
        }
    )
    spec = pd.DataFrame(columns=["feature", "banding"])
    result = export_line_bar_plots(
        data,
        list(data.columns),
        ["driver_age"],
        spec,
        "total_premium",
        "weight",
        [],
        [True] * 3,
        "0",
        False,
        False,
        valid_features=["driver_age", "vehicle_brand"],
    )
    groups = {g["id"]: g for g in _sections(result.decode())["lines"]["groups"]}
    assert [f["name"] for f in groups["modelled"]["features"]] == ["driver_age"]
    assert [f["name"] for f in groups["non-modelled"]["features"]] == ["vehicle_brand"]


def test_export_line_plots_without_model_keeps_all_features_on_one_tab():
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame({"response": [1.0, 2.0], "factor": ["a", "b"]})
    spec = pd.DataFrame(columns=["feature", "banding"])
    result = export_line_bar_plots(
        data, list(data.columns), None, spec, "response", "N", [], [True, True], "0", False, False
    )
    groups = _sections(result.decode())["lines"]["groups"]
    assert [g["id"] for g in groups] == ["all"]
    assert [f["name"] for f in groups[0]["features"]] == ["response", "factor"]


def test_export_line_plots_includes_empty_group_tab():
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame({"response": [1.0, 2.0], "feature": [3.0, 4.0]})
    spec = pd.DataFrame(columns=["feature", "banding"])
    result = export_line_bar_plots(
        data, ["feature"], ["feature"], spec, "response", "N", [], [True, True], "0", False, False
    )
    groups = {g["id"]: g for g in _sections(result.decode())["lines"]["groups"]}
    assert groups["non-modelled"]["features"] == [] and len(groups["modelled"]["features"]) == 1
    assert "No features in this group." in result.decode()


def test_export_report_adds_shap_ale_and_metrics_tabs():
    pytest.importorskip("PyALE")
    from types import SimpleNamespace
    from daisy.report import ale_section, export_line_bar_plots, metrics_section, shap_section
    from daisy.gbm import prediction_performance_metrics

    rng = np.random.default_rng(0)
    n = 80
    data = pd.DataFrame(
        {
            "driver_age": rng.uniform(18, 80, n),
            "vehicle_brand": rng.choice(["a", "b"], n),
            "total_premium": rng.uniform(1, 3, n),
            "train_test": np.arange(n) % 5 == 0,
        }
    )
    shap_frame = pd.DataFrame(
        {
            "lgbm_SHAP_driver_age": rng.normal(0, 0.1, n),
            "lgbm_SHAP_vehicle_brand": rng.normal(0, 0.1, n),
            "lgbm_SHAP_base_score": np.zeros(n),
        }
    )
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        name="GBM 1",
        link="log",
        features=["driver_age", "vehicle_brand"],
        weight="N",
        response="total_premium",
        SHAP=shap_frame,
        SHAP_rows=np.arange(n),
        predictions=data["total_premium"].to_numpy() * 1.05,
        pred_rows=None,
        predict=lambda frame: frame["driver_age"].to_numpy() / 50 + (frame["vehicle_brand"] == "a").to_numpy(),
    )
    spec = pd.DataFrame(columns=["feature", "banding"])
    mask = np.ones(n, dtype=bool)
    metrics = prediction_performance_metrics(data, "total_premium", model.predictions, "N", None, mask)
    extra = [
        shap_section(data, model, model.features, spec, "N", mask),
        ale_section(data, model, "gbm", model.features, mask, bins=5, max_rows=50),
        metrics_section(metrics, note="GBM 1"),
    ]
    html = export_line_bar_plots(
        data,
        list(data.columns),
        model.features,
        spec,
        "total_premium",
        "N",
        [],
        mask,
        "0",
        True,
        False,
        valid_features=["driver_age", "vehicle_brand"],
        extra_sections=extra,
        model_name="GBM 1",
    ).decode()
    sections = _sections(html)
    assert list(sections) == ["lines", "shap", "ale", "metrics"]
    shap = sections["shap"]["groups"][0]["features"]
    assert [f["name"] for f in shap] == ["driver_age", "vehicle_brand"] and len(shap[0]["figs"]) == 2
    ale = sections["ale"]["groups"][0]["features"]
    assert [f["name"] for f in ale] == ["driver_age", "vehicle_brand"] and ale[0]["error"] is None
    assert "50 filtered rows" in sections["ale"]["note"]
    assert sections["metrics"]["kind"] == "html" and "<table" in sections["metrics"]["html"]
    assert "Train" in sections["metrics"]["html"] and "Test" in sections["metrics"]["html"]


def test_export_offers_train_and_test_views_for_every_plot_and_metrics():
    from daisy.report import export_line_bar_plots, metrics_section

    data = pd.DataFrame(
        {
            "driver_age": [20.0, 30.0, 40.0, 50.0],
            "total_premium": [100.0, 200.0, 300.0, 400.0],
            "train_test": [0, 0, 1, 1],
        }
    )
    spec = pd.DataFrame(columns=["feature", "banding"])
    ones = np.ones(4, dtype=np.int8)
    masks = {
        "All": ones,
        "Train": (ones * (1 - data["train_test"])).astype(np.int8),
        "Test": (ones * data["train_test"]).astype(np.int8),
    }
    seen = []

    def metrics_for(mask):
        seen.append(int(mask.sum()))
        return metrics_section(pd.DataFrame({"Dataset": ["x"], "GINI": [float(mask.sum())]}))

    result = export_line_bar_plots(
        data,
        ["driver_age"],
        ["driver_age"],
        spec,
        "total_premium",
        "N",
        [],
        ones,
        "0",
        False,
        False,
        valid_features=["driver_age"],
        extra_sections=[metrics_for],
        view_masks=masks,
        default_view="Test",
    ).decode()
    payload = _export_payload(result)
    assert payload["views"] == ["All", "Train", "Test"] and payload["defaultView"] == "Test"
    assert seen == [4, 2, 2]  # callable sections are evaluated once per view
    feature = payload["sections"][0]["groups"][0]["features"][0]
    assert set(feature["views"]) == {"Train", "Test"} and feature["figs"] and feature["views"]["Train"]["figs"]
    metrics = payload["sections"][1]
    assert set(metrics["views"]) == {"Train", "Test"} and "4.0000" in metrics["html"]


def test_export_without_views_has_a_single_all_view():
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame({"driver_age": [20.0, 30.0], "total_premium": [1.0, 2.0]})
    result = export_line_bar_plots(
        data,
        ["driver_age"],
        None,
        pd.DataFrame(columns=["feature", "banding"]),
        "total_premium",
        "N",
        [],
        [True, True],
        "0",
        False,
        False,
    )
    payload = _export_payload(result.decode())
    assert payload["views"] == ["All"]
    assert "views" not in payload["sections"][0]["groups"][0]["features"][0]


def _fake_gbm(**overrides):
    from types import SimpleNamespace

    params = dict(
        objective="tweedie",
        learning_rate=0.05,
        num_leaves=5,
        max_depth=-1,
        feature_fraction=1.0,
        bagging_fraction=0.75,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=0.0,
        min_data_in_leaf=100,
        verbose=-1,
        seed=1,
        num_threads=-1,
        metric="tweedie",
        tweedie_variance_power=1.5,
        monotone_constraints=[1, 0, -1],
        interaction_constraints=[[0, 1], [2]],
    )
    base = dict(
        params=params,
        training_config={
            "num_iterations": 1000,
            "early_stopping_round": 50,
            "shap_sample": "10k",
            "num_threads": -1,
            "seed": 1,
        },
        best_iteration=212,
        features=["driver_age", "vehicle_brand", "total_claims"],
        response="total_premium",
        weight="exposure",
        init_score="no offset",
        weight_basis="rate_weighted",
        interaction_groups=["driver"],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gbm_parameter_table_lists_parameters_and_values():
    from daisy.gbm import gbm_parameter_table

    table = gbm_parameter_table(_fake_gbm())
    assert list(table.columns) == ["Parameter", "Value"]
    values = dict(zip(table["Parameter"], table["Value"]))
    assert values["objective"] == "tweedie" and values["learning_rate"] == "0.05"
    assert values["num_iterations"] == "1000" and values["best_iteration"] == "212"
    assert values["early_stopping_round"] == "50" and values["tweedie_variance_power"] == "1.5"
    assert values["max_depth"] == "-1" and values["weight"] == "exposure" and values["interaction_groups"] == "driver"
    assert values["number_of_features"] == "3" and values["offset (init_score)"] == "no offset"
    assert not {"verbose", "monotone_constraints", "interaction_constraints", "num_threads"} & set(values)
    assert list(table["Parameter"])[:2] == ["objective", "metric"]  # key parameters come first


def test_gbm_monotonicity_table_shows_applied_constraints():
    from daisy.gbm import gbm_monotonicity_table

    table = gbm_monotonicity_table(_fake_gbm())
    assert list(table.columns) == ["Feature", "Monotonicity"]
    assert dict(zip(table["Feature"], table["Monotonicity"])) == {
        "driver_age": "Increasing",
        "vehicle_brand": "None",
        "total_claims": "Decreasing",
    }
    unconstrained = _fake_gbm()
    unconstrained.params = {
        key: vehicle_value for key, vehicle_value in unconstrained.params.items() if key != "monotone_constraints"
    }
    assert set(gbm_monotonicity_table(unconstrained)["Monotonicity"]) == {"None"}


def test_overview_section_can_carry_extra_tables():
    from daisy.report import overview_section
    from daisy.gbm import gbm_monotonicity_table, gbm_parameter_table

    model = _fake_gbm()
    overview = pd.DataFrame({"Statistic": ["Total Weight"], "Train": [10.0], "Test": [5.0]})
    section = overview_section(
        overview,
        tables=[
            ("GBM parameters", gbm_parameter_table(model)),
            ("Monotonicity applied", gbm_monotonicity_table(model)),
        ],
    )
    html = section["html"]
    assert html.index("Total Weight") < html.index("GBM parameters") < html.index("Monotonicity applied")
    assert "learning_rate" in html and "Increasing" in html and 'class="table-pair"' in html
    assert "table-pair" not in overview_section(overview)["html"]


def test_report_logo_is_read_from_the_images_folder_without_web_addresses(tmp_path):
    from daisy.report import IMAGES_DIR, LOGO_FILE, logo_html

    assert (IMAGES_DIR / LOGO_FILE).exists()
    logo = logo_html()
    assert logo.startswith("<svg") and "Daisy flower" in logo and "http" not in logo
    custom = tmp_path / "mark.svg"
    custom.write_text('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 4"><rect/></svg>')
    assert logo_html(custom) == '<svg viewBox="0 0 4 4"><rect/></svg>'
    png = tmp_path / "mark.png"
    png.write_bytes(b"\x89PNG\r\n")
    assert logo_html(png).startswith('<img src="data:image/png;base64,')
    fallback = logo_html(tmp_path / "missing.svg")
    assert fallback.startswith("<svg") and "http" not in fallback


def test_report_page_loads_nothing_from_the_internet():
    import re
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame({"driver_age": [20.0, 30.0], "total_premium": [1.0, 2.0]})
    html = export_line_bar_plots(
        data,
        ["driver_age"],
        None,
        pd.DataFrame(columns=["feature", "banding"]),
        "total_premium",
        "N",
        [],
        [True, True],
        "0",
        False,
        False,
    ).decode()
    page = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)  # Plotly's own bundle is embedded
    assert "<svg" in page and "Daisy flower" in page  # the logo is inlined from the images folder
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page, flags=re.I)
    assert "@import" not in page and "<link" not in page
    assert "http" not in page  # no web addresses at all outside Plotly's embedded bundle


def test_report_title_is_the_app_title():
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame({"driver_age": [20.0, 30.0], "total_premium": [1.0, 2.0]})
    args = (
        data,
        ["driver_age"],
        None,
        pd.DataFrame(columns=["feature", "banding"]),
        "total_premium",
        "N",
        [],
        [True, True],
        "0",
        False,
        False,
    )
    default = export_line_bar_plots(*args).decode()
    assert "<title>Daisy</title>" in default and "<h1>Daisy</h1>" in default
    custom = export_line_bar_plots(*args, app_title="Pricing <Lab> & Co").decode()
    assert "<title>Pricing &lt;Lab&gt; &amp; Co</title>" in custom
    assert "<h1>Pricing &lt;Lab&gt; &amp; Co</h1>" in custom and "<h1>Daisy</h1>" not in custom


def test_lorenz_curve_has_a_bounded_number_of_points():
    from types import SimpleNamespace
    from daisy.charts import LIFT_MAX_POINTS, lorenz_figure

    rng = np.random.default_rng(3)
    n = 20000
    data = pd.DataFrame({"y": rng.gamma(2.0, 50.0, n), "train_test": rng.integers(0, 2, n)})
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        pred_rows=np.arange(n), response="y", predictions=data["y"].to_numpy() * rng.uniform(0.8, 1.2, n)
    )
    figure = lorenz_figure(data, model, "N", np.ones(n, dtype=np.int8))
    curves = [trace for trace in figure.data if trace.name.startswith(("Train", "Test"))]
    assert len(curves) == 2 and all(len(trace.x) <= LIFT_MAX_POINTS for trace in curves)
    assert all(trace.x[-1] == 1.0 and abs(trace.y[-1] - 1.0) < 1e-9 for trace in curves)  # ends at (1, 1)


def test_export_puts_model_overview_first():
    from daisy.report import export_line_bar_plots, overview_section

    data = pd.DataFrame(
        {"driver_age": [20.0, 30.0, 40.0, 50.0], "total_premium": [1.0, 2.0, 3.0, 4.0], "train_test": [0, 0, 1, 1]}
    )
    overview = pd.DataFrame(
        {"Statistic": ["Total Weight", "Mean Predicted"], "Train": [1234.0, 2.5], "Test": [10.5, np.nan]}
    )
    ones = np.ones(4, dtype=np.int8)
    masks = {"All": ones, "Train": np.array([1, 1, 0, 0], dtype=np.int8), "Test": np.array([0, 0, 1, 1], dtype=np.int8)}
    result = export_line_bar_plots(
        data,
        ["driver_age"],
        ["driver_age"],
        pd.DataFrame(columns=["feature", "banding"]),
        "total_premium",
        "N",
        [],
        ones,
        "0",
        False,
        False,
        valid_features=["driver_age"],
        view_masks=masks,
        leading_sections=[overview_section(overview, note="GBM 1")],
    ).decode()
    payload = _export_payload(result)
    assert [section["id"] for section in payload["sections"]] == ["overview", "lines"]
    overview_html = payload["sections"][0]["html"]
    assert "Total Weight" in overview_html and "1,234" in overview_html and "10.50" in overview_html
    assert "N/A" in overview_html and "views" not in payload["sections"][0]  # same table for every Rows view
    assert payload["views"] == ["All", "Train", "Test"]


def test_shap_flame_figure_labels_exposure_bars_with_weight_name():
    from types import SimpleNamespace
    from daisy.charts import shap_flame_figure

    rng = np.random.default_rng(1)
    n = 30
    data = pd.DataFrame({"driver_age": rng.uniform(18, 80, n), "exposure": rng.uniform(0.5, 1, n)})
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        link="log",
        features=["driver_age"],
        SHAP_rows=np.arange(n),
        SHAP=pd.DataFrame({"lgbm_SHAP_driver_age": rng.normal(0, 0.1, n)}),
    )
    mask = np.ones(n, dtype=bool)
    default = shap_flame_figure(data, model, "driver_age", "N", "10", mask)
    assert [t.name for t in default.data if t.type == "bar"] == ["weight"]
    labelled = shap_flame_figure(data, model, "driver_age", "exposure", "10", mask, weight_label="exposure")
    assert [t.name for t in labelled.data if t.type == "bar"] == ["exposure"]
    assert labelled.layout.yaxis2.title.text == "exposure"
    assert "exposure" in [t for t in labelled.data if t.type == "bar"][0].hovertemplate


def test_shap_section_reports_features_without_shap_values():
    from types import SimpleNamespace
    from daisy.report import shap_section

    data = pd.DataFrame({"driver_age": [20.0, 30.0], "total_premium": [1.0, 2.0]})
    model = SimpleNamespace(name="GBM", link="log", features=["driver_age"], SHAP=None, SHAP_rows=None)
    section = shap_section(
        data, model, ["driver_age"], pd.DataFrame(columns=["feature", "banding"]), "N", np.ones(2, dtype=bool)
    )
    entry = section["groups"][0]["features"][0]
    assert "no SHAP values" in entry["error"] and entry["figs"] == []


def test_non_modelled_features_exclude_only_response_and_weight():
    from daisy.app import _chart_feature_choices

    columns = ["driver_age", "vehicle_brand", "total_premium", "N_exp", "train_test", "circulation_area"]
    choices = _chart_feature_choices(
        columns, ["vehicle_brand", "driver_age"], excluded={"total_premium", "N_exp", "train_test"}
    )
    assert list(choices["Modelled"]) == ["vehicle_brand", "driver_age"]  # most important first, as given
    assert list(choices["Non-modelled"]) == ["circulation_area"]  # the app passes train_test as excluded too


def test_line_bar_figure_primary_axis_uses_kpi_name_when_given():
    from daisy.charts import line_bar_figure

    summary = pd.DataFrame({"driver_age": [20, 30], "total_premium": [100.0, 200.0], "weight": [5.0, 7.0]})
    assert line_bar_figure(summary, "driver_age", "total_premium", []).layout.yaxis.title.text == "total_premium"
    figure = line_bar_figure(summary, "driver_age", "total_premium", [], y_title="Insurance premium")
    assert figure.layout.yaxis.title.text == "Insurance premium"
    assert figure.layout.yaxis2.title.text == "Prop. of weight"
    # the bars are each band's share of the total weight, with the weight itself kept for the hover text
    exposure = line_bar_figure(summary, "driver_age", "total_premium", [], weight_label="Exposure")
    assert exposure.layout.yaxis2.title.text == exposure.data[0].name == "Prop. of Exposure"
    assert list(exposure.data[0].y) == pytest.approx([5 / 12, 7 / 12])
    assert list(exposure.data[0].customdata) == [5.0, 7.0] and exposure.layout.yaxis2.tickformat == ".1%"
    assert [trace.name for trace in figure.data if trace.type == "scatter"] == ["Insurance premium"]
    labelled = line_bar_figure(summary, "driver_age", "total_premium", [], weight_label="exposure")
    assert labelled.layout.yaxis2.title.text == "Prop. of exposure"
    assert [trace.name for trace in labelled.data if trace.type == "bar"] == ["Prop. of exposure"]
    assert "exposure" in labelled.data[0].hovertemplate
    assert [
        trace.name
        for trace in line_bar_figure(summary, "driver_age", "total_premium", []).data
        if trace.type == "scatter"
    ] == ["total_premium"]


def test_export_uses_kpi_name_as_primary_axis_title():
    import json
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame({"driver_age": [20.0, 30.0], "total_premium": [100.0, 200.0]})
    result = export_line_bar_plots(
        data,
        ["driver_age"],
        None,
        pd.DataFrame(columns=["feature", "banding"]),
        "total_premium",
        "N",
        [],
        [True, True],
        "0",
        False,
        False,
        y_title="Insurance premium",
    )
    figure = json.loads(_export_payload(result.decode())["sections"][0]["groups"][0]["features"][0]["figs"][0]["fig"])
    assert figure["layout"]["yaxis"]["title"]["text"] == "Insurance premium"


def test_kpi_launch_defaults_and_axis_title():
    pytest.importorskip("shiny")
    from daisy.app import _kpi_axis_title, _kpi_defaults

    kpi = load_specification(None, demo_spec_paths()["kpi"], "kpi")
    columns = ["total_premium", "driver_age", "vehicle_age", "power_to_weight_ratio", "exposure"]
    assert _kpi_defaults(kpi, columns, None) == ("Avg Premium", "total_premium", "N")
    assert _kpi_defaults(kpi, columns, "driver_age") == (None, "driver_age", "N")
    assert _kpi_defaults(kpi, columns + ["total_claims"], "total_claims") == (
        None,
        "total_claims",
        "N",
    )  # no KPI for it
    assert _kpi_defaults(kpi, columns, "unknown") == ("Avg Premium", "total_premium", "N")  # not a column
    weighted = kpi.copy()
    weighted.loc[0, "kpi_denominator"] = "exposure"
    assert _kpi_defaults(weighted, columns, "total_premium") == ("Avg Premium", "total_premium", "exposure")
    assert _kpi_defaults(kpi, ["unrelated"], None) is None
    assert _kpi_axis_title(kpi, "Customer driver_age", "driver_age", "N") == "driver_age"
    assert (
        _kpi_axis_title(kpi, "Customer driver_age", "total_premium", "N") == "total_premium"
    )  # response changed by hand
    assert _kpi_axis_title(kpi, None, "total_premium", "N") == "total_premium"


def test_summary_table_labels_response_rate_with_kpi_name():
    pytest.importorskip("shiny")
    from daisy.app import _label_response_with_kpi

    summary = pd.DataFrame(
        {
            "driver_age": ["20", "Total"],
            "count": [2, 2],
            "weight": [2.0, 2.0],
            "total_premium": [150.0, 150.0],
            "total_premium_sum": [300.0, 300.0],
            "sigma": [1.0, np.nan],
        }
    )
    labelled = _label_response_with_kpi(summary, "total_premium", "Insurance premium")
    assert list(labelled.columns) == [
        "driver_age",
        "count",
        "weight",
        "Insurance premium",
        "total_premium_sum",
        "sigma",
    ]
    assert list(summary.columns)[3] == "total_premium"  # the plot's summary is untouched
    assert _label_response_with_kpi(summary, "total_premium", "total_premium") is summary
    assert _label_response_with_kpi(summary, "total_premium", "sigma") is summary  # no clashing column names


def test_summary_table_labels_weight_column_with_weight_name():
    pytest.importorskip("shiny")
    from daisy.app import _label_weight_column

    summary = pd.DataFrame({"driver_age": ["20"], "count": [2], "weight": [2.0], "total_premium": [150.0]})
    assert list(_label_weight_column(summary, "exposure").columns) == [
        "driver_age",
        "count",
        "exposure",
        "total_premium",
    ]
    assert list(_label_weight_column(summary, "N").columns) == ["driver_age", "count", "N", "total_premium"]
    assert _label_weight_column(summary, "total_premium") is summary  # no clashing column names
    assert _label_weight_column(summary, "weight") is summary


def test_feature_label_uses_label_column_with_feature_name_fallback():
    from daisy.data import demo_spec_paths
    from daisy.specs import feature_label, load_specification

    spec = pd.DataFrame(
        {"feature": ["driver_age", "vehicle_brand", "policy_type"], "label": ["PolicyHolderAge", "", "  "]}
    )
    assert feature_label(spec, "driver_age") == "PolicyHolderAge"
    assert (
        feature_label(spec, "vehicle_brand") == "vehicle_brand" and feature_label(spec, "policy_type") == "policy_type"
    )
    assert feature_label(spec, "unknown") == "unknown"
    assert (
        feature_label(spec.drop(columns="label"), "driver_age") == "driver_age"
        and feature_label(None, "driver_age") == "driver_age"
    )
    demo = load_specification(None, demo_spec_paths()["feature"], "feature")
    assert feature_label(demo, "driver_age") == "Driver Age"


def test_plots_title_the_x_axis_with_the_feature_label():
    import json
    from types import SimpleNamespace
    from daisy.charts import ale_figure, line_bar_figure, mean_shap_figure
    from daisy.report import export_line_bar_plots, shap_section

    summary = pd.DataFrame({"driver_age": [20, 30], "total_premium": [100.0, 200.0], "weight": [5.0, 7.0]})
    assert line_bar_figure(summary, "driver_age", "total_premium", []).layout.xaxis.title.text == "driver_age"
    assert (
        line_bar_figure(summary, "driver_age", "total_premium", [], x_title="PolicyHolderAge").layout.xaxis.title.text
        == "PolicyHolderAge"
    )
    result = pd.Series([0.0, 0.1], index=[20, 30], name="eff").to_frame()
    assert ale_figure(result, ["driver_age"], ["PolicyHolderAge"]).layout.xaxis.title.text == "PolicyHolderAge"
    assert ale_figure(result, ["driver_age"]).layout.xaxis.title.text == "driver_age"
    level_summary = pd.DataFrame({"level": ["a", "b"], "mean": [1.0, 1.1], "weight": [1.0, 1.0], "count": [1, 1]})
    assert (
        mean_shap_figure(level_summary, "driver_age", "log", x_title="PolicyHolderAge").layout.xaxis.title.text
        == "PolicyHolderAge"
    )

    spec = pd.DataFrame({"feature": ["driver_age"], "label": ["PolicyHolderAge"], "banding": ["10"]})
    data = pd.DataFrame({"driver_age": [20.0, 30.0, 40.0, 50.0], "total_premium": [1.0, 2.0, 3.0, 4.0]})
    html = export_line_bar_plots(
        data,
        ["driver_age"],
        ["driver_age"],
        spec,
        "total_premium",
        "N",
        [],
        [True] * 4,
        "0",
        False,
        False,
        valid_features=["driver_age"],
    ).decode()
    start = html.index('<script id="daisy-data" type="application/json">') + len(
        '<script id="daisy-data" type="application/json">'
    )
    payload = json.loads(html[start : html.index("</script>", start)])
    figure = json.loads(payload["sections"][0]["groups"][0]["features"][0]["figs"][0]["fig"])
    assert figure["layout"]["xaxis"]["title"]["text"] == "PolicyHolderAge"

    rng = np.random.default_rng(2)
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        name="GBM",
        link="log",
        features=["driver_age"],
        SHAP_rows=np.arange(4),
        SHAP=pd.DataFrame({"lgbm_SHAP_driver_age": rng.normal(0, 0.1, 4)}),
    )
    section = shap_section(data, model, ["driver_age"], spec, "N", np.ones(4, dtype=bool))
    for item in section["groups"][0]["features"][0]["figs"]:
        assert json.loads(item["fig"])["layout"]["xaxis"]["title"]["text"] == "PolicyHolderAge"


def test_chart_feature_choices_use_fitted_glm_features(d):
    from daisy.app import _chart_feature_choices

    model = build_glm(d.iloc[:1000], "total_premium", "N", "driver_age + C(municipality_type)", "gaussian")
    choices = _chart_feature_choices(list(d.columns), model.features(d.columns))
    assert set(choices["Modelled"]) == {"driver_age", "municipality_type"}
    assert "total_claims" in choices["Non-modelled"]


def test_ale_numeric_and_categorical_effects():
    from daisy.ale import calculate_ale
    from daisy.charts import ale_figure

    class Predictor:
        def predict(
            self,
            data,
        ):
            return 2 * data["driver_age"].to_numpy() + (data["group"].astype(str) == "b").to_numpy()

    data = pd.DataFrame({"driver_age": np.arange(20, dtype=float), "group": ["a", "b"] * 10})
    numeric = calculate_ale(data, Predictor(), ["driver_age"], grid_size=5)
    categorical = calculate_ale(data, Predictor(), ["group"], grid_size=5)
    assert numeric["eff"].iloc[-1] - numeric["eff"].iloc[0] == pytest.approx(38)
    assert categorical.loc["b", "eff"] - categorical.loc["a", "eff"] == pytest.approx(1)
    assert ale_figure(numeric, ["driver_age"]).data[0].type == "scatter"
    with pytest.raises(ValueError, match="two numeric features"):
        calculate_ale(data, Predictor(), ["driver_age", "group"])


def test_ale_observation_counts_cover_every_row_including_the_first_level():
    from daisy.ale import observation_counts

    ages = pd.Series(np.arange(20, dtype=float))
    grid = [0.0, 4.0, 8.0, 12.0, 16.0, 19.0]  # quantile-style grid ending at the maximum
    counts = observation_counts(ages, grid, numeric=True)
    assert list(counts) == [4, 4, 4, 4, 3, 1] and counts.sum() == 20 and counts[0] > 0
    groups = pd.Series(["a", "b", "b", "c", "b", "a"], dtype="category")
    levels = ["a", "b", "c"]
    assert list(observation_counts(groups, levels, numeric=False)) == [2, 3, 1]
    assert list(observation_counts(groups, ["a", "b", "c", "z"], numeric=False)) == [2, 3, 1, 0]


def test_ale_table_size_column_is_not_zero_for_first_level():
    pytest.importorskip("PyALE")
    from daisy.ale import calculate_ale

    class Predictor:
        def predict(
            self,
            data,
        ):
            return 2 * data["driver_age"].to_numpy() + (data["group"].astype(str) == "b").to_numpy()

    data = pd.DataFrame({"driver_age": np.arange(20, dtype=float), "group": ["a", "b"] * 10})
    numeric = calculate_ale(data, Predictor(), ["driver_age"], grid_size=5)
    categorical = calculate_ale(data, Predictor(), ["group"], grid_size=5)
    assert numeric["size"].iloc[0] > 0 and numeric["size"].sum() == 20
    assert list(categorical["size"]) == [10, 10]


def test_ale_two_numeric_features_and_invalid_data():
    from daisy.ale import calculate_ale
    from daisy.charts import ale_figure

    class Predictor:
        def predict(
            self,
            data,
        ):
            return data["x"].to_numpy() * data["y"].to_numpy()

    data = pd.DataFrame({"x": np.tile(np.arange(5, dtype=float), 5), "y": np.repeat(np.arange(5, dtype=float), 5)})
    result = calculate_ale(data, Predictor(), ["x", "y"], grid_size=4)
    figure = ale_figure(result, ["x", "y"])
    assert figure.data[0].type == "heatmap"
    np.testing.assert_allclose(figure.data[0].z, result.to_numpy().T)
    with pytest.raises(ValueError, match="two distinct values"):
        calculate_ale(data.iloc[:1], Predictor(), ["x"])


def test_ale_with_fitted_glm(d):
    from daisy.ale import calculate_ale

    data = d.iloc[:1000].copy()
    model = build_glm(data, "total_premium", "N", "driver_age + C(municipality_type)", "gaussian")
    for feature in ("driver_age", "municipality_type"):
        result = calculate_ale(data, model, [feature], grid_size=5)
        assert np.isfinite(result["eff"]).all()


def test_ale_with_fitted_gbm():
    pytest.importorskip("lightgbm")
    from daisy.ale import calculate_ale
    from daisy.gbm import build_gbm

    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "driver_age": rng.uniform(18, 80, 120),
            "group": np.where(np.arange(120) % 2, "a", "b"),
            "target": rng.uniform(1, 3, 120),
            "train_test": np.arange(120) % 5 == 0,
        }
    )
    model = build_gbm(
        data,
        "target",
        "N",
        ["driver_age", "group"],
        {"objective": "regression", "num_iterations": 3, "early_stopping_round": 0, "min_data_in_leaf": 5},
        shap_sample="No",
        num_threads=1,
    )
    for feature in ("driver_age", "group"):
        result = calculate_ale(data, model, [feature], grid_size=5)
        assert np.isfinite(result["eff"]).all()


def test_specs_load():
    p = demo_spec_paths()
    assert "driver_age" in set(load_specification(None, p["feature"], "feature")["feature"])
    assert "Avg Premium" in set(load_specification(None, p["kpi"], "kpi")["kpi_name"])
    g = filter_choices(load_specification(None, p["filter"], "filter"))
    assert any("driver_age<25" in filters for filters in g.values())


def test_report_titles_use_the_kpi_name_for_the_response():
    from daisy.app import _kpi_name_for

    kpi = pd.DataFrame(
        {
            "kpi_name": ["Avg Premium", "Claim Frequency", "Claim Frequency (per exposure)"],
            "kpi_numerator": ["total_premium", "total_claims", "total_claims"],
            "kpi_denominator": ["N", "total_exposure", "N"],
        }
    )
    assert _kpi_name_for(kpi, "total_premium", "N") == "Avg Premium"
    assert _kpi_name_for(kpi, "total_claims", "total_exposure") == "Claim Frequency"
    assert _kpi_name_for(kpi, "total_claims", "N") == "Claim Frequency (per exposure)"
    assert _kpi_name_for(kpi, "driver_age", "N") == "driver_age"  # no KPI: the column name


def test_report_plots_apply_the_specification_min_and_max_to_other_features():
    from daisy.report import _feature_limits, _feature_banding

    data = pd.DataFrame({"driver_age": [18.0, 30.0, 70.0], "vehicle_brand": ["A", "B", "A"]})
    spec = pd.DataFrame(
        {"feature": ["driver_age", "vehicle_brand"], "min": ["17", ""], "max": ["100", ""], "banding": ["5", ""]}
    )
    assert _feature_limits(data, "driver_age", spec) == (17.0, 100.0, None)
    assert _feature_limits(data, "vehicle_brand", spec) == (None, None, None)
    assert _feature_banding(data, "driver_age", spec) == "5.0"
    # the selected feature keeps the chart setting it was given
    assert _feature_limits(data, "driver_age", spec, "driver_age", (20.0, None, None)) == (20.0, None, None)


def test_one_way_summary_limits_the_number_of_groups():
    from daisy.summary import MAX_GROUPS, one_way_summary

    rng = np.random.default_rng(9)
    n = 20000
    d = pd.DataFrame(
        {
            "x": rng.uniform(0, 1e5, n),
            "code": [f"c{i}" for i in rng.integers(0, 5000, n)],
            "claims": rng.poisson(0.1, n).astype(float),
            "N": 1.0,
        }
    )
    numeric = one_way_summary(d, "claims", "N", "x", [], banding="0")
    assert len(numeric) <= MAX_GROUPS and "bands" in numeric.attrs["grouping_note"]
    assert numeric["count"].sum() == n  # every row still counted once
    categorical = one_way_summary(d, "claims", "N", "code", [], banding="0")
    assert len(categorical) == MAX_GROUPS and "Other" in set(categorical["code"].astype(str))
    assert categorical["count"].sum() == n
    small = one_way_summary(d.head(50), "claims", "N", "x", [], banding="10000")
    assert small.attrs["grouping_note"] is None


def test_overlay_bands_apply_the_specification_min_and_max():
    from daisy.overlay import Overlay, band_starts

    values = [500.0, 5000.0, 200000.0]
    assert list(band_starts(values, 1000, minimum=1000, maximum=115000)) == [1000.0, 5000.0, 115000.0]
    assert list(band_starts(values, 1000)) == [0.0, 5000.0, 200000.0]  # no limits: unchanged
    overlay = Overlay("vv", "GLM 1", "vehicle_value", 1000.0, minimum=1000.0, maximum=115000.0)
    assert list(overlay.band_keys(pd.Series(values))) == [1000.0, 5000.0, 115000.0]


def test_zero_ae_band_keeps_relativity_one():
    from daisy.overlay import suggest

    table = pd.DataFrame({"n": [50, 50], "A": [0.0, 80.0], "E": [10.0, 40.0], "AE": [0.0, 2.0]}, index=[0.0, 1.0])
    intervals = pd.DataFrame(
        {
            "lo99": [0.0, 1.5],
            "hi99": [0.0, 2.5],
            "lo95": [0.0, 1.6],
            "hi95": [0.0, 2.4],
            "lo90": [0.0, 1.7],
            "hi90": [0.0, 2.3],
        },
        index=[0.0, 1.0],
    )
    out = suggest(table, intervals)
    assert out.loc[0.0, "status"] == "Adjust" and out.loc[0.0, "relativity"] == 1.0
    assert out.loc[1.0, "status"] == "Adjust" and out.loc[1.0, "relativity"] == pytest.approx(2.0)


def test_filter_dropdown_values_are_the_filter_expressions(d):
    from daisy.specs import filter_select_choices
    from daisy.filters import apply_filter

    choices = filter_select_choices(load_specification(None, demo_spec_paths()["filter"], "filter"))
    assert choices["no filter"] == "no filter"
    assert choices["DriverAge"] == {"driver_age<25": "driver_age<25", "driver_age>=25": "driver_age>=25"}
    for group, options in choices.items():
        for value in [options] if isinstance(options, str) else options.values():
            user, _, message = apply_filter(d, value)
            assert message == value or value == "no filter", (group, value, message)
    user, _, _ = apply_filter(d, choices["DriverAge"]["driver_age<25"])
    assert user.sum() == (d.driver_age < 25).sum()


def test_spec_editor_parses_csv_and_line_filters(d):
    from daisy.app import _parse_editor_spec

    feature_text = "feature,base_level,min,max,interaction_grouping\ndriver_age,40,17,100,driver"
    feature = _parse_editor_spec(feature_text, "feature", list(d.columns))
    filters = _parse_editor_spec(
        "--- Make\nvehicle_brand %in% c('BMW','FORD')\ndriver_age > 20", "filter", list(d.columns)
    )

    assert feature.loc[0, "feature"] == "driver_age"
    assert filters["filter"].tolist() == ["--- Make", "vehicle_brand %in% c('BMW','FORD')", "driver_age > 20"]


def test_spec_editor_rejects_missing_schema_and_unknown_feature(d):
    from daisy.app import _parse_editor_spec

    with pytest.raises(ValueError, match="missing required columns"):
        _parse_editor_spec("feature,base_level\ndriver_age,40", "feature", list(d.columns))
    feature_text = "feature,base_level,min,max,interaction_grouping\n" "not_a_column,,, ,driver"
    with pytest.raises(ValueError, match="unknown feature columns"):
        _parse_editor_spec(feature_text, "feature", list(d.columns))


def test_spec_editor_validates_table_edits(d):
    from daisy.app import _read_editor_grid, _validate_editor_spec
    from shiny.types import SilentException

    edited = pd.DataFrame(
        {
            "feature": ["driver_age"],
            "base_level": ["40"],
            "min": ["17"],
            "max": ["100"],
            "interaction_grouping": ["driver"],
        }
    )
    validated = _validate_editor_spec(edited, "feature", list(d.columns))

    assert validated.loc[0, "base_level"] == "40"
    edited.loc[0, "feature"] = "missing_feature"
    with pytest.raises(ValueError, match="unknown feature columns"):
        _validate_editor_spec(edited, "feature", list(d.columns))

    class UnmountedGrid:
        def data_patched(self):
            raise SilentException()

    fallback = _read_editor_grid(UnmountedGrid(), validated, "feature", list(d.columns))
    pd.testing.assert_frame_equal(fallback, validated)


def test_gbm_feature_choices_use_modelling_valid_spec():
    from daisy.app import _gbm_features_from_spec

    feature_spec = pd.DataFrame(
        {
            "feature": ["driver_age", "total_premium", "train_test", "total_claims", "unknown"],
            "modelling_valid": ["feature", "feature", "feature", "", "feature"],
        }
    )

    assert _gbm_features_from_spec(
        feature_spec, ["driver_age", "total_premium", "train_test", "total_claims"], "total_premium", "total_claims"
    ) == ["driver_age"]
    assert (
        _gbm_features_from_spec(pd.DataFrame({"feature": ["driver_age"]}), ["driver_age"], "total_premium", "N") == []
    )


def test_missing_spec_falls_back_to_template(d):
    assert list(load_specification(d, "nope.csv", "filter")["filter"]) == ["no filter"]
    assert len(load_specification(d, None, "feature")) == d.shape[1]


def test_feature_template_mirrors_the_bundled_feature_spec(d):
    from daisy.data import demo_spec_paths

    bundled = pd.read_csv(demo_spec_paths()["feature"], dtype=str, keep_default_na=False)
    template = load_specification(d, None, "feature")
    assert list(template.columns) == list(bundled.columns)
    assert list(template["feature"]) == list(d.columns)
    valid = set(template.loc[template["modelling_valid"] == "feature", "feature"])
    assert valid == set(d.columns) - {"train_test"}
    assert (template["label"] == "").all()


def test_model_overview_summarises_train_and_test():
    from daisy.gbm import model_overview

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0], "w": [1.0, 1.0, 2.0, 2.0, 1.0, 3.0], "train_test": [0, 0, 0, 0, 1, 1]}
    )
    pred = np.array([2.0, 2.0, 4.0, 6.0, 8.0, 12.0])
    overview = model_overview(data, "y", pred, "w", weight_basis="average")  # y and pred are per unit of weight
    assert list(overview.columns) == ["Statistic", "Train", "Test"]
    assert list(overview["Statistic"]) == [
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
    train, test = overview["Train"].to_numpy(), overview["Test"].to_numpy()
    assert train[0] == 6 and test[0] == 4
    assert train[1] == 1 + 2 + 3 + 4 and test[1] == 10 + 20  # total response: the sum of the response
    assert np.isclose(train[2], (1 + 2 + 6 + 8) / 6) and np.isclose(test[2], (10 + 60) / 4)
    assert np.isclose(train[3], (2 + 2 + 8 + 12) / 6) and np.isclose(test[3], (8 + 36) / 4)
    assert train[4] == 2.0 and train[8] == 6.0 and np.isclose(train[6], 3.5)
    assert np.isclose(train[5], np.quantile([2.0, 2.0, 4.0, 6.0], 0.25)) and np.isclose(
        train[7], np.quantile([2.0, 2.0, 4.0, 6.0], 0.75)
    )
    assert test[4] == 8.0 and test[8] == 12.0 and np.isclose(test[6], 10.0)
    # response and predictions that are totals (the default): actual and predicted per unit of weight are sum / sum
    totals = model_overview(data, "y", pred, "w")
    assert np.isclose(totals.loc[2, "Train"], (1 + 2 + 3 + 4) / 6) and np.isclose(totals.loc[3, "Test"], (8 + 12) / 4)
    assert totals.loc[1, "Train"] == 1 + 2 + 3 + 4 and totals.loc[1, "Test"] == 10 + 20
    # a user filter restricts the rows, and without a train_test column there is a single "All" column
    filtered = model_overview(data, "y", pred, "w", rows_mask=np.array([1, 1, 0, 0, 1, 1]))
    assert filtered.loc[0, "Train"] == 2 and filtered.loc[0, "Test"] == 4 and filtered.loc[1, "Train"] == 1 + 2
    assert list(model_overview(data.drop(columns="train_test"), "y", pred, "w").columns) == ["Statistic", "All"]


def test_banding_guesser(d):
    assert banding_guesser(d["driver_age"]) == 1 and banding_guesser(d["vehicle_value"]) == 5000


def test_filters(d):
    u, t, m = apply_filter(d, "driver_age<25")
    assert u.sum() == (d.driver_age < 25).sum() and m == "driver_age<25"
    u, t, _ = apply_filter(d, "vehicle_brand %in% c('BMW','FORD')")
    assert u.sum() == d.vehicle_brand.isin(["BMW", "FORD"]).sum()
    u, t, _ = apply_filter(d, "driver_age<25", "Train")
    assert t.sum() == ((d.driver_age < 25) & (d.train_test == 0)).sum()
    u, t, m = apply_filter(d, "nonsense_col > 3")
    assert u.sum() == len(d) and m != "no filter"  # bad filter falls back to no filter
    assert combine_filters(["a>1", "b<2"], "NAND") == "~((a>1) & (b<2))"


def test_one_way_summary_reconciles(d):
    s = one_way_summary(d, "total_premium", "N", "driver_age", ["total_claims"], banding=5)
    assert s["count"].sum() == len(d)
    assert s["total_premium_sum"].sum() == pytest.approx(d["total_premium"].sum())
    tot = add_total_row(s, "total_premium", ["total_claims"]).iloc[-1]
    assert tot["total_premium"] == pytest.approx(d["total_premium"].mean())


def test_low_exposure_grouping(d):
    s = one_way_summary(d, "total_premium", "N", "vehicle_brand", [], group_low_exposure="1%")
    assert len(s) <= d.vehicle_brand.nunique()
    assert s["count"].sum() == len(d)


def test_patsy_design_supports_interactions_transforms_and_reuse(d):
    design = Design(
        "driver_age * C(municipality_type) + I(np.clip(age_driving_licence, 0, 10)) "
        "+ bs(driver_age, df=3, lower_bound=0, upper_bound=100)"
    )
    train_matrix = design.build(d.iloc[:1000], fit=True)
    prediction_matrix = design.build(d.iloc[1000:1100], fit=False)

    assert train_matrix.columns[0] == "(Intercept)"
    assert any(":" in column for column in train_matrix.columns)
    assert prediction_matrix.columns.tolist() == train_matrix.columns.tolist()
    assert prediction_matrix.shape == (100, train_matrix.shape[1])


def test_patsy_prediction_restores_fitted_categorical_levels(d):
    design = Design("municipality_type", base_levels={"municipality_type": "I"})
    design.build(d, fit=True)
    single_level = pd.DataFrame({"municipality_type": ["I"]})

    prediction_matrix = design.build(single_level, fit=False)

    assert prediction_matrix.columns.tolist() == design.columns
    assert prediction_matrix.iloc[0, 0] == 1
    assert prediction_matrix.iloc[0, 1:].sum() == 0


def test_glm_demo_formula_and_sklearn_crosscheck(d):
    from sklearn.linear_model import GammaRegressor

    f = "\n".join(l for l in (DATA_DIR / "glm_formula.csv").read_text().splitlines() if not l.startswith("####"))
    m = build_glm(d, "total_premium", "N", f, "gamma", data_to_use="Training")
    assert m.fit.converged and not np.isnan(m.fit.coef).any()
    tr = d[d.train_test == 0]
    X = m.design.build(tr.reset_index(drop=True), fit=False).drop(columns="(Intercept)").to_numpy(float)
    sk = GammaRegressor(alpha=0, tol=1e-10, max_iter=1000, solver="newton-cholesky").fit(
        X, tr.total_premium.to_numpy(float)
    )
    np.testing.assert_allclose(m.predict(tr), sk.predict(X), rtol=1e-3)


def test_poisson_matches_sklearn(d):
    from sklearn.linear_model import PoissonRegressor

    formula = "I(np.minimum(10, age_driving_licence)) + I(1 * (policy_status == 'C')) + municipality_type"
    m = build_glm(d, "total_claims", "N", formula, "poisson")
    X = m.design.build(d, fit=False).drop(columns="(Intercept)").to_numpy(float)
    sk = PoissonRegressor(alpha=0, tol=1e-12, max_iter=1000, solver="newton-cholesky").fit(
        X, d.total_claims.to_numpy(float)
    )
    np.testing.assert_allclose(m.predictions, sk.predict(X), rtol=1e-6)
    assert m.fit.dispersion == 1.0


def test_glm_terms_reconcile(d):
    formula = (
        "I(np.minimum(10, age_driving_licence)) + municipality_type "
        "+ bs(np.clip(driver_age, 25, 65), df=3, lower_bound=25, upper_bound=65)"
    )
    m = build_glm(d, "total_premium", "N", formula, "gamma")
    tc = m.term_contributions(d)
    lp = m.linear_predictor(d)
    np.testing.assert_allclose(tc.sum(axis=1) + m.design.col_means @ np.nan_to_num(m.fit.coef), lp, atol=1e-9)
    assert set(m.feature_contributions(d).columns) == {
        "glm_LP_age_driving_licence",
        "glm_LP_municipality_type",
        "glm_LP_driver_age",
    }


def test_aliased_columns_get_nan(d):
    m = build_glm(d, "total_premium", "N", "driver_age + I(2 * driver_age) + municipality_type", "gaussian")
    assert np.isnan(m.fit.coef).sum() == 1  # duplicated term -> aliased -> NA


def test_glm_errors(d):
    with pytest.raises(ValueError, match="Tweedie"):
        build_glm(d, "total_premium", "N", "driver_age", "tweedie", var_power=None)
    with pytest.raises(ValueError, match="response in formula"):
        build_glm(d, "total_premium", "N", "total_premium + driver_age", "gamma")
    with pytest.raises(ValueError, match="train_test"):
        build_glm(d.drop(columns="train_test"), "total_premium", "N", "driver_age", "gamma", data_to_use="Training")


def test_tweedie_fits(d):
    m = build_glm(
        d, "total_premium", "N", "I(np.minimum(10, age_driving_licence)) + municipality_type", "tweedie", var_power=1.6
    )
    assert m.fit.converged and np.isfinite(m.fit.deviance)


def test_ratebook_base_relativity_is_one(
    d,
    tmp_path,
):
    fs = load_specification(d, demo_spec_paths()["feature"], "feature")
    formula = (
        "I(np.minimum(10, age_driving_licence)) + municipality_type + I(np.isin(vehicle_brand, ['BMW', 'AUDI']) * 1)"
    )
    m = build_glm(d, "total_premium", "N", formula, "gamma")
    tabs = tabulate_model(d, m.predict, ["age_driving_licence", "municipality_type", "vehicle_brand"], fs)
    assert tabs["municipality_type"].set_index("level").loc["I", "relativity"] == pytest.approx(1.0)
    assert tabs["age_driving_licence"].set_index("level").loc[40.0, "relativity"] == pytest.approx(1.0)
    p = write_ratebook_excel(tabs, str(tmp_path / "rb.xlsx"), "t")
    import openpyxl

    assert "municipality_type" in openpyxl.load_workbook(p).sheetnames


def test_gbm_interaction_constraints_and_monotonicity():
    from daisy.gbm import make_interaction_constraints, convert_monotonicity

    ft = pd.DataFrame({"feature": ["a", "b", "c"], "interaction_grouping": ["g1", "g1", "g2"]})
    assert make_interaction_constraints(ft, ["g1"], ["a", "b", "c"]) == [[2], [0, 1]] or make_interaction_constraints(
        ft, ["g1"], ["a", "b", "c"]
    ) == [[0, 1], [2]]
    assert convert_monotonicity(["Increasing", "decreasing", "", "1", "-1"]) == [1, -1, 0, 1, -1]


def test_gbm_gamma_refuses_zero_responses_with_a_clear_message():
    rng = np.random.default_rng(3)
    n = 300
    d = pd.DataFrame({"driver_age": rng.uniform(17, 80, n), "N": np.ones(n), "train_test": rng.integers(0, 2, n)})
    d["claims_cost"] = np.where(rng.uniform(size=n) < 0.5, 0.0, rng.gamma(2.0, 300.0, n))
    with pytest.raises(ValueError, match="gamma objective needs a response above 0"):
        build_gbm(d, "claims_cost", "N", ["driver_age"], {"objective": "gamma", "num_iterations": 5})
    positive = d.assign(claims_cost=d["claims_cost"].replace(0.0, 1.0))
    assert (
        build_gbm(positive, "claims_cost", "N", ["driver_age"], {"objective": "gamma", "num_iterations": 5}) is not None
    )


def test_gbm_validation_dataset_matches_reference_parameters():
    import warnings

    pytest.importorskip("lightgbm")
    from daisy.gbm import build_gbm

    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "feature": rng.normal(size=100),
            "category": np.where(np.arange(100) % 2, "a", "b"),
            "target": rng.uniform(1, 2, size=100),
            "exposure": rng.uniform(0.5, 1, size=100),
            "train_test": np.arange(100) % 4 == 0,
        }
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = build_gbm(
            data,
            "target",
            "exposure",
            ["feature", "category"],
            {"objective": "poisson", "num_iterations": 4, "early_stopping_round": 0, "min_data_in_leaf": 5},
            shap_sample="No",
            num_threads=1,
        )

    assert not any("Overriding the parameters from Reference Dataset" in str(warning.message) for warning in captured)
    assert "test" in model.evaluation_log
    assert len(model.predictions) == len(data)


def test_gbm_end_to_end(d):
    pytest.importorskip("lightgbm")
    from daisy.gbm import build_gbm

    fs = load_specification(d, demo_spec_paths()["feature"], "feature")
    feats = ["driver_age", "vehicle_age", "total_claims", "municipality_type", "vehicle_brand"]
    m = build_gbm(
        d.iloc[:6000],
        "total_premium",
        "N",
        feats,
        dict(objective="gamma", num_iterations=60, early_stopping_round=10, num_leaves=4, min_data_in_leaf=50),
        feature_table=fs,
        shap_sample="All",
        num_threads=1,
    )
    assert m.predictions.shape[0] == 6000 and np.all(m.predictions > 0)
    assert m.SHAP.shape == (6000, len(feats) + 1)
    # SHAP additivity: exp(sum of SHAP incl. base) == prediction (log link, no offset)
    np.testing.assert_allclose(np.exp(m.SHAP.sum(axis=1)), m.predictions, rtol=1e-6)
    assert m.importances["Gain"].sum() == pytest.approx(100)


def test_lift_table_uses_aligned_train_test_rows():
    from types import SimpleNamespace
    from daisy.charts import lift_table_frame

    data = pd.DataFrame({"total_claims": [4, 1, 3, 2], "exposure": [2, 1, 1, 2], "train_test": [0, 0, 1, 1]})
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        response="total_claims", predictions=np.array([8, 2, 6, 2]), pred_rows=np.ones(4, dtype=bool)
    )
    table = lift_table_frame(data, model, "exposure", n_bins=2)

    assert table["Dataset"].tolist() == ["Train", "Train", "Test", "Test"]
    test = table[table["Dataset"] == "Test"].set_index("Bin")
    assert test.loc[1, ["Exposure", "Actual", "Predicted"]].tolist() == [2, 2, 2]  # lowest predicted rate first
    assert test.loc[2, ["Exposure", "Actual", "Predicted"]].tolist() == [1, 3, 6]
    assert test.loc[2, "A/E"] == pytest.approx(0.5)
    assert test.loc[2, "Lift"] == pytest.approx(1.8)  # top-bin actual rate 3 over overall 5/3
    assert table.groupby("Dataset")["Exposure"].sum().tolist() == [3, 3]


def test_lift_bins_have_roughly_equal_exposure_and_conserve_totals():
    from daisy.charts import lift_bins

    rng = np.random.default_rng(1)
    n = 5000
    exposure = rng.uniform(0.2, 1.0, n)
    predicted = exposure * rng.gamma(4.0, 0.02, n)
    actual = rng.poisson(predicted).astype(float)
    bins = lift_bins(actual, predicted, exposure, 20)

    assert len(bins) == 20 and bins["Bin"].tolist() == list(range(1, 21))
    assert bins["Exposure"].sum() == pytest.approx(exposure.sum())
    assert bins["Actual"].sum() == pytest.approx(actual.sum())
    assert bins["Exposure"].between(0.9 * exposure.sum() / 20, 1.1 * exposure.sum() / 20).all()
    assert bins["Predicted rate"].is_monotonic_increasing


def test_lorenz_gini_for_ranked_and_flat_actuals():
    from daisy.charts import lorenz_points

    exposure = np.ones(4)
    actual = np.array([0.0, 0.0, 1.0, 3.0])
    _, _, ranked = lorenz_points(actual, actual, exposure)  # predictions rank the rows perfectly
    _, _, flat = lorenz_points(np.ones(4), np.arange(4.0), exposure)  # everyone has the same actual
    assert ranked == pytest.approx(0.625)
    assert flat == pytest.approx(0.0)
    assert lorenz_points(np.zeros(3), np.ones(3), np.ones(3)) is None


def test_lift_chart_draws_train_and_test_panels_with_mean_line():
    from types import SimpleNamespace
    from daisy.charts import lift_chart_figure

    rng = np.random.default_rng(5)
    n = 4000
    data = pd.DataFrame({"total_claims": rng.poisson(0.1, n).astype(float), "train_test": rng.integers(0, 2, n)})
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        response="total_claims", predictions=np.full(n, 0.1) * rng.uniform(0.5, 1.5, n), pred_rows=np.arange(n)
    )
    figure = lift_chart_figure(data, model, "N", n_bins=10)
    names = [trace.name for trace in figure.data]

    assert names.count("Observed") == 2 and names.count("Predicted") == 2 and names.count("Exposure") == 2
    assert names.count("Portfolio mean") == 2
    assert [a.text.split(":")[0] for a in figure.layout.annotations] == ["Train", "Test"]
    assert "top-bin lift" in figure.layout.annotations[0].text


def test_gbm_performance_metrics_by_split():
    from types import SimpleNamespace
    from daisy.gbm import gbm_performance_metrics

    data = pd.DataFrame({"total_claims": [1, 2, 1, 4], "exposure": [1, 1, 1, 1], "train_test": [0, 0, 1, 1]})
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        response="total_claims", predictions=np.array([1, 2, 2, 2]), pred_rows=np.ones(4, dtype=bool)
    )
    metrics = gbm_performance_metrics(data, model, "exposure").set_index("Dataset")

    assert list(metrics.index) == ["Train", "Test"]
    assert metrics.loc["Train", "GINI"] == pytest.approx(1.0 / 6.0)
    assert metrics.loc["Train", "PoDE"] == pytest.approx(1.0)
    assert metrics.loc["Train", "RMSE"] == pytest.approx(0.0)
    assert metrics.loc["Test", "RMSE"] == pytest.approx(np.sqrt(2.5))
    assert metrics.loc["Test", "Within 20% (%)"] == pytest.approx(0.0)


def test_metrics_gini_matches_lorenz_gini_for_count_response():
    from daisy.charts import lorenz_points
    from daisy.gbm import prediction_performance_metrics

    rng = np.random.default_rng(2)
    n = 3000
    exposure = rng.uniform(0.1, 1.0, n)
    predicted = exposure * rng.gamma(4.0, 0.03, n)
    data = pd.DataFrame({"total_claims": rng.poisson(predicted).astype(float), "exposure": exposure})
    metrics = prediction_performance_metrics(data, "total_claims", predicted, "exposure")
    _, _, gini = lorenz_points(data["total_claims"].to_numpy(), predicted, exposure)

    assert metrics.loc[0, "GINI"] == pytest.approx(gini)


def test_glm_prior_weights_recover_the_poisson_rate():
    from daisy.glm import fit_irls, make_family

    rng = np.random.default_rng(7)
    n = 4000
    exposure = rng.uniform(0.2, 1.0, n)
    x = rng.normal(size=n)
    counts = rng.poisson(exposure * np.exp(-2.0 + 0.4 * x)).astype(float)
    X = pd.DataFrame({"(Intercept)": 1.0, "x": x})
    fit = fit_irls(X, counts / exposure, make_family("poisson"), prior_weights=exposure)

    np.testing.assert_allclose(fit.coef, [-2.0, 0.4], atol=0.1)
    assert fit.null_deviance > fit.deviance





def test_average_basis_gini_and_charts_match_total_basis():
    from types import SimpleNamespace
    from daisy.charts import lift_table_frame, lorenz_points
    from daisy.gbm import model_overview, prediction_performance_metrics

    rng = np.random.default_rng(4)
    n = 3000
    count = rng.integers(1, 6, n).astype(float)  # claim count (the weight)
    severity = rng.gamma(4.0, 250.0, n)  # average severity per claim
    prediction = severity * rng.uniform(0.7, 1.3, n)  # predicted severity
    total, predicted_total = severity * count, prediction * count
    data_total = pd.DataFrame({"amount": total, "count": count})
    data_average = pd.DataFrame({"amount": severity, "count": count})

    gini_total = prediction_performance_metrics(data_total, "amount", predicted_total, "count").loc[0, "GINI"]
    gini_average = prediction_performance_metrics(
        data_average, "amount", prediction, "count", weight_basis="average"
    ).loc[0, "GINI"]
    assert gini_average == pytest.approx(gini_total)
    assert gini_total == pytest.approx(lorenz_points(total, predicted_total, count)[2])

    model_total = SimpleNamespace(
        response="amount", predictions=predicted_total, pred_rows=np.arange(n), weight_basis="rate_weighted"
    )
    model_average = SimpleNamespace(
        response="amount", predictions=prediction, pred_rows=np.arange(n), weight_basis="average"
    )
    by_total = lift_table_frame(data_total, model_total, "count", n_bins=10)
    by_average = lift_table_frame(data_average, model_average, "count", n_bins=10)
    pd.testing.assert_frame_equal(by_total, by_average)

    overview_total = model_overview(data_total, "amount", predicted_total, "count")
    overview_average = model_overview(data_average, "amount", prediction, "count", weight_basis="average")
    # the total weight and the actual and predicted per unit of weight agree (total response differs with the
    # response's scale); the quantile rows describe row-level predictions on each model's own scale
    pd.testing.assert_frame_equal(overview_total.drop(index=1).iloc[:3], overview_average.drop(index=1).iloc[:3])


def test_weight_basis_helpers_and_rate_weighted_scale():
    from types import SimpleNamespace
    from daisy.gbm import gbm_scoring_offset, model_weight_basis
    from daisy.weights import WEIGHT_BASES, WEIGHT_BASIS_HELP, scale_basis, to_total_scale

    assert list(WEIGHT_BASES) == ["rate_weighted", "average"] and set(WEIGHT_BASIS_HELP) == set(WEIGHT_BASES)
    assert [scale_basis(b) for b in WEIGHT_BASES] == ["total", "average"]
    data = pd.DataFrame({"count": [1.0, 2.0, 4.0]})
    fitted = SimpleNamespace(weight="count", weight_basis="rate_weighted", init_score="no offset", link="log")
    np.testing.assert_allclose(to_total_scale(fitted, data, [10.0, 10.0, 10.0]), [10.0, 20.0, 40.0])
    assert model_weight_basis(fitted) == "total"  # response and stored predictions are totals
    assert gbm_scoring_offset(fitted, data) is None  # fitted per unit of weight: no log(weight) offset
    assert model_weight_basis(SimpleNamespace(weight_basis="average")) == "average"
    # the removed log-offset basis is refused rather than scored wrongly
    with pytest.raises(ValueError, match="no longer supported"):
        model_weight_basis(SimpleNamespace(weight="count", weight_basis="total"))


def test_glm_score_returns_totals_for_each_weight_basis():
    from types import SimpleNamespace
    from daisy.glm import GLMModel

    data = pd.DataFrame({"count": [1.0, 2.0, 4.0, 0.0]})

    def fake(basis):
        return SimpleNamespace(weight="count", weight_basis=basis, predict=lambda d: np.ones(len(d)))

    np.testing.assert_allclose(GLMModel.score(fake("rate_weighted"), data), [1.0, 2.0, 4.0, 0.0])  # rate 1 x weight
    np.testing.assert_allclose(GLMModel.score(fake("average"), data), [1.0, 1.0, 1.0, 1.0])  # per unit of weight


def test_glm_rejects_the_old_total_basis_and_fits_rate_weighted():
    from daisy.glm import build_glm

    rng = np.random.default_rng(11)
    n = 3000
    data = pd.DataFrame({"driver_age": rng.uniform(18, 80, n), "exposure": rng.uniform(0.2, 1.0, n)})
    data["total_claims"] = rng.poisson(data["exposure"] * np.exp(-2.5 + 0.01 * data["driver_age"])).astype(float)
    with pytest.raises(ValueError):
        build_glm(data, "total_claims", "exposure", "driver_age", "poisson", weight_basis="total")
    per_unit = build_glm(data, "total_claims", "exposure", "driver_age", "poisson", weight_basis="rate_weighted")
    # a Poisson model with an intercept reproduces the total claim count exactly
    assert per_unit.predictions.sum() == pytest.approx(data.total_claims.sum(), rel=1e-3)
    np.testing.assert_allclose(per_unit.score(data), per_unit.predictions)
    build_glm(data, "total_claims", "exposure", "driver_age", "gaussian", weight_basis="rate_weighted")  # any link
def test_all_metrics_agree_across_weight_bases_and_poisson_pode_is_exact():
    from daisy.gbm import prediction_performance_metrics

    rng = np.random.default_rng(0)
    n = 20000
    exposure = rng.uniform(0.05, 1.0, n)
    mu = exposure * np.exp(-2 + 0.8 * rng.normal(size=n))  # expected claim counts
    total_claims = rng.poisson(mu).astype(float)
    as_total = prediction_performance_metrics(
        pd.DataFrame({"y": total_claims, "w": exposure}), "y", mu, "w", objective="poisson"
    )
    as_rate = prediction_performance_metrics(
        pd.DataFrame({"y": total_claims / exposure, "w": exposure}),
        "y",
        mu / exposure,
        "w",
        objective="poisson",
        weight_basis="average",
    )
    pd.testing.assert_frame_equal(as_total, as_rate)

    def poisson_deviance(
        y,
        m,
    ):
        return 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / m), 0) - (y - m))

    null = exposure * total_claims.sum() / exposure.sum()
    exact = 1 - poisson_deviance(total_claims, mu) / poisson_deviance(total_claims, null)
    assert as_total.loc[0, "PoDE"] == pytest.approx(exact, abs=1e-9)


def test_first_modelled_feature_follows_the_drop_down_order():
    from daisy.app import _chart_feature_choices, _first_modelled_feature

    columns = ["driver_age", "vehicle_brand", "fuel_type", "policy_type"]
    assert (
        _first_modelled_feature(columns, ["policy_type", "vehicle_brand"]) == "policy_type"
    )  # the most important modelled feature
    assert _first_modelled_feature(columns, None) is None and _first_modelled_feature(columns, []) is None
    assert _first_modelled_feature(columns, ["not_a_column"]) is None
    assert next(iter(_chart_feature_choices(columns, ["policy_type", "vehicle_brand"])["Modelled"])) == "policy_type"


def test_one_way_summary_treats_per_unit_columns_like_totals():
    from daisy.summary import one_way_summary

    rng = np.random.default_rng(1)
    n = 5000
    exposure = rng.uniform(0.1, 1.0, n)
    group = rng.integers(0, 3, n)
    mu = exposure * np.where(group == 0, 0.05, np.where(group == 1, 0.10, 0.20))
    total_claims = rng.poisson(mu).astype(float)
    totals = pd.DataFrame({"g": group, "e": exposure, "total_claims": total_claims, "pred": mu})
    per_unit = pd.DataFrame({"g": group, "e": exposure, "total_claims": total_claims / exposure, "pred": mu / exposure})

    expected = one_way_summary(totals, "total_claims", "e", "g", ["pred"])
    wrong = one_way_summary(per_unit, "total_claims", "e", "g", ["pred"])
    fixed = one_way_summary(per_unit, "total_claims", "e", "g", ["pred"], per_unit_cols=["total_claims", "pred"])

    pd.testing.assert_frame_equal(fixed, expected)
    assert not np.allclose(
        wrong["total_claims"], expected["total_claims"], rtol=0.05
    )  # the old behaviour was off by ~2x
    # a total response with only the prediction column per unit
    mixed = per_unit.assign(total_claims=total_claims)
    pd.testing.assert_frame_equal(
        one_way_summary(mixed, "total_claims", "e", "g", ["pred"], per_unit_cols={"pred"}), expected
    )
    # weight N: per unit and total are the same thing
    no_weight = one_way_summary(totals, "total_claims", "N", "g", ["pred"])
    pd.testing.assert_frame_equal(
        one_way_summary(totals, "total_claims", "N", "g", ["pred"], per_unit_cols=["total_claims"]), no_weight
    )


def _lift_importance_models():
    from types import SimpleNamespace

    rng = np.random.default_rng(8)
    n = 600
    data = pd.DataFrame(
        {
            "driver_age": rng.uniform(18, 80, n),
            "vehicle_brand": rng.choice(["a", "b"], n),
            "total_claims": rng.poisson(0.1, n).astype(float),
            "exposure": rng.uniform(0.3, 1.0, n),
            "train_test": np.arange(n) % 4 == 0,
        }
    )
    predictions = data["exposure"].to_numpy() * rng.uniform(0.05, 0.15, n)
    gbm = SimpleNamespace(
        weight_basis="rate_weighted",
        name="GBM 1",
        response="total_claims",
        predictions=predictions,
        pred_rows=np.arange(n),
        importances=pd.DataFrame(
            {"Feature": ["driver_age", "vehicle_brand"], "Gain": [70.0, 30.0], "Split": [60.0, 40.0]}
        ),
    )
    glm = SimpleNamespace(
        weight_basis="rate_weighted",
        name="GLM 1",
        response="total_claims",
        predictions=predictions,
        pred_rows=np.arange(n),
        feature_contributions=lambda frame: pd.DataFrame(
            {"glm_LP_age": np.full(len(frame), 0.2), "glm_LP_make": -np.full(len(frame), 0.05)}
        ),
        term_importance=lambda frame: pd.Series({"age": 20.0, "make": 5.0}),
    )
    return data, gbm, glm


def test_export_orders_metrics_before_lines_and_lift_then_importance_after_ale():
    from daisy.report import export_line_bar_plots, importance_section, lift_section, metrics_section, overview_section

    data, gbm, _ = _lift_importance_models()
    n = len(data)
    ones = np.ones(n, dtype=np.int8)
    tt = data["train_test"].to_numpy().astype(np.int8)
    masks = {"All": ones, "Train": (1 - tt).astype(np.int8), "Test": tt}
    overview = pd.DataFrame({"Statistic": ["Total Weight"], "Train": [1.0], "Test": [2.0]})
    metrics_for = lambda mask: metrics_section(pd.DataFrame({"Dataset": ["x"], "GINI": [float(mask.sum())]}))
    html = export_line_bar_plots(
        data,
        ["driver_age"],
        ["driver_age"],
        pd.DataFrame(columns=["feature", "banding"]),
        "total_claims",
        "exposure",
        [],
        ones,
        "0",
        False,
        False,
        valid_features=["driver_age"],
        view_masks=masks,
        leading_sections=[overview_section(overview), metrics_for],
        extra_sections=[
            lambda mask: lift_section(data, gbm, "exposure", mask, 10),
            importance_section(data, gbm, "gbm"),
        ],
    ).decode()
    sections = _export_payload(html)["sections"]
    assert [section["id"] for section in sections] == ["overview", "metrics", "lines", "lift", "importance"]
    assert [section["title"] for section in sections][3:] == ["Lift Chart", "Feature importance"]

    lift = sections[3]
    entry = lift["groups"][0]["features"][0]
    assert lift["single"] and entry["error"] is None and len(entry["figs"]) == 2
    assert "Actual vs expected by bin" in entry["html"] and "Train" in entry["html"]
    assert (
        set(entry["views"]) == {"Train", "Test"} and "Test" not in entry["views"]["Train"]["html"].split("<tbody>")[1]
    )
    importance = sections[4]["groups"][0]["features"][0]
    assert importance["error"] is None and len(importance["figs"]) == 1 and "views" not in sections[4]
    assert "driver_age" in importance["html"] and "70.0000" in importance["html"]


def test_importance_section_for_a_glm_uses_mean_absolute_contribution():
    import json
    from daisy.report import importance_section

    data, _, glm = _lift_importance_models()
    entry = importance_section(data, glm, "glm")["groups"][0]["features"][0]
    assert entry["error"] is None and "mean |LP| x 100" in entry["html"]
    assert "20.0000" in entry["html"] and "5.0000" in entry["html"] and "glm_LP_" not in entry["html"]
    assert json.loads(entry["figs"][0]["fig"])["layout"]["xaxis"]["title"]["text"] == "mean |LP| x 100"


def test_ratebook_scoring_returns_totals_for_total_and_per_unit_models(tmp_path):
    pytest.importorskip("openpyxl")
    from daisy.ratebook import write_ratebook_excel
    from daisy.scoring import read_ratebook, score_ratebook

    tables = {
        "__base__": pd.DataFrame({"feature": ["driver_age"], "base_level": ["30"]}),
        "driver_age": pd.DataFrame(
            {"level": [20.0, 30.0, 40.0], "prediction": [0.05, 0.10, 0.20], "relativity": [0.5, 1.0, 2.0]}
        ),
    }
    tables["__base__"].attrs["base_prediction"] = 0.10
    data = pd.DataFrame({"driver_age": [20.0, 30.0, 45.0], "exposure": [1.0, 0.5, 2.0]})
    rates = [0.05, 0.10, 0.20]

    for basis, expected in (("total", [0.05, 0.05, 0.40]), ("rate_weighted", [0.05, 0.05, 0.40]), ("average", rates)):
        path = write_ratebook_excel(
            tables, str(tmp_path / f"{basis}.xlsx"), "GLM 1", weight_basis=basis, weight="exposure"
        )
        ratebook = read_ratebook(path)
        assert ratebook.weight_basis == basis and ratebook.weight == "exposure" and ratebook.model_name == "GLM 1"
        predictions, notes = score_ratebook(ratebook, data, "exposure")
        np.testing.assert_allclose(predictions, expected)
        assert any("multiplied by exposure" in n for n in notes) == (basis != "average")
    # no weight column chosen: per-unit rates and a note; a missing weight column is an error
    path = write_ratebook_excel(tables, str(tmp_path / "t.xlsx"), "GLM 1", weight_basis="total", weight="exposure")
    predictions, notes = score_ratebook(read_ratebook(path), data, "N")
    np.testing.assert_allclose(predictions, rates)
    assert any("no weight column" in n for n in notes)
    with pytest.raises(ValueError):
        score_ratebook(read_ratebook(path), data, "missing")
    # ratebooks written without the setting (older files, or a weight of N) are scored as before
    legacy = write_ratebook_excel(tables, str(tmp_path / "legacy.xlsx"), "GLM 1")
    book = read_ratebook(legacy)
    assert book.weight_basis is None and book.weight == "N"
    np.testing.assert_allclose(score_ratebook(book, data, "exposure")[0], rates)


def test_prediction_column_metrics_by_split():
    from daisy.gbm import prediction_performance_metrics

    data = pd.DataFrame(
        {
            "total_claims": [1, 2, 1, 4],
            "exposure": [1, 1, 1, 1],
            "incumbent_prediction": [1, 2, 2, 2],
            "train_test": [0, 0, 1, 1],
        }
    )
    metrics = prediction_performance_metrics(
        data, "total_claims", data["incumbent_prediction"].to_numpy(), "exposure"
    ).set_index("Dataset")

    assert list(metrics.index) == ["Train", "Test"]
    assert metrics.loc["Train", "RMSE"] == pytest.approx(0.0)
    assert metrics.loc["Test", "RMSE"] == pytest.approx(np.sqrt(2.5))


def test_prediction_metrics_and_lift_without_train_test():
    from types import SimpleNamespace
    from daisy.charts import lift_chart_figure, lift_table_frame, lorenz_figure
    from daisy.gbm import prediction_performance_metrics

    data = pd.DataFrame({"total_claims": [1, 2, 1], "exposure": [1, 1, 1], "prediction": [1, 2, 2]})
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        response="total_claims", predictions=np.array([1, 2, 2]), pred_rows=np.ones(3, dtype=bool), link="identity"
    )
    metrics = prediction_performance_metrics(data, "total_claims", model.predictions, "exposure", model.pred_rows)
    lift = lift_chart_figure(data, model, "exposure", n_bins=3)

    assert metrics["Dataset"].tolist() == ["All"]
    assert metrics.loc[0, "RMSE"] == pytest.approx(1 / np.sqrt(3))
    assert lift_table_frame(data, model, "exposure", n_bins=3)["Dataset"].unique().tolist() == ["All"]
    assert [a.text.split(":")[0] for a in lift.layout.annotations] == ["All"]
    assert [trace.name.split(" ")[0] for trace in lorenz_figure(data, model, "exposure").data] == ["Equality", "All"]


def test_prediction_metrics_and_lift_accept_glm_row_indices():
    from types import SimpleNamespace
    from daisy.charts import lift_chart_figure, lift_table_frame
    from daisy.gbm import prediction_performance_metrics

    data = pd.DataFrame({"total_claims": [10, 4, 8, 6], "exposure": [1, 0, 1, 1], "train_test": [0, 0, 1, 1]})
    pred_rows = np.array([0, 2, 3])
    model = SimpleNamespace(
        weight_basis="rate_weighted",
        response="total_claims", predictions=np.array([9, 7, 5]), pred_rows=pred_rows, link="identity"
    )
    metrics = prediction_performance_metrics(data, "total_claims", model.predictions, "exposure", pred_rows)
    lift = lift_chart_figure(data, model, "exposure", n_bins=2)

    assert metrics["Dataset"].tolist() == ["Train", "Test"]
    assert [a.text.split(":")[0] for a in lift.layout.annotations] == ["Train", "Test"]
    assert lift_table_frame(data, model, "exposure", n_bins=2)["Dataset"].unique().tolist() == ["Train", "Test"]


def test_prediction_export_frame_preserves_row_order_and_split(d):
    from daisy.app import _prediction_export_frame

    data = d.iloc[:3].copy()
    predictions = np.array([101.0, 202.0, 303.0])
    export = _prediction_export_frame(data, "total_premium", predictions)

    assert export["row_index"].tolist() == [0, 1, 2]
    assert export["train_test"].tolist() == data["train_test"].tolist()
    assert export["total_premium"].tolist() == data["total_premium"].tolist()
    assert export["prediction"].tolist() == predictions.tolist()


def test_gbm_metric_comparison_marks_directional_changes():
    from daisy.gbm import compare_gbm_metrics

    old = pd.DataFrame(
        {
            "Dataset": ["Test"],
            "GINI": [0.2],
            "PoDE": [0.8],
            "RMSE": [2.0],
            "Within 5% (%)": [60.0],
            "Within 10% (%)": [70.0],
            "Within 20% (%)": [80.0],
        }
    )
    new = pd.DataFrame(
        {
            "Dataset": ["Test"],
            "GINI": [0.3],
            "PoDE": [0.9],
            "RMSE": [1.5],
            "Within 5% (%)": [50.0],
            "Within 10% (%)": [70.0],
            "Within 20% (%)": [90.0],
        }
    )
    comparison = compare_gbm_metrics(new, old).set_index("Metric")

    assert comparison.loc["GINI", "Status"] == "Improved"
    assert comparison.loc["GINI", "Change"] == pytest.approx(50.0)
    assert comparison.loc["PoDE", "Status"] == "Improved"
    assert comparison.loc["PoDE", "Change"] == pytest.approx(0.1)
    assert comparison.loc["RMSE", "Status"] == "Improved"
    assert comparison.loc["RMSE", "Change"] == pytest.approx(-25.0)
    assert comparison.loc["Within 5% (%)", "Status"] == "Deteriorated"
    assert comparison.loc["Within 10% (%)", "Status"] == "Unchanged"
    assert comparison.loc["Within 20% (%)", "Status"] == "Improved"


def test_gini_percentage_change_is_undefined_from_zero_baseline():
    from daisy.gbm import compare_gbm_metrics

    old = pd.DataFrame({"Dataset": ["Test"], "GINI": [0.0]})
    new = pd.DataFrame({"Dataset": ["Test"], "GINI": [0.1]})
    comparison = compare_gbm_metrics(new, old)

    assert np.isnan(comparison.loc[0, "Change"])
    assert comparison.loc[0, "Status"] == "Improved"


def test_rmse_percentage_change_is_undefined_from_zero_baseline():
    from daisy.gbm import compare_gbm_metrics

    old = pd.DataFrame({"Dataset": ["Test"], "RMSE": [0.0]})
    new = pd.DataFrame({"Dataset": ["Test"], "RMSE": [0.1]})
    comparison = compare_gbm_metrics(new, old)

    assert np.isnan(comparison.loc[0, "Change"])
    assert comparison.loc[0, "Status"] == "Deteriorated"


def test_starting_tab_accepts_only_the_visible_tab_titles():
    pytest.importorskip("shiny")
    from daisy.app import _TAB_TITLES, make_app

    assert _TAB_TITLES == ("Data", "Charts", "Maps", "GBMs", "GLMs", "Scoring")
    data = pd.DataFrame({"total_premium": [1.0, 2.0], "driver_age": [20.0, 30.0]})
    for old_name in ("nonsense", None):
        with pytest.raises(ValueError, match="starting_tab must be one of"):
            make_app(data, starting_tab=old_name)


# ---------------------------------------------------------------- calculation audit
# PoDE objective, model weight, GBM offset level, zero-weight scoring, weighted AIC, null deviance with an
# offset, ratebook banding and the ratebook link check.
@pytest.fixture(scope="module")
def frequency():
    rng = np.random.default_rng(0)
    n = 20000
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.integers(0, 5, n).astype(float)})
    d["expo"] = rng.uniform(0.05, 1.0, n)
    d["total_claims"] = rng.poisson(np.exp(-2 + 0.3 * d.x1 + 0.1 * d.x2) * d.expo).astype(float)
    d["amount"] = np.where(
        d.total_claims > 0, rng.gamma(2.0 * np.maximum(d.total_claims, 1), np.exp(7 + 0.2 * d.x1) / 2.0), 0.0
    )
    d["train_test"] = rng.binomial(1, 0.2, n)
    return d


def test_model_metrics_use_the_models_objective_and_weight(frequency):
    severity = frequency[frequency.total_claims > 0].reset_index(drop=True)
    model = build_glm(severity, "amount", "total_claims", "x1 + x2", "gamma", weight_basis="rate_weighted")
    assert model_objective_power(model) == ("gamma", 1.5)
    metrics = model_performance_metrics(severity, model).set_index("Dataset")
    # PoDE on the gamma deviance equals 1 - deviance / null deviance of the fit (all rows were fitted)
    y, w = (severity.amount / severity.total_claims).to_numpy(), severity.total_claims.to_numpy()
    mu = model.predictions / w
    expected = 1 - np.average(unit_deviance("gamma", y, mu), weights=w) / np.average(
        unit_deviance("gamma", y, np.full_like(y, np.average(y, weights=w))), weights=w
    )
    both = model_performance_metrics(severity.assign(train_test=0), model).set_index("Dataset")
    assert both.loc["Train", "PoDE"] == pytest.approx(expected)
    assert both.loc["Train", "PoDE"] == pytest.approx(1 - model.fit.deviance / model.fit.null_deviance)
    # and it is not the Poisson figure the table used to show
    poisson = prediction_performance_metrics(
        severity.assign(train_test=0), "amount", model.predictions, "total_claims", model.pred_rows
    ).set_index("Dataset")
    assert abs(poisson.loc["Train", "PoDE"] - expected) > 1e-6
    assert set(metrics.index) == {"Train", "Test"}


def test_gbm_with_an_offset_starts_from_the_overall_level(frequency):
    frequency = frequency.assign(expected=frequency.expo)  # an expected total column (exposure here)
    model = build_gbm(
        frequency,
        "total_claims",
        "expo",
        ["x1", "x2"],
        dict(objective="poisson", num_iterations=20, early_stopping_round=0),
        init_score="expected",
        shap_sample="All",
    )
    train = (frequency.train_test == 0).to_numpy()
    assert model.offset_base == pytest.approx(np.log(frequency.total_claims[train].sum() / frequency.expo[train].sum()))
    # 20 trees at a 0.05 learning rate: balanced overall (it was several times too high before)
    assert model.predictions[train].sum() == pytest.approx(frequency.total_claims[train].sum(), rel=0.02)
    np.testing.assert_allclose(score_gbm(model, frequency), model.predictions)
    # SHAP values plus the base score still add up to the offset-free raw score
    np.testing.assert_allclose(
        model.SHAP.to_numpy().sum(axis=1), np.log(model.predict(frequency.iloc[model.SHAP_rows])), atol=1e-8
    )


def test_gamma_gbm_can_model_only_the_rows_with_a_positive_response(frequency):
    from daisy.gbm import NonPositiveResponseError

    severity = frequency.assign(amount=frequency.total_claims * 100.0 + 1.0)  # positive on every row
    severity.loc[severity.index[:5], "amount"] = 0.0
    params = dict(objective="gamma", num_iterations=5, early_stopping_round=0)
    with pytest.raises(NonPositiveResponseError) as error:
        build_gbm(severity, "amount", "expo", ["x1"], params, shap_sample="No", num_threads=1)
    assert error.value.count == 5
    severity.loc[severity.index[5], "expo"] = 0.0  # a zero weight: one more row left out, reported in the same count
    with pytest.raises(NonPositiveResponseError) as both:
        build_gbm(severity, "amount", "expo", ["x1"], params, shap_sample="No", num_threads=1)
    assert both.value.count == 6 and both.value.total == len(severity)
    model = build_gbm(severity, "amount", "expo", ["x1"], params, shap_sample="No", num_threads=1, positive_response=True)
    positive = ((severity.amount > 0) & (severity.expo > 0)).to_numpy()
    np.testing.assert_array_equal(model.pred_rows, positive)  # the data is not changed, the rows are only left out
    assert severity.amount.iloc[0] == 0.0 and severity.expo.iloc[5] == 0.0


def test_gamma_glm_can_model_only_the_rows_with_a_positive_response_and_weight(frequency):
    from daisy.glm import build_glm
    from daisy.weights import NonPositiveResponseError

    severity = frequency.assign(amount=frequency.total_claims * 100.0 + 1.0)
    severity.loc[severity.index[:5], "amount"] = 0.0
    severity.loc[severity.index[5], "expo"] = 0.0  # one more row left out: a zero weight, reported in the same count
    with pytest.raises(NonPositiveResponseError) as error:
        build_glm(severity, "amount", "expo", "x1", "gamma")
    assert error.value.count == 6 and error.value.total == len(severity)
    model = build_glm(severity, "amount", "expo", "x1", "gamma", positive_response=True)
    positive = ((severity.amount > 0) & (severity.expo > 0)).to_numpy()
    np.testing.assert_array_equal(model.pred_rows, np.flatnonzero(positive))  # GLM rows are positions
    assert severity.amount.iloc[0] == 0.0 and severity.expo.iloc[5] == 0.0  # the data is not changed


def test_shap_charts_skip_zero_weight_rows_instead_of_failing(frequency):
    from daisy.charts import shap_flame_figure, shap_heatmap_figure, stacked_shap_figure
    from daisy.gbm import build_gbm

    model = build_gbm(
        frequency,
        "total_claims",
        "expo",
        ["x1", "x2"],
        dict(objective="poisson", num_iterations=10, early_stopping_round=0),
        shap_sample="All",
        num_threads=1,
    )
    # a sidebar weight with zeros in the x1 < 0 band: that band has no weight above 0 to average over
    charted = frequency.assign(sidebar=np.where(frequency.x1 < 0, 0.0, 1.0))
    everything = np.ones(len(charted), dtype=bool)
    shap_flame_figure(charted, model, "x1", "sidebar", "0.5", everything)
    shap_heatmap_figure(charted, model, "x1", "x2", "sidebar", "0.5", "0.5", everything)
    stacked_shap_figure(charted, model, "x1", "sidebar", "0.5", everything)
    # no weight above 0 at all: an empty chart rather than an error
    none = shap_flame_figure(charted.assign(sidebar=0.0), model, "x1", "sidebar", "0.5", everything)
    assert "no rows with a weight above 0" in none.layout.title.text


def test_shap_charts_use_the_charts_minimum_maximum_and_bins(frequency):
    from daisy.charts import band_by_settings, shap_flame_figure
    from daisy.gbm import build_gbm

    model = build_gbm(
        frequency,
        "total_claims",
        "expo",
        ["x1", "x2"],
        dict(objective="poisson", num_iterations=10, early_stopping_round=0),
        shap_sample="All",
        num_threads=1,
    )
    everything = np.ones(len(frequency), dtype=bool)
    # a maximum caps values above it, so they join the last band
    capped = band_by_settings(frequency.x1, "0", None, 0.0, None)
    assert capped.max() <= 0.0
    # bins replace the banding: ordered labels, every row is in a bin
    binned = band_by_settings(frequency.x1, "0", None, None, 4)
    assert len(binned.cat.categories) == 4
    assert binned.notna().all()
    # the flame chart has one x position per band, in the Charts order
    figure = shap_flame_figure(frequency, model, "x1", "expo", "0", everything, maximum=0.0)
    assert len(figure.data[0].x) == capped.nunique()


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_models_refuse_a_weight_with_zero_or_negative_values(frequency, bad_value):
    with_bad = frequency.assign(expo=frequency.expo.where(frequency.index != 0, bad_value))
    with pytest.raises(ValueError, match="0 or less"):
        build_glm(with_bad, "total_claims", "expo", "x1", "poisson")
    with pytest.raises(ValueError, match="0 or less"):
        build_gbm(with_bad, "total_claims", "expo", ["x1"], dict(objective="poisson", num_iterations=2))
    build_glm(frequency, "total_claims", "expo", "x1", "poisson")  # all positive: builds


def test_rate_weighted_models_score_zero_weight_rows_as_zero(frequency):
    np.testing.assert_allclose(
        zero_weight_predictions([1.0, 2.0, 3.0, 4.0], [0.0, 2.0, np.nan, -1.0]), [0.0, 2.0, np.nan, np.nan]
    )
    glm = build_glm(frequency, "total_claims", "expo", "x1 + x2", "poisson")
    gbm = build_gbm(
        frequency,
        "total_claims",
        "expo",
        ["x1", "x2"],
        dict(objective="poisson", num_iterations=10, early_stopping_round=0),
        shap_sample="No",
    )
    new = frequency.head(4).copy()
    new["expo"] = [0.0, 0.5, np.nan, 1.0]
    for scored in (glm.score(new), score_gbm(gbm, new)):
        assert scored[0] == 0.0 and np.isnan(scored[2]) and scored[1] > 0 and scored[3] > 0
    assert glm.score(new)[1] == pytest.approx(0.5 * glm.predict(new)[1])


def test_aic_uses_prior_weights(frequency):
    severity = frequency[frequency.total_claims > 0].reset_index(drop=True)
    model = build_glm(severity, "amount", "total_claims", "x1 + x2", "gamma", weight_basis="rate_weighted")
    y, w = (severity.amount / severity.total_claims).to_numpy(), severity.total_claims.to_numpy()
    mu = model.predictions / w
    dispersion = model.fit.deviance / w.sum()
    expected = -2 * (w * stats.gamma.logpdf(y, a=1 / dispersion, scale=mu * dispersion)).sum() + 2 + 2 * 3
    assert model.fit.aic == pytest.approx(expected)
    # Gaussian with prior weights: standard formula, including - sum(log(weights))
    gaussian = build_glm(frequency, "amount", "expo", "x1 + x2", "gaussian", weight_basis="average")
    n, wt = len(frequency), frequency.expo.to_numpy()
    assert gaussian.fit.aic == pytest.approx(
        n * (np.log(2 * np.pi * gaussian.fit.deviance / n) + 1) + 2 - np.log(wt).sum() + 2 * 3
    )
    # a Poisson with exposure: AIC on the totals (claim counts with mean = predicted total)
    freq = build_glm(frequency, "total_claims", "expo", "x1", "poisson", weight_basis="rate_weighted")
    rank = int(np.sum(~np.isnan(freq.fit.coef)))
    counts = frequency.total_claims.to_numpy(float)[freq.pred_rows]
    expected_aic = -2 * stats.poisson.logpmf(counts, freq.predictions).sum() + 2 * rank
    assert freq.fit.aic == pytest.approx(expected_aic, rel=1e-9)
    # no likelihood: a weighted binomial
    hit = frequency.assign(hit=(frequency.total_claims > 0).astype(float))
    assert np.isnan(build_glm(hit, "hit", "expo", "x1", "binomial", weight_basis="average").fit.aic)


def test_null_deviance_with_prior_weights_is_the_intercept_only_fit(frequency):
    severity = frequency[frequency.total_claims > 0].reset_index(drop=True)
    model = build_glm(severity, "amount", "total_claims", "x1 + x2", "gamma", weight_basis="rate_weighted")
    # gamma with prior weights w on the rate y / w: the intercept-only rate is sum(y) / sum(w)
    y, w = severity.amount.to_numpy(), severity.total_claims.to_numpy()
    rate = y / w
    null_rate = np.sum(y) / np.sum(w)
    assert model.fit.null_deviance == pytest.approx(np.sum(w * unit_deviance("gamma", rate, null_rate)), rel=1e-8)


def test_ratebook_grid_keeps_the_specified_banding():
    d = pd.DataFrame({"cc": np.arange(0, 6000, 7.0), "vehicle_value": np.linspace(0, 95000, 858)})
    grid = _grid(d, "cc", {"min": "0", "max": "10000", "banding": "50"})
    assert len(grid) == 201 and grid[1] - grid[0] == 50 and grid[-1] == 10000
    grid = _grid(d, "vehicle_value", {"min": "0", "max": "100000", "banding": "200"})
    assert len(grid) == 501 and grid[1] - grid[0] == 200
    # beyond the cap the bands widen by a whole multiple of the banding
    grid = _grid(d, "vehicle_value", {"min": "0", "max": "100000", "banding": "10"})
    assert len(grid) <= 2000 and (grid[1] - grid[0]) % 10 == 0


def test_ratebook_needs_a_log_link():
    check_ratebook_link("log")
    check_ratebook_link(None)
    with pytest.raises(ValueError, match="log-link"):
        check_ratebook_link("identity")
    d = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="log-link"):
        tabulate_model(d, lambda frame: np.ones(len(frame)), ["x"], pd.DataFrame({"feature": ["x"]}), link="logit")


def test_importance_chart_shows_every_feature():
    from daisy.charts import importance_figure, importance_height
    from daisy.report import importance_section

    names = [f"a_long_feature_name_{i:02d}" for i in range(60)]
    imp = pd.DataFrame({"Feature": names, "Gain": np.linspace(10, 0, 60), "Split": np.arange(60)})
    figure = importance_figure(imp)
    assert list(figure.data[0].y) == names[::-1]  # all 60, most important at the top
    assert list(figure.layout.yaxis.tickvals) == names[::-1]  # and every bar keeps its label
    assert figure.layout.yaxis.automargin is True
    assert importance_height(60) > importance_height(25) > importance_height(1) >= 260
    assert len(importance_figure(imp, top=10).data[0].y) == 10

    class Model:
        name, importances = "GBM 1", imp

    section = importance_section(pd.DataFrame(), Model(), "gbm")
    exported = section["groups"][0]["features"][0]
    assert exported["error"] is None and names[-1] in str(exported)


# ---------------------------------------------------------------- minimum / maximum and bins
def test_bins_are_labelled_intervals_in_order():
    from daisy.banding import apply_treatment, bin_count, resolve_treatment

    values = pd.Series([0, 4.9, 5, 9, 10, 14, 15, 20, 150, np.nan])
    treatment = resolve_treatment(values, 0, 20, 4)
    assert treatment["labels"] == ["[0,5)", "[5,10)", "[10,15)", "[15,Inf)"]
    binned = apply_treatment(values, treatment)
    assert binned.cat.ordered and list(binned.cat.categories) == treatment["labels"]
    assert list(binned.astype(object)[:9]) == [
        "[0,5)",
        "[0,5)",
        "[5,10)",
        "[5,10)",
        "[10,15)",
        "[10,15)",
        "[15,Inf)",
        "[15,Inf)",
        "[15,Inf)",
    ]
    assert binned.isna().tolist() == [False] * 9 + [True]
    assert apply_treatment(binned, treatment) is binned  # already binned: left alone
    # without a minimum or maximum the bins cover the data, and other data is held to that range
    open_ended = resolve_treatment(pd.Series([0, 20]), None, None, 4)
    assert open_ended["labels"] == treatment["labels"]
    assert list(apply_treatment(pd.Series([-5.0, 12, 400]), open_ended).astype(str)) == ["[0,5)", "[10,15)", "[15,Inf)"]
    assert resolve_treatment(pd.Series([0.1, 0.9]), 0, 1, 2)["labels"] == ["[0,0.5)", "[0.5,Inf)"]
    assert resolve_treatment(pd.Series([7, 7]), None, None, 3)["labels"] == ["[7,Inf)"]  # nothing to split
    # capping only
    assert apply_treatment(values, resolve_treatment(values, 5, 15)).tolist()[:9] == [5, 5, 5, 9, 10, 14, 15, 15, 15]
    assert resolve_treatment(values) is None and resolve_treatment(values, bins=0) is None
    assert bin_count("") is None and bin_count("0") is None and bin_count("4") == 4
    with pytest.raises(ValueError, match="whole number"):
        resolve_treatment(values, 0, 20, 2.5)
    with pytest.raises(ValueError, match="minimum must be below"):
        resolve_treatment(values, 100, 1)
    with pytest.raises(ValueError, match="more than"):
        resolve_treatment(values, None, None, 100000)


def test_one_way_summary_minimum_maximum_and_bins():
    frame = pd.DataFrame({"driver_age": np.arange(1, 131), "y": 1.0})
    capped = one_way_summary(frame, "y", "N", "driver_age", banding="10", minimum=20, maximum=100)
    assert capped["driver_age"].tolist() == [20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert capped["count"].tolist() == [29, 10, 10, 10, 10, 10, 10, 10, 31]
    bins = one_way_summary(frame, "y", "N", "driver_age", banding="0", minimum=0, maximum=100, bins=10)
    assert bins["driver_age"].tolist() == [
        "[0,10)",
        "[10,20)",
        "[20,30)",
        "[30,40)",
        "[40,50)",
        "[50,60)",
        "[60,70)",
        "[70,80)",
        "[80,90)",
        "[90,Inf)",
    ]  # in bin order, not alphabetical
    assert bins["count"].tolist() == [9] + [10] * 8 + [41]
    assert capped["count"].sum() == bins["count"].sum() == 130
    four = one_way_summary(frame.assign(driver_age=frame.driver_age % 21), "y", "N", "driver_age", bins=4)
    assert four["driver_age"].tolist() == ["[0,5)", "[5,10)", "[10,15)", "[15,Inf)"]
    unchanged = one_way_summary(frame, "y", "N", "driver_age", banding="10")
    assert unchanged["driver_age"].tolist() == list(range(0, 131, 10))


def test_feature_specification_records_the_treatment():
    from daisy.banding import treatments_from_spec
    from daisy.specs import feature_treatment_spec, set_feature_treatment

    spec = pd.DataFrame(
        {
            "feature": ["driver_age", "vehicle_brand"],
            "base_level": "",
            "min": ["17", ""],
            "max": ["100", ""],
            "banding": ["1", ""],
            "interaction_grouping": "",
        }
    )
    assert feature_treatment_spec(spec, "driver_age") is None  # min / max alone do not change the modelling data
    columns = ["min", "max", "banding", "bins", "banding_type"]
    updated = set_feature_treatment(spec, "driver_age", 20, 80, 0, bins=6)
    assert updated.loc[0, columns].tolist() == ["20", "80", "", "6", "bins"]
    assert updated.loc[1, "banding_type"] == "" and "banding_type" not in spec.columns
    assert feature_treatment_spec(updated, "driver_age") == {"min": 20.0, "max": 80.0, "bins": 6}
    capped = set_feature_treatment(updated, "driver_age", "", 80, 5)
    assert capped.loc[0, columns].tolist() == ["", "80", "5", "", "capped"]
    assert feature_treatment_spec(capped, "driver_age") == {"min": None, "max": 80.0, "bins": None}
    cleared = set_feature_treatment(capped, "driver_age", None, None, 5)
    assert cleared.loc[0, "banding_type"] == "" and feature_treatment_spec(cleared, "driver_age") is None
    added = set_feature_treatment(spec, "vehicle_value", 0, None, 100)
    assert (
        added["feature"].tolist() == ["driver_age", "vehicle_brand", "vehicle_value"]
        and added.loc[2, "banding_type"] == "capped"
    )
    # a spec written to csv and read back gives the same treatments
    data = pd.DataFrame({"driver_age": [18.0, 25, 44, 95], "vehicle_brand": list("abcd")})
    reread = pd.read_csv(io.StringIO(updated.to_csv(index=False)), dtype=str, keep_default_na=False)
    treatments = treatments_from_spec(data, reread, ["driver_age", "vehicle_brand", "missing"])
    assert list(treatments) == ["driver_age"] and treatments["driver_age"]["labels"][0] == "[20,30)"


def test_models_apply_their_treatments_to_any_data(d):
    from daisy.banding import resolve_treatment
    from daisy.gbm import build_gbm, score_gbm

    treatments = {
        "driver_age": resolve_treatment(d.driver_age, 20, 80, 6),
        "vehicle_age": resolve_treatment(d.vehicle_age, None, 5),
    }
    spec = pd.DataFrame({"feature": ["driver_age", "vehicle_age"], "monotonicity": ["decreasing", "decreasing"]})
    gbm = build_gbm(
        d,
        "total_premium",
        "N",
        ["driver_age", "vehicle_age", "vehicle_brand"],
        dict(objective="gamma", num_iterations=20, early_stopping_round=0),
        shap_sample="No",
        treatments=treatments,
        feature_table=spec,
    )
    assert "driver_age" in gbm.categories and gbm.params["monotone_constraints"] == [0, -1, 0]  # no order to constrain
    glm = build_glm(d, "total_premium", "N", "driver_age + vehicle_age + vehicle_brand", "gamma", treatments=treatments)
    assert "driver_age[T.[30,40)]" in glm.fit.columns and "driver_age[T.[70,Inf)]" in glm.fit.columns
    raw = d.head(6).copy()
    raw["driver_age"], raw["vehicle_age"] = [17, 20, 29, 30, 85, 200], 9  # outside the caps, and never seen in training
    same = raw.copy()
    same["driver_age"], same["vehicle_age"] = [20, 25, 21, 35, 70, 75], 5  # the same bins and caps
    np.testing.assert_allclose(score_gbm(gbm, raw), score_gbm(gbm, same))
    np.testing.assert_allclose(glm.score(raw), glm.score(same))
    assert np.isfinite(glm.score(raw)).all()
    np.testing.assert_allclose(score_gbm(gbm, d), gbm.predictions)
    # a ratebook varies the stored (numeric) feature, so its levels stay numeric band starts
    table = tabulate_model(
        d,
        lambda frame: gbm.predict(frame),
        ["driver_age"],
        pd.DataFrame({"feature": ["driver_age"], "min": ["20"], "max": ["80"], "banding": ["10"]}),
    )
    assert pd.api.types.is_numeric_dtype(table["driver_age"]["level"])
    untreated = build_glm(d, "total_premium", "N", "driver_age + vehicle_age + vehicle_brand", "gamma")
    assert untreated.design.treatments == {} and "driver_age" in untreated.fit.columns


def test_export_uses_the_chart_limits_and_specification_bins():
    from daisy.report import export_line_bar_plots

    data = pd.DataFrame({"driver_age": np.arange(1, 131.0), "km": np.arange(130.0) * 100, "total_premium": 1.0})
    spec = pd.DataFrame(
        {
            "feature": ["driver_age", "km"],
            "min": ["", "0"],
            "max": ["", "5000"],
            "banding": ["10", ""],
            "bins": ["", "5"],
            "banding_type": ["", "bins"],
        }
    )
    html = export_line_bar_plots(
        data,
        list(data.columns),
        None,
        spec,
        "total_premium",
        "N",
        [],
        np.ones(130),
        "0",
        True,
        False,
        selected_feature="driver_age",
        selected_banding="0",
        selected_limits=(0, 100, 4),
    ).decode()
    payload = json.dumps(_export_payload(html))
    assert "[25,50)" in payload and "[75,Inf)" in payload  # the selected feature: the Charts settings
    assert "[1000,2000)" in payload and "[4000,Inf)" in payload  # another feature: what the specification applies


# ---------------------------------------------------------------- monotonicity, importance order, report views
def test_monotonicity_is_saved_to_the_specification():
    from daisy.specs import feature_monotonicity, set_feature_monotonicity

    spec = pd.DataFrame(
        {"feature": ["driver_age", "vehicle_brand"], "monotonicity": ["", ""], "interaction_grouping": ["", ""]}
    )
    updated = set_feature_monotonicity(spec, "driver_age", "decreasing")
    assert updated.loc[0, "monotonicity"] == "decreasing" and updated.loc[1, "monotonicity"] == ""
    assert (
        feature_monotonicity(updated, "driver_age") == "decreasing"
        and feature_monotonicity(updated, "vehicle_brand") == ""
    )
    assert feature_monotonicity(set_feature_monotonicity(updated, "driver_age", "none"), "driver_age") == ""
    added = set_feature_monotonicity(spec.drop(columns="monotonicity"), "vehicle_value", "increasing")
    assert added.loc[added["feature"] == "vehicle_value", "monotonicity"].iloc[0] == "increasing"
    assert feature_monotonicity(None, "driver_age") == ""


def test_report_importance_is_full_height_and_two_way_views_are_included(d):
    from daisy import report
    from daisy.charts import importance_height
    from daisy.gbm import build_gbm

    gbm = build_gbm(
        d,
        "total_premium",
        "N",
        ["driver_age", "vehicle_age", "vehicle_brand"],
        dict(objective="gamma", num_iterations=20, early_stopping_round=0),
        shap_sample="All",
        num_threads=1,
    )
    ranked = report.importance_frame(d, gbm, "gbm")
    assert ranked["Gain"].is_monotonic_decreasing  # most important first
    section = report.importance_section(d, gbm, "gbm")
    figure = json.loads(section["groups"][0]["features"][0]["figs"][0]["fig"])
    assert figure["layout"]["height"] == importance_height(len(ranked))  # every feature gets room for its label
    shap = report.shap_section(
        d,
        gbm,
        ["driver_age", "vehicle_age"],
        pd.DataFrame(columns=["feature"]),
        "N",
        np.ones(len(d), dtype=bool),
        pairs=[("driver_age", "vehicle_age")],
    )
    assert [g["id"] for g in shap["groups"]] == ["shap", "shap-two-way"]
    two_way = shap["groups"][1]["features"]
    assert [entry["name"] for entry in two_way] == ["driver_age x vehicle_age"] and not two_way[0].get("error")
    ale = report.ale_section(
        d,
        gbm,
        "gbm",
        ["driver_age", "vehicle_age"],
        np.ones(len(d), dtype=bool),
        bins=5,
        max_rows=2000,
        pairs=[("driver_age", "vehicle_age")],
    )
    assert [g["id"] for g in ale["groups"]] == ["ale", "ale-two-way"]
    assert [entry["name"] for entry in ale["groups"][1]["features"]] == ["driver_age x vehicle_age"]
    assert not ale["groups"][1]["features"][0].get("error")


def test_error_bar_tickbox_and_total_response_in_the_report():
    from daisy.report import render_report_html

    data = pd.DataFrame({"driver_age": np.arange(1, 61.0), "total_premium": 1.0, "N_exp": 2.0})
    from daisy.report import line_bar_section

    section = line_bar_section(
        data,
        ["driver_age"],
        None,
        pd.DataFrame(columns=["feature"]),
        "total_premium",
        "N_exp",
        [],
        np.ones(60),
        "0",
        True,
        False,
    )
    fig = json.loads(section["groups"][0]["features"][0]["figs"][0]["fig"])
    assert any("error_y" in trace for trace in fig["data"])  # the bars are there for the tickbox
    html = render_report_html([section], "total_premium", "N_exp", "test", error_bars=False).decode()
    assert 'id="errbars"' in html and '"errorBars": false' in html  # unticked when the app's Sigma bars are off
    assert (
        '"errorBars": true' in render_report_html([section], "total_premium", "N_exp", "test", error_bars=True).decode()
    )
    from daisy.gbm import OVERVIEW_STATISTICS, model_overview

    assert OVERVIEW_STATISTICS[:2] == ["Total Weight", "Total Response"]
    assert "Total Response" in model_overview(data, "total_premium", np.ones(60), "N_exp").iloc[:, 0].tolist()


def test_gbm_parameters_have_hover_help():
    from daisy.gbm import PARAMETER_HELP, _PARAMETER_ORDER, parameter_help
    from daisy.report import text_table_html

    for name in _PARAMETER_ORDER + [
        "weight used as",
        "offset (init_score)",
        "starting level (link scale)",
        "number_of_features",
        "interaction_groups",
        "shap_sample",
        "seed",
    ]:
        assert PARAMETER_HELP.get(name), name
    assert parameter_help("learning_rate").startswith("How much each tree adds")
    assert parameter_help("some_unknown_setting").startswith("A LightGBM setting")
    table = pd.DataFrame({"Parameter": ["learning_rate", "num_leaves"], "Value": ["0.05", "5"]})
    html = text_table_html(table, hover=parameter_help)
    assert 'title="How much each tree adds' in html and "<td>learning_rate</td>" not in html
    assert ">learning_rate</td>" in html and text_table_html(table).count("title=") == 0


# ---------------------------------------------------------------- adjustment overlays
def test_overlay_rule_adjusts_reviews_and_keeps_bands():
    from daisy.overlay import band_table, bootstrap_intervals, suggest

    rng = np.random.default_rng(5)
    keys = np.repeat([0.0, 10.0, 20.0], [4000, 4000, 60])
    expected = np.ones(len(keys))
    actual = np.concatenate([rng.poisson(1.20, 4000), rng.poisson(1.00, 4000), rng.poisson(1.00, 60)]).astype(float)
    table = band_table(actual, expected, keys)
    intervals = bootstrap_intervals(actual, expected, keys, samples=300)
    result = suggest(table, intervals)
    assert result.loc[0.0, "status"] == "Adjust" and result.loc[0.0, "relativity"] == pytest.approx(1.2, abs=0.05)
    assert result.loc[10.0, "status"] in ("Keep at 1", "Review") and result.loc[10.0, "relativity"] == 1.0
    assert result.loc[20.0, "status"] != "Adjust" and result.loc[20.0, "relativity"] == 1.0  # small band: not adjusted


def test_overlay_applies_relativities_by_band_and_keeps_others_at_one():
    from daisy.overlay import Overlay, apply_overlay, band_starts, check_relativities, evaluate

    data = pd.DataFrame({"driver_age": [17.0, 21.0, 29.0, 33.0, np.nan]})
    overlay = Overlay("driver_age overlay", "GLM 1", "driver_age", 5.0, {20.0: 1.1, 30.0: 0.9})
    expected = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    adjusted = apply_overlay(overlay, data, expected)
    assert list(adjusted) == pytest.approx([10.0, 11.0, 10.0, 9.0, 10.0])  # 17 and 29 are not listed: 1; blank: 1
    assert list(band_starts(data["driver_age"], 5.0)[:4]) == [15.0, 20.0, 25.0, 30.0]
    assert check_relativities({20.0: 1.1, 30.0: 0.0}) and not check_relativities({20.0: 1.1})
    actual = np.array([10.0, 11.0, 12.0, 9.0, 10.0])
    table, summary = evaluate(overlay, data, actual, expected)
    assert summary["A/E total adjusted"] == pytest.approx(actual.sum() / adjusted.sum())
    assert summary["RMSE adjusted"] < summary["RMSE base"]
    assert "A/E adjusted" in table.columns


def test_overlay_categorical_levels_are_bands():
    from daisy.overlay import (
        Overlay,
        apply_overlay,
        band_label,
        band_table,
        bootstrap_intervals,
        category_levels,
        suggest,
    )

    data = pd.DataFrame({"region": ["north", "south", None, "east", "west"]})
    levels = category_levels(data["region"])
    assert levels == ["east", "north", "south", "west"]
    overlay = Overlay("region overlay", "GLM 1", "region", 0.0, {0.0: 1.2, 2.0: 0.8}, levels=levels)
    expected = np.full(5, 10.0)
    adjusted = apply_overlay(overlay, data, expected)
    # north -> 1 (not listed), south -> 0.8, blank -> 1, east -> 1.2, west -> 1 (not listed)
    assert list(adjusted) == pytest.approx([10.0, 8.0, 10.0, 12.0, 10.0])
    unseen = apply_overlay(overlay, pd.DataFrame({"region": ["south", "mars"]}), np.array([10.0, 10.0]))
    assert list(unseen) == pytest.approx([8.0, 10.0])
    assert [band_label(i, 0.0, levels) for i in range(4)] == levels
    actual = np.array([10.0, 11.0, 12.0, 9.0, 10.0])
    keys = np.array([1.0, 0.0, 2.0, 2.0, 3.0])  # north, east, south, south, west
    suggestion = suggest(band_table(actual, expected, keys), bootstrap_intervals(actual, expected, keys, samples=200))
    assert set(suggestion.index) == {0.0, 1.0, 2.0, 3.0}


def test_overlay_adjustments_export_one_tab_per_feature():
    import io
    from daisy.overlay import Overlay, adjustments_workbook

    numeric = Overlay("driver_age overlay", "GLM 1", "driver_age", 5.0, {20.0: 1.1, 30.0: None}, training_rows=100)
    region = Overlay("region overlay", "GBM 1", "region", 0.0, {0.0: 1.2}, training_rows=100, levels=["east", "north"])
    sheets = pd.read_excel(
        io.BytesIO(adjustments_workbook([(numeric, "total_premium"), (region, "total_premium")])), sheet_name=None
    )
    assert list(sheets) == ["driver_age", "region"]
    assert list(sheets["driver_age"].columns) == ["Feature", "Band", "Relativity"]
    assert list(sheets["driver_age"]["Band"]) == ["[20,25)", "[30,35)"]
    assert sheets["driver_age"]["Relativity"].iloc[0] == pytest.approx(1.1)
    assert pd.isna(sheets["driver_age"]["Relativity"].iloc[1])
    assert list(sheets["region"]["Band"]) == ["east"]


def test_overlays_on_one_base_model_apply_together():
    from daisy.overlay import Overlay, apply_overlay, apply_overlays

    data = pd.DataFrame({"driver_age": [21.0, 33.0], "region": ["east", "north"]})
    driver_age = Overlay("driver_age", "GBM 1", "driver_age", 5.0, {20.0: 1.1, 30.0: 0.9})
    region = Overlay("region", "GBM 1", "region", 0.0, {0.0: 1.2}, levels=["east", "north"])
    expected = np.array([10.0, 10.0])
    together = apply_overlays([driver_age, region], data, expected)
    assert list(together) == pytest.approx([10.0 * 1.1 * 1.2, 10.0 * 0.9])
    assert list(together) == pytest.approx(list(apply_overlay(region, data, apply_overlay(driver_age, data, expected))))


# ---------------------------------------------------------------- scoring out-of-time data
class _Booster:  # picklable stand-in for a LightGBM booster
    def predict(
        self,
        X,
        num_iteration=None,
        raw_score=False,
    ):
        return 0.01 * X["driver_age"].to_numpy(dtype=float)


def _gbm(**overrides) -> GBMModel:
    base = dict(
        booster=_Booster(),
        response="total_premium",
        weight="N",
        init_score="no offset",
        params={"objective": "tweedie", "tweedie_variance_power": 1.4},
        features=["driver_age"],
        categories={},
        link="log",
        offset=None,
        predictions=np.zeros(1),
        pred_rows=np.arange(1),
        importances=pd.DataFrame(),
        evaluation_log={},
        best_iteration=0,
    )
    base.update(overrides)
    return GBMModel(**base)


def test_normalise_objective_maps_glm_and_lightgbm_names():
    assert [
        normalise_objective(name)
        for name in (
            "poisson",
            "quasipoisson",
            "Gamma",
            "tweedie",
            "gaussian (log link)",
            "regression",
            "huber",
            "binary",
            "cross_entropy",
        )
    ] == ["poisson", "poisson", "gamma", "tweedie", "gaussian", "gaussian", "gaussian", "binomial", "binomial"]
    assert normalise_objective(None) == "poisson" and normalise_objective("nonsense") == "poisson"
    assert normalise_objective("nonsense", default=None) is None


def test_unit_deviance_is_zero_at_the_observation_and_none_for_unsuitable_data():
    y = np.array([0.5, 1.0, 2.0])
    for objective in ("poisson", "gamma", "tweedie", "gaussian"):
        assert np.allclose(unit_deviance(objective, y, y, 1.5), 0.0)
        assert np.all(unit_deviance(objective, y, y * 1.3, 1.5) > 0)
    assert np.allclose(
        unit_deviance("binomial", np.array([0.0, 1.0]), np.array([0.2, 0.9])), [-2 * np.log(0.8), -2 * np.log(0.9)]
    )
    assert unit_deviance("gamma", np.array([0.0, 1.0]), np.array([1.0, 1.0])) is None  # gamma needs y > 0
    assert unit_deviance("poisson", np.array([-1.0]), np.array([1.0])) is None
    assert unit_deviance("tweedie", y, y, power=2.5) is None


def test_pode_depends_on_the_objective_and_defaults_to_poisson():
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"y": rng.gamma(2.0, 50.0, 400)})
    pred = data["y"].to_numpy() * rng.uniform(0.8, 1.25, 400)
    default = prediction_performance_metrics(data, "y", pred)
    poisson = prediction_performance_metrics(data, "y", pred, objective="poisson")
    gamma = prediction_performance_metrics(data, "y", pred, objective="gamma")
    gaussian = prediction_performance_metrics(data, "y", pred, objective="regression")
    assert default["PoDE"].iloc[0] == poisson["PoDE"].iloc[0]
    assert len({round(float(frame["PoDE"].iloc[0]), 6) for frame in (poisson, gamma, gaussian)}) == 3
    # Gaussian PoDE is R-squared (1 - SSE / SST) for equally weighted rows
    expected = 1 - np.sum((data["y"] - pred) ** 2) / np.sum((data["y"] - data["y"].mean()) ** 2)
    assert np.isclose(gaussian["PoDE"].iloc[0], expected)
    labelled = prediction_performance_metrics(
        data.assign(train_test=np.arange(400) % 2), "y", pred, dataset_label="Unseen Data"
    )
    assert list(labelled["Dataset"]) == ["Unseen Data"]


def test_score_metrics_uses_the_standard_metrics_format():
    data = pd.DataFrame({"total_premium": [100.0, 200.0, 300.0, 400.0], "exposure": [1.0, 1.0, 2.0, 2.0]})
    metrics = score_metrics(data, "total_premium", np.array([110.0, 190.0, 310.0, 380.0]), "exposure", "gamma")
    assert list(metrics.columns) == [
        "Dataset",
        "GINI",
        "PoDE",
        "RMSE",
        "Within 5% (%)",
        "Within 10% (%)",
        "Within 20% (%)",
    ]
    assert list(metrics["Dataset"]) == [DATASET_LABEL]
    with pytest.raises(ValueError, match="no 'total_claims' column"):
        score_metrics(data, "total_claims", np.ones(4))
    with pytest.raises(ValueError, match="weight"):
        score_metrics(data, "total_premium", np.ones(4), "missing")


def test_csv_is_read_in_chunks_with_progress(tmp_path):
    path = tmp_path / "oot.csv"
    pd.DataFrame({"driver_age": np.arange(1000), "vehicle_brand": ["a", "b"] * 500}).to_csv(path, index=False)
    updates = []
    data = read_csv_with_progress(path, lambda fraction, rows: updates.append((fraction, rows)), chunksize=300)
    assert data.shape == (1000, 2) and list(data["driver_age"][:3]) == [0, 1, 2]
    fractions = [fraction for fraction, _ in updates]
    assert len(updates) >= 5 and fractions == sorted(fractions) and fractions[-1] == 1.0
    assert updates[-1][1] == 1000 and 0 < fractions[0] < 1
    assert [rows for _, _, rows in iter_csv_chunks(path, 400)] == [400, 800, 1000]
    with pytest.raises(FileNotFoundError):
        read_csv_with_progress(tmp_path / "missing.csv")
    (tmp_path / "empty.csv").write_text("driver_age,vehicle_brand\n")
    with pytest.raises(ValueError, match="no rows"):
        read_csv_with_progress(tmp_path / "empty.csv")


def test_ratebook_scoring_matches_the_model_it_was_written_from(tmp_path):
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {"driver_age": rng.integers(18, 80, 300).astype(float), "vehicle_brand": rng.choice(list("ABC"), 300)}
    )

    def model(frame):
        factor = np.where(frame["vehicle_brand"] == "B", 1.2, np.where(frame["vehicle_brand"] == "C", 0.9, 1.0))
        return 100 * np.exp(0.01 * frame["driver_age"].to_numpy()) * factor

    spec = pd.DataFrame(
        {
            "feature": ["driver_age", "vehicle_brand"],
            "banding": ["1", ""],
            "base_level": ["", "A"],
            "min": ["", ""],
            "max": ["", ""],
        }
    )
    path = write_ratebook_excel(
        tabulate_model(train, model, ["driver_age", "vehicle_brand"], spec),
        str(tmp_path / "rb.xlsx"),
        "GLM 1: gamma ~ Avg Premium",
    )
    ratebook = read_ratebook(path)
    assert ratebook.model_name == "GLM 1: gamma ~ Avg Premium" and set(ratebook.features) == {
        "driver_age",
        "vehicle_brand",
    }
    new = pd.DataFrame({"driver_age": [18.0, 30.0, 55.0, 79.0], "vehicle_brand": ["A", "B", "C", "A"]})
    predictions, notes = score_ratebook(ratebook, new)
    assert notes == [] and np.allclose(predictions, model(new))  # driver_age band of 1: exact

    edge = pd.DataFrame({"driver_age": [10.0, 90.0, np.nan], "vehicle_brand": ["A", "Z", "B"]})
    predictions, notes = score_ratebook(ratebook, edge)
    assert len(notes) == 3 and np.all(np.isfinite(predictions))
    assert any("outside the ratebook range" in note for note in notes)
    assert any("not in the ratebook (Z)" in note for note in notes) and any("missing values" in n for n in notes)
    with pytest.raises(ValueError, match="missing ratebook features"):
        score_ratebook(ratebook, new[["driver_age"]])


def test_read_ratebook_rejects_other_workbooks(tmp_path):
    from openpyxl import Workbook

    other = Workbook()
    other.active.title = "Data"
    other.save(tmp_path / "other.xlsx")
    with pytest.raises(ValueError, match="Summary"):
        read_ratebook(tmp_path / "other.xlsx")


def test_gbm_pickle_is_read_and_scored_with_the_training_offset(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(_gbm(weight="exposure")))
    model = read_gbm_pickle(path)
    data = pd.DataFrame({"driver_age": [10.0, 20.0], "exposure": [1.0, 2.0]})
    assert np.allclose(
        score_gbm(model, data), np.exp(0.01 * data["driver_age"]) * data["exposure"]
    )  # offset = log(weight)
    assert np.allclose(score_gbm(_gbm(), data), np.exp(0.01 * data["driver_age"]))
    assert model_objective(model) == ("tweedie", 1.4)
    with pytest.raises(ValueError, match="exposure"):
        score_gbm(model, data[["driver_age"]])
    with pytest.raises(ValueError, match="missing model features"):
        score_gbm(model, data[["exposure"]])
    other = tmp_path / "other.pkl"
    other.write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(ValueError, match="Daisy GBM"):
        read_gbm_pickle(other)


def test_gbm_pickle_can_be_read_from_the_model_bundle_zip(tmp_path):
    import zipfile

    bundle = tmp_path / "daisy_gbm_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("model.pkl", pickle.dumps(_gbm()))
        archive.writestr("metadata.json", "{}")
    assert read_gbm_pickle(bundle).features == ["driver_age"]
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("readme.txt", "x")
    with pytest.raises(ValueError, match="no model.pkl"):
        read_gbm_pickle(empty)


def test_glm_objective_and_scored_frame():
    glm = SimpleNamespace(objective="tweedie", var_power=1.6)
    assert model_objective(glm) == ("tweedie", 1.6)
    data = pd.DataFrame({"a": [1, 2]})
    out = scored_frame(data, np.array([0.5, 1.5]))
    assert list(out.columns) == ["a", "prediction"] and list(out["prediction"]) == [0.5, 1.5]
    assert list(data.columns) == ["a"]  # the input is untouched


def test_map_summary_gives_predicted_rate_and_ae_by_area():
    from daisy.maps import postcode_area_summary

    d = pd.DataFrame(
        {"circulation_area": ["AB", "AB", "CD"], "total_claims": [2.0, 0.0, 1.0], "exposure": [10.0, 10.0, 5.0]}
    )
    summary = postcode_area_summary(
        d,
        "total_claims",
        "exposure",
        np.ones(3, dtype=int),
        area_col="circulation_area",
        expected=np.array([1.0, 1.0, 1.0]),
    )
    row = summary.set_index("area")
    assert row.loc["AB", "predicted"] == pytest.approx(2.0 / 20.0)  # expected 2 over exposure 20
    assert row.loc["AB", "ae"] == pytest.approx(2.0 / 2.0)  # actual 2 over expected 2
    assert row.loc["CD", "ae"] == pytest.approx(1.0 / 1.0)
    assert (
        "predicted"
        not in postcode_area_summary(
            d, "total_claims", "exposure", np.ones(3, dtype=int), area_col="circulation_area"
        ).columns
    )


def test_chart_feature_choices_group_modelled_and_non_modelled_columns():
    from daisy.app import _chart_feature_choices

    columns = ["age", "brand", "premium"]
    # before a model is built the drop-down is the flat list of columns
    assert _chart_feature_choices(columns, None) == {c: c for c in columns}
    choices = _chart_feature_choices(columns, ["age"], excluded={"premium"})
    assert list(choices) == ["Modelled", "Non-modelled"]
    assert choices["Modelled"] == {"age": "age"}
    assert choices["Non-modelled"] == {"brand": "brand"}
