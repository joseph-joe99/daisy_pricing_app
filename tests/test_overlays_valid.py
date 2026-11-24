import pandas as pd

from daisy.specs import ensure_overlays_valid, load_specification, overlays_valid_features, specification_template


def test_template_has_overlays_valid_next_to_modelling_valid():
    template = specification_template(pd.DataFrame({"age": [1], "train_test": ["train"]}), "feature")
    columns = list(template.columns)
    assert columns.index("overlays_valid") == columns.index("modelling_valid") + 1
    assert template.loc[template["feature"] == "age", "overlays_valid"].tolist() == ["feature"]
    assert template.loc[template["feature"] == "train_test", "overlays_valid"].tolist() == [""]


def test_missing_column_is_copied_from_modelling_valid():
    spec = pd.DataFrame({"feature": ["a", "b"], "modelling_valid": ["feature", ""], "min": "", "max": "",
                         "base_level": "", "interaction_grouping": ""})
    out = ensure_overlays_valid(spec)
    assert list(out.columns).index("overlays_valid") == list(out.columns).index("modelling_valid") + 1
    assert out["overlays_valid"].tolist() == ["feature", ""]


def test_overlays_valid_features_only_lists_marked_rows():
    spec = pd.DataFrame({"feature": ["a", "b", "c"], "overlays_valid": ["feature", "", " Feature "]})
    assert overlays_valid_features(spec) == {"a", "c"}


def test_overlays_valid_features_empty_without_column():
    assert overlays_valid_features(pd.DataFrame({"feature": ["a"]})) == set()


def test_loaded_feature_spec_gets_overlays_valid(tmp_path):
    path = tmp_path / "spec.csv"
    pd.DataFrame({"feature": ["a"], "base_level": [""], "min": [""], "max": [""], "interaction_grouping": [""],
                  "modelling_valid": ["feature"]}).to_csv(path, index=False)
    loaded = load_specification(None, str(path), "feature")
    assert loaded["overlays_valid"].tolist() == ["feature"]


def test_bundled_spec_marks_same_features_as_modelling_valid():
    from daisy.data import DATA_DIR

    spec = pd.read_csv(DATA_DIR / "insurance_feature_spec.csv", dtype=str, keep_default_na=False)
    assert (spec["overlays_valid"] == spec["modelling_valid"]).all()
