"""KPI / filter / feature specification files."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

FEATURE_SPEC_NOT_MODELLING_VALID = {"train_test"}


def ensure_overlays_valid(feature_spec: pd.DataFrame) -> pd.DataFrame:
    """Add the ``overlays_valid`` column next to ``modelling_valid`` when a feature spec does not have it yet.

    The new column starts as a copy of ``modelling_valid``, so a spec saved before the column existed keeps the same
    features for the Overlays suggestions as it had for the models.
    """
    if "overlays_valid" in feature_spec.columns:
        return feature_spec
    out = feature_spec.copy()
    values = out["modelling_valid"] if "modelling_valid" in out.columns else ""
    position = list(out.columns).index("modelling_valid") + 1 if "modelling_valid" in out.columns else len(out.columns)
    out.insert(position, "overlays_valid", values)
    return out


def overlays_valid_features(feature_spec: pd.DataFrame) -> set[str]:
    """The features the feature specification marks ``overlays_valid = feature``: the only features Overlays suggestions use."""
    if "overlays_valid" not in feature_spec.columns:
        return set()
    marked = feature_spec["overlays_valid"].fillna("").astype(str).str.strip().str.lower().eq("feature")
    return set(feature_spec.loc[marked, "feature"].astype(str).str.strip())

REQUIRED = {
    "feature": ["feature", "base_level", "min", "max", "interaction_grouping"],
    "kpi": [
        "kpi_name",
        "kpi_numerator",
        "kpi_denominator",
        "kpi_dp",
        "kpi_signif",
        "kpi_divisor",
        "kpi_prefix",
        "kpi_suffix",
    ],
    "filter": ["filter"],
}


def specification_template(
    d: pd.DataFrame | None,
    spec_type: str,
) -> pd.DataFrame:
    """Empty specification table for the given type ('kpi', 'filter' or 'feature'), with its columns set up."""
    if spec_type == "feature":
        # same columns as the bundled insurance_feature_spec.csv; every column except the train / test flag
        # is marked modelling-valid, and a blank label falls back to the feature name
        cols = list(d.columns) if d is not None else []
        return pd.DataFrame(
            {
                "feature": cols,
                "label": "",
                "base_level": "",
                "min": "",
                "max": "",
                "banding": "",
                "monotonicity": "",
                "interaction_grouping": "",
                "modelling_valid": ["" if c in FEATURE_SPEC_NOT_MODELLING_VALID else "feature" for c in cols],
                "overlays_valid": ["" if c in FEATURE_SPEC_NOT_MODELLING_VALID else "feature" for c in cols],
                "bins": "",
                "banding_type": "",
            }
        )
    if spec_type == "kpi":
        return pd.DataFrame(
            [
                dict(
                    kpi_name="User defined",
                    kpi_numerator="Numerator",
                    kpi_denominator="Denominator",
                    kpi_dp=0,
                    kpi_signif=0,
                    kpi_divisor=1,
                    kpi_prefix="",
                    kpi_suffix="",
                )
            ]
        )
    if spec_type == "filter":
        return pd.DataFrame({"filter": ["no filter"]})
    raise ValueError(f"unknown specification type {spec_type!r}")


def load_specification(
    d,
    specification,
    spec_type: str,
) -> pd.DataFrame:
    """Load a spec from a csv path / DataFrame, falling back to a template.

    Filter specs are read line by line (they may contain commas), and keep
    ``--- header`` lines, which the UI uses as section titles.
    """
    x = None
    if specification is None:
        return specification_template(d, spec_type)
    if isinstance(specification, pd.DataFrame):
        x = specification.copy()
    else:
        p = Path(str(specification))
        if not p.is_file():
            return specification_template(d, spec_type)
        if spec_type == "filter":
            lines = [ln.rstrip("\r\n") for ln in p.read_text(encoding="utf-8").splitlines()]
            if lines and lines[0].strip().lower() == "filter":
                lines = lines[1:]
            x = pd.DataFrame({"filter": lines})
        else:
            x = pd.read_csv(p, dtype=str, keep_default_na=False)
    x = x.fillna("")
    if not all(c in x.columns for c in REQUIRED[spec_type]):
        return specification_template(d, spec_type)
    if spec_type == "feature":
        x = ensure_overlays_valid(x)
    return x


def filter_choices(filter_spec: pd.DataFrame) -> dict:
    """Group filter expressions under their '--- title' headers (for a grouped select)."""
    groups: dict[str, list[str]] = {"No filter": ["no filter"]}
    current = "Filters"
    for f in filter_spec["filter"]:
        f = str(f).strip()
        if not f or f == "no filter":
            continue
        if f.startswith("---"):
            current = f.lstrip("- ").strip() or "Filters"
            continue
        groups.setdefault(current, []).append(f)
    return groups


def filter_select_choices(filter_spec: pd.DataFrame) -> dict:
    """Choices for the Filter drop-down: ``{"no filter": "no filter", group: {expression: expression}}``.

    Shiny shows each group as an optgroup only when its choices are a dict, and the value of each option is the
    filter expression that gets applied (a list would show the group title as the only option).
    """
    choices: dict = {"no filter": "no filter"}
    for group, items in filter_choices(filter_spec).items():
        if group != "No filter" and items:
            choices[group] = {item: item for item in items}
    return choices


def feature_lookup(
    feature_spec: pd.DataFrame,
    feature: str,
) -> dict:
    """Specification row for one feature as a dict (empty when the feature is not in the spec)."""
    row = feature_spec[feature_spec["feature"] == feature]
    return {} if row.empty else row.iloc[0].to_dict()


def feature_label(
    feature_spec: pd.DataFrame | None,
    feature: str,
) -> str:
    """Display label of a feature (``label`` column of the feature spec), else the feature name."""
    if feature_spec is None or "label" not in feature_spec.columns:
        return str(feature)
    label = feature_lookup(feature_spec, feature).get("label")
    if label is None or (isinstance(label, float) and pd.isna(label)) or not str(label).strip():
        return str(feature)
    return str(label).strip()


def numeric_or_none(x):
    """Convert a value to float, returning None when it is blank or not a number."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


BANDING_TYPES = ("capped", "bins")


def feature_treatment_spec(
    feature_spec: pd.DataFrame | None,
    feature: str,
) -> dict | None:
    """What the specification says to do to a feature's values for modelling, or None.

    ``banding_type`` is ``capped`` (limit the values to ``min`` / ``max``) or ``bins`` (also split the range into
    the number of equal bins in the ``bins`` column); when it is blank the feature is modelled as it is.
    """
    if feature_spec is None or "banding_type" not in feature_spec.columns:
        return None
    row = feature_lookup(feature_spec, feature)
    kind = str(row.get("banding_type", "")).strip().lower()
    if kind not in BANDING_TYPES:
        return None

    def number(value):
        """Numeric value for a treatment cell (None when blank or NaN). Bins are read as a whole count."""
        value = numeric_or_none(value)
        return None if value is None or value != value else value

    bins = number(row.get("bins")) if kind == "bins" else None
    bins = int(bins) if bins and bins > 0 and float(bins).is_integer() else None
    if kind == "bins" and not bins:
        return None
    return {"min": number(row.get("min")), "max": number(row.get("max")), "bins": bins}


def set_feature_treatment(
    feature_spec: pd.DataFrame,
    feature: str,
    minimum=None,
    maximum=None,
    banding=None,
    bins=None,
) -> pd.DataFrame:
    """A copy of the feature specification with ``min``, ``max``, ``banding``, ``bins`` and ``banding_type`` set
    for one feature. ``banding_type`` becomes ``bins`` when a number of bins is given, else ``capped`` when a
    minimum or maximum is given, else blank. The columns and the feature's row are added when missing."""

    def text(value) -> str:
        """Text form of a treatment value for storing in the spec: blank when missing, otherwise a short number."""
        value = numeric_or_none(value)
        return "" if value is None or value != value else f"{value:g}"

    spec = feature_spec.copy()
    columns = ["min", "max", "banding", "bins", "banding_type"]
    for column in columns:
        if column not in spec.columns:
            spec[column] = ""
        spec[column] = spec[column].astype(object).where(spec[column].notna(), "")
    if not (spec["feature"] == feature).any():
        row = {column: "" for column in spec.columns}
        row["feature"] = feature
        spec = pd.concat([spec, pd.DataFrame([row])], ignore_index=True)
    bins = text(bins) if numeric_or_none(bins) else ""
    kind = "bins" if bins else ("capped" if text(minimum) or text(maximum) else "")
    chosen = spec["feature"] == feature
    spec.loc[chosen, columns] = [text(minimum), text(maximum), "" if bins else text(banding), bins, kind]
    return spec


MONOTONICITY_VALUES = ("increasing", "decreasing")


def feature_monotonicity(
    feature_spec: pd.DataFrame | None,
    feature: str,
) -> str:
    """``increasing``, ``decreasing`` or ``""`` (no constraint) for a feature in the feature specification."""
    if feature_spec is None or "monotonicity" not in feature_spec.columns:
        return ""
    value = str(feature_lookup(feature_spec, feature).get("monotonicity", "") or "").strip().lower()
    return value if value in MONOTONICITY_VALUES else ""


def set_feature_monotonicity(
    feature_spec: pd.DataFrame,
    feature: str,
    value: str,
) -> pd.DataFrame:
    """A copy of the feature specification with the ``monotonicity`` of one feature set to ``increasing``,
    ``decreasing`` or blank (none). The column and the feature's row are added when missing."""
    spec = feature_spec.copy()
    if "monotonicity" not in spec.columns:
        spec["monotonicity"] = ""
    spec["monotonicity"] = spec["monotonicity"].astype(object).where(spec["monotonicity"].notna(), "")
    if not (spec["feature"] == feature).any():
        row = {column: "" for column in spec.columns}
        row["feature"] = feature
        spec = pd.concat([spec, pd.DataFrame([row])], ignore_index=True)
    text = str(value or "").strip().lower()
    spec.loc[spec["feature"] == feature, "monotonicity"] = text if text in MONOTONICITY_VALUES else ""
    return spec


def spec_chart_limits(
    feature_spec: pd.DataFrame,
    feature: str,
) -> tuple:
    """(Min, Max) a feature starts with on the charts: the modelling treatment's values when the specification sets one,
    otherwise the specification's min and max columns."""
    applied = feature_treatment_spec(feature_spec, feature) or {}
    row = feature_lookup(feature_spec, feature)
    minimum = applied.get("min") if applied.get("min") is not None else numeric_or_none(row.get("min"))
    maximum = applied.get("max") if applied.get("max") is not None else numeric_or_none(row.get("max"))
    return minimum, maximum
