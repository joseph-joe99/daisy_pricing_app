"""The GLM builder writes Patsy formulas: each option it can emit must build a GLM."""

import numpy as np
import pandas as pd
import pytest

from daisy.glm import build_glm


def _data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "claims": rng.poisson(0.3, n).astype(float),
            "N": np.ones(n),
            "driver age": rng.uniform(18, 80, n),
            "vehicle_km": rng.uniform(1000, 30000, n),
            "make": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "region": pd.Categorical(rng.choice(["north", "south"], n)),
        }
    )


# The Patsy text the builder emits for each term type and interaction operator.
FORMULAS = [
    "make",
    "C(make)",
    "bs(vehicle_km, df=4)",
    "cr(vehicle_km, df=4)",
    "cc(vehicle_km, df=4)",
    "vehicle_km + I(vehicle_km ** 2)",
    "np.log(vehicle_km)",
    "np.sqrt(vehicle_km)",
    "I(vehicle_km ** 2)",
    "center(vehicle_km)",
    "np.clip(vehicle_km, 2000, 25000)",
    "np.maximum(vehicle_km, 2000)",
    "np.minimum(vehicle_km, 25000)",
    'Q("driver age")',
    'bs(Q("driver age"), df=3)',
    "(C(make)) : (C(region))",
    "(C(make)) * (C(region))",
    "(C(make)) / (C(region))",
    "(C(make)) : (bs(vehicle_km, df=4))",
]


@pytest.mark.parametrize("formula", FORMULAS)
def test_builder_formula_terms_fit(formula):
    d = _data()
    model = build_glm(d, "claims", "N", f"C(region) + {formula}", objective="poisson")
    assert np.isfinite(model.fit.deviance)


@pytest.mark.parametrize("formula", ["0 + make", "0 + C(make) + vehicle_km"])
def test_no_intercept_prefix_fits(formula):
    d = _data()
    model = build_glm(d, "claims", "N", formula, objective="poisson")
    assert "(Intercept)" not in list(model.fit.columns)


def test_term_importance_lists_an_interaction_as_its_own_term():
    d = _data()
    model = build_glm(d, "claims", "N", "C(region) + C(make) + C(make):C(region)", objective="poisson")
    imp = model.term_importance(d.iloc[model.pred_rows])
    assert any(":" in term for term in imp.index)
    assert len(imp) == len(model.design.terms)
    assert list(imp.values) == sorted(imp.values, reverse=True)
    # the interaction is not counted again inside the main-effect rows: each term's own mean |contribution| x 100
    tc = model.term_contributions(d.iloc[model.pred_rows])
    assert np.isclose(imp.iloc[0], (tc.abs().mean() * 100).max())


# The Patsy the builder writes for a categorical term with selected levels.
CATEGORICAL_FORMULAS = [
    "C(np.where(np.isin(make, ['a', 'c']), make, '(other)'), Treatment('a'))",  # selected levels
]


@pytest.mark.parametrize("formula", CATEGORICAL_FORMULAS)
def test_builder_categorical_reference_and_levels_fit_and_predict(formula):
    d = _data()
    model = build_glm(d, "claims", "N", formula, objective="poisson")
    assert np.isfinite(model.fit.deviance)
    new = d.iloc[:50]
    assert np.isfinite(model.predict(new)).all()


def test_selected_levels_give_each_selected_level_its_own_coefficient():
    d = _data()
    formula = "C(np.where(np.isin(make, ['a', 'c']), make, '(other)'), Treatment('a'))"
    model = build_glm(d, "claims", "N", formula, objective="poisson")
    # a is the baseline; c has its own coefficient; b (not selected) is grouped with the other levels
    terms = [name.rsplit("[T.", 1)[1].rstrip("]") for name in model.fit.columns if "[T." in name]
    assert terms == ["(other)", "c"]


BUILT_FORMULA = (
    "# a comment\n"
    "I(np.clip(vehicle_km, 2000, 25000)) +\n"
    "C(np.where(np.isin(make, ['a', 'c']), make, '(other)'), Treatment('a')) +\n"
    "bs(Q('driver age'), df=3)"
)


def test_formula_terms_split_at_top_level_plus_only():
    from daisy.glm import formula_terms

    terms = formula_terms(BUILT_FORMULA)
    assert terms == [
        "I(np.clip(vehicle_km, 2000, 25000))",
        "C(np.where(np.isin(make, ['a', 'c']), make, '(other)'), Treatment('a'))",
        "bs(Q('driver age'), df=3)",
    ]


def test_formula_text_round_trip_gives_the_same_formula():
    from daisy.glm import formula_from_import, formula_terms, formula_to_text

    exported = formula_to_text(BUILT_FORMULA)
    assert exported.splitlines()[0].endswith(" +")
    assert len(exported.splitlines()) == 3
    assert formula_terms(formula_from_import(exported)) == formula_terms(BUILT_FORMULA)


def test_text_formula_import_keeps_terms_and_drops_comments():
    from daisy.glm import formula_from_import, formula_terms

    assert formula_terms(formula_from_import(BUILT_FORMULA)) == formula_terms(BUILT_FORMULA)


def test_one_term_per_line_without_plus_signs_is_read():
    from daisy.glm import formula_terms

    assert formula_terms("I(np.clip(driver_age, 18, 75))\nC(policy_type)\nbs(year, df=5)\n") == [
        "I(np.clip(driver_age, 18, 75))",
        "C(policy_type)",
        "bs(year, df=5)",
    ]


def test_empty_import_is_rejected():
    from daisy.glm import formula_from_import

    with pytest.raises(ValueError):
        formula_from_import("# only a comment\n\n")
