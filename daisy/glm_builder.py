"""Visual builder for GLM formulas.

Features are dragged (or clicked) into an ordered list of model terms. Each term says how it enters the model:
linear, categorical, B-spline, natural or cyclic spline, polynomial, log, square root, square, clipped, or centred.
Terms can be combined into interactions (a:b, a*b, a/b) and the intercept can be removed. The builder writes the
Patsy formula into the Formula tab's text box as it changes, so the Formula tab always shows the same model.

Everything here runs in the browser: the list of terms lives in the page, not on the server.
"""

import json
from pathlib import Path

from shiny import ui

_CSS = """
.glm-chips { display: flex; flex-wrap: wrap; gap: .4rem; }
.glm-chip { cursor: grab; border: 1px solid #adb5bd; border-radius: 999px; background: #f8f9fa;
  padding: .2rem .7rem; font-size: .9rem; }
.glm-chip[data-kind="categorical"] { border-style: dashed; }
.glm-chip:hover { background: #e9ecef; }
.glm-terms { min-height: 160px; border: 2px dashed #ced4da; border-radius: .5rem; padding: .5rem;
  display: flex; flex-direction: column; gap: .4rem; }
.glm-term { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; border: 1px solid #dee2e6;
  border-radius: .4rem; padding: .35rem .5rem; background: #fff; }
.glm-term-name { font-weight: 600; min-width: 7rem; }
.glm-term select, .glm-term input { font-size: .85rem; }
.glm-term input[type="number"] { width: 5.5rem; }
.glm-term code { font-size: .8rem; }
.glm-term-buttons { margin-left: auto; display: flex; gap: .25rem; }
.glm-hint { color: #6c757d; font-size: .9rem; }
.glm-interaction { background: #f1f8ff; }
.glm-preview { white-space: pre-wrap; word-break: break-all; font-size: .85rem; }
"""

# Browser code (static/glm_builder.js): the builder state, drag and drop, and writing the formula into the Formula
# tab's text box.
GLM_BUILDER_JS = (Path(__file__).parent / "static" / "glm_builder.js").read_text(encoding="utf-8")


def builder_ui():
    """The Builder tab: the feature chips above the model terms, with interactions and the intercept below."""
    return ui.div(
        ui.tags.style(_CSS),
        ui.p(
            "Drag features into Model terms, or click them. Choose how each term enters the model. "
            "The formula is written to the Formula tab as you build.",
            class_="text-muted small",
        ),
        ui.card(
            ui.card_header("Features"),
            ui.output_ui("glm_builder_chips"),
            ui.p("Solid: numeric. Dashed: categorical.", class_="text-muted small mt-2 mb-0"),
        ),
        ui.card(
            ui.card_header("Model terms"),
            ui.div(
                ui.div("Drop features here, or click a feature above.", id="glm-hint", class_="glm-hint"),
                id="glm-terms",
                class_="glm-terms",
            ),
            ui.hr(),
            ui.tags.strong("Interaction"),
            ui.div(
                ui.tags.select(id="glm-pick-a", class_="form-select form-select-sm", style="width:auto"),
                ui.tags.select(id="glm-op", class_="form-select form-select-sm", style="width:auto"),
                ui.tags.select(id="glm-pick-b", class_="form-select form-select-sm", style="width:auto"),
                ui.tags.button(
                    "Add interaction",
                    {"type": "button", "id": "glm-add-interaction", "class": "btn btn-sm btn-outline-primary"},
                ),
                class_="d-flex flex-wrap align-items-center gap-2 mb-3",
            ),
            ui.div(
                ui.tags.label(
                    ui.tags.input({"type": "checkbox", "id": "glm-intercept", "checked": "checked"}),
                    " Include intercept",
                    class_="me-3",
                ),
                ui.tags.button(
                    "Clear all terms",
                    {"type": "button", "id": "glm-clear", "class": "btn btn-sm btn-outline-secondary"},
                ),
                class_="d-flex flex-wrap align-items-center gap-2",
            ),
            ui.hr(),
            ui.tags.strong("Formula"),
            ui.tags.pre(ui.tags.code("Add features to start the formula.", id="glm-preview"), class_="glm-preview"),
            ui.hr(),
            ui.tags.strong("Metrics"),
            ui.p(
                "For the GLM chosen in the GLMs tab's Model drop-down, on the current filter. Build the formula to see "
                "its metrics here.",
                class_="text-muted small mb-1",
            ),
            ui.output_data_frame("glm_builder_metrics"),
        ),
        ui.tags.script(GLM_BUILDER_JS),
        class_="p-2",
    )


def feature_chips(features: list[tuple]):
    """The draggable feature buttons: ``(column, label, "numeric" or "categorical")``, plus, for a categorical feature,
    its levels as a fourth item (a list of values, used by the reference and level pickers)."""
    if not features:
        return ui.p("No features to add.", class_="text-muted small")
    return ui.div(
        *[
            ui.tags.button(
                label,
                {
                    "type": "button",
                    "class": "glm-chip",
                    "draggable": "true",
                    "data-feature": name,
                    "data-kind": kind,
                    "title": name,
                    "data-levels": json.dumps(list(levels[0]) if levels else [], default=str),
                },
            )
            for name, label, kind, *levels in features
        ],
        class_="glm-chips",
    )
