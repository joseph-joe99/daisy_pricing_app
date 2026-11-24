# GLMs and GBMs

## Building models

- **GLMs** are built from a formula (the default is in `daisy/data/glm_formula.csv`, and the GLMs tab takes a
  formula with one term per line). Terms use Patsy and Python syntax; the GLM engine is our own IRLS fit.
- The GLMs tab's **Builder** tab builds the formula without typing it: drag features (or click them) into the model
  terms, then choose each term's form: linear, categorical `C()`, B-spline `bs()`, natural cubic `cr()`, cyclic
  `cc()`, polynomial, log, square root, square, clip to a minimum and maximum (`np.clip`), or centred `center()`.
  A categorical term can also pick **selected levels**. Each selected level gets its own coefficient, the other levels
  are grouped as `(other)`, and the first selected level is the baseline.
- The **Formula** tab exports the formula as text (`glm_formula.txt`), one term per line. To import, choose a text
  file with one term per line (`#` comments are allowed). The formula goes into the Formula box. The Builder does not
  read imported text, so its terms stay as they were until you change them there.
  Add interactions as `a:b`, `a*b` or `a/b`, and turn the intercept off. The formula is written to the **Formula**
  tab as you build. Edits made directly in the Formula tab are not read back into the Builder.
- **GBMs** are LightGBM models. Each build setting in the GBMs sidebar has a hover message explaining it, and the
  GBM parameters table shows the same messages. Monotonicity and interaction groups come from the feature
  specification.
- Each model shows metrics, a lift chart with its Lorenz curve, predictions, and importance.
The gamma objective needs a response above 0 on every row. If a gamma GBM or GLM finds rows with a zero, negative or
missing response or weight, the app asks whether to build it without those rows. Agreeing leaves those rows out of that
model's training, test and metrics only; the data itself is not changed, and the model is named "(response > 0)". Rows with a zero weight are left out of that model too, and the dialog gives one count of the rows left out.

## Response and weight

When a weight is chosen, "Response and weight" says what the response is:

- **Total response, per unit of weight (frequency and severity models, default):** the response is a total (for
  example ClaimCount or total claim amount) and the model is fitted on response / weight, with the weight as a prior
  weight. Predictions are returned as totals, so for a frequency model they are expected claim counts. Works with any
  link.
- **Average response (already per unit):** the response is already per unit of weight, and the weight is a prior
  weight. Predictions are per unit.

The weight must be above 0 on every row: a build stops if the weight column has a zero or negative value, so remove
those rows from the data first. Models saved with the removed log-offset basis (`total`) are refused with a message
to rebuild them.

PoDE, RMSE, Within % and GINI follow the model's setting. Metrics, the lift chart and the model overview use the
weight the model was built with, so they do not change when the sidebar weight changes. The 1-way, SHAP and map
charts use the sidebar weight.

PoDE is the proportion of deviance explained on the model's own objective (gamma deviance for a gamma model,
Tweedie deviance at the model's variance power, and so on). A GLM's AIC follows the standard glm definition. For a
Poisson with a weight, the AIC is on the totals (response x weight, e.g. claim counts), and it is shown as NaN for a
weighted binomial and for quasi and Tweedie models.

## GBMs with an offset

LightGBM does not start from the average when it is given an offset, so Daisy fits the overall level first: with a
log link, log(total response / total offset) on the training rows. This level is stored as `offset_base`, shown as
"starting level (link scale)" in the GBM parameters, included in the SHAP base value, and written to
`metadata.json` in the model bundle.

An **Offset** column is an expected total (for example an expected claim count that includes the exposure). Daisy
divides it by the weight to get the per-unit starting value, so it is used together with the weight rather than
replacing it.

## Ratebooks

**Ratebook (.xlsx)** writes a base prediction and one sheet of relativities per feature. A ratebook multiplies
relativities together, so it is only offered for log-link models. Numeric features are tabulated at the min, max
and banding from the feature specification, up to 2,000 levels per feature. Beyond that the bands widen by a whole
multiple of the banding, so the levels still fall on the specified banding.

Scoring a ratebook (see [scoring](scoring.md)) multiplies the base prediction by the relativity of each feature's
level. Numeric features use the band that starts at the nearest ratebook level at or below the value, and unseen
levels use a relativity of 1.

## Model bundles and model files

**Model bundle (.zip)** contains `model.pkl`, the complete GBM, and `metadata.json`. The `model.txt` in the bundle is
the raw LightGBM booster: for a model trained with an offset, add `offset_base` from `metadata.json` to its raw
score. Only open model files you trust, because loading a pickle can run code.

## Keep only this model

In Charts, **Keep only this model** removes every other GLM and GBM from the session to save memory, after a
confirmation. Overlays built on the removed models are removed too. No files on disk are deleted.
