# Scoring

The **Scoring** tab scores a cleaned CSV of unseen data and measures the predictions. Hide the tab with
`make_app(..., show_Scoring=False)`.

## Sidebar

1. **Data:** type the path to the CSV and press *Load data*. A progress bar shows the rows read, and the
   Column summary appears once the data is in.
2. **Model:** *Model selected in the app* scores with a GLM or GBM built in the session. Otherwise read one in:
   - **GBM:** a `model.pkl`, or the Model bundle (.zip) that contains it. The model's own offset and weight convention
     is applied to the new data. `model.pkl` is the complete model.
   - **GLM:** a ratebook (`.xlsx`). Predictions are the base prediction times the relativity of each feature's level
     (see [models](models.md)).
3. **KPI & Model Objective:** choose the response and weight columns. Pick *None* to skip metrics. For a ratebook,
   pick its objective and Tweedie variance power, which PoDE uses.

## Tabs

- **Column summary**, **Predictions** (with notes) and **Metrics**: the standard metrics table (GINI, PoDE, RMSE,
  Within 5, 10 and 20%) for the unseen data.
- **1-way line and bar:** the predictions against the response for any column, with the same bars, lines, Min, Max,
  Bins, low-exposure grouping and Prev / Next as the Charts tab. The response must be chosen first. Features are
  split into Modelled and Non-modelled groups. Banding, Min, Max and Bins start from the feature specification.

## Outputs

- **Score data** produces the predictions.
- **Predictions (.csv)** downloads only the `prediction` column, one row per scored row.
- **Export scoring report (.html)** saves the scored data details (file, rows, model, objective, response and weight,
  mean prediction, missing predictions and notes), the metrics table and the 1-way charts. It is titled "Scored Data
  Report" and named after the model. It needs a response column, so choose one and score the data first.

## Rules

- For a model that uses the weight, a row with a weight of 0 is scored as 0, and a row with a missing or negative
  weight is scored as blank (NaN).
- Only open model files you trust, because loading a pickle can run code.
- The scoring logic lives in `daisy/scoring.py`.
