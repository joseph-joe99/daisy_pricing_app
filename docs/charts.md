# Charts

The Charts tab explores a response, KPI or model by band of any feature.

## 1-way line and bar

- **Controls** (left to right): Feature, Banding, Min, Max, Bins, Monotonicity, Group low exposure, Add columns,
  Weight bars and Sigma bars. **Prev** and **Next** under Feature step through the features in the drop-down order.
  Banding, Min, Max and Bins apply when you press Enter or leave the box.
- **Banding** sets the width of each band. Other features start from their specification banding, or a guessed one.
- **Min** and **Max** cap the chart: rows below Min join the first band and rows above Max join the last. Blank means
  no limit. They start from the feature specification.
- **Bins** splits the feature into that many equal bins from Min to Max, for example 4 bins from 0 to 20 are
  [0,5), [5,10), [10,15), [15,Inf). Bins and Banding are alternatives: clear Banding (or set it to 0) to use bins.
- **Monotonicity** (None, Increasing, Decreasing) is the constraint a GBM built from now on gets for the feature.
  The saved constraint is shown above the Banding box.
- **Update specification** saves the feature's banding, min, max, bins and monotonicity to the feature
  specification (see [specifications](specifications.md)).
- The **bars** show each band's share of the total weight, labelled "Prop. of" the weight name.

## Other sections

With a model selected, the Charts tab also shows:

- **Model Overview:** total weight and response, averages and prediction quantiles for train and test, and for a
  GBM its parameters and monotonicity.
- **Metrics:** the train and test metrics for the selected model.
- **1-way line and bar:** Modelled features (most important first) and Non-modelled features, in separate tabs.
- **SHAP** (GBMs) and **ALE** (GLMs and GBMs): effects of each feature on the model's predictions, with a Two-way
  view of the top 5 features in the report.
- **Lift Chart:** binned actual vs predicted for train and test, the Lorenz curve with its Gini, and the A/E table.
- **Feature importance:** the last tab, GBM gain or GLM contribution for every feature.
- **Keep only this model:** removes every other GLM and GBM from the session to free memory. No files on disk are
  deleted.

**Export charts (.html)** saves all of this as one self-contained page (see [reports](reports.md)).

## Rows, filters and maps

Under Rows, All, Train or Test selects the rows, and the filter specification narrows them further. The Maps view
draws a postcode-area choropleth and needs `geopandas`; the repo ships the area geometry.
