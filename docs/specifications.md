# Specifications

Daisy reads three specification files: KPI, filter and feature. They are csv files in the same format as the app
(examples are in `daisy/data/`). If a feature specification is not given, the app builds one template with one row
per data column, a blank label (the feature name is shown instead) and `modelling_valid = feature` for every column
except `train_test`.

## Feature specification

| Column | Meaning |
|---|---|
| `feature` | The data column. |
| `label` | The name shown on axes. Blank means the feature name. |
| `base_level` | The base level for a GLM's categorical features. |
| `min`, `max` | The range for the Charts and Scoring plots. Charts applies them as caps; see below. |
| `banding` | The band width for numeric features. |
| `monotonicity` | `increasing` or `decreasing` for a GBM. Set it in Charts with Monotonicity, then Update specification. |
| `interaction_grouping` | Features in the same group may interact in a GBM. |
| `modelling_valid` | `feature` to use the column in models; blank or other values leave it out. |
| `banding_type` | Optional. `capped` limits the values to `min` and `max` for modelling; `bins` splits the range into `bins` equal bins for modelling. Blank uses the feature as it is. |
| `bins` | Optional. The number of bins when `banding_type` is `bins`. |

### What is applied where

- **Charts and Scoring plots:** Banding, Min, Max and Bins start from the specification when a feature is selected. If
  `min` or `max` is blank, the limit is not applied.
- **Modelling:** only the `banding_type` setting changes the values a model is built on. A capped feature is modelled
  with its values limited, and a binned one as a factor with one level per bin. The data itself is not changed, so
  filters and charts keep the original values. A model remembers the treatment it was built with and applies it to
  anything it scores. A binned feature has no monotonicity constraint in a GBM.
- The `bins` and `banding_type` columns are optional. Specifications without them work as before.

**Update specification** (in Charts) saves a feature's banding, min, max, bins and monotonicity. It writes the csv
when the feature specification was given to `make_app` as a file path; otherwise the change lasts for the session,
and you can download the specifications from the Specifications tab.

## Banding and bins in Charts

Banding is the width of each band. Min and Max cap the chart: rows below Min join the first band and rows above Max
join the last. Bins split the feature into equal bins from Min to Max (the data's range when those are blank), and
the last bin has no upper end. Bins and Banding are alternatives.

## KPI and filter specifications

The KPI specification names the KPIs shown in the app: each has a numerator (the response) and a denominator (the
weight). If the app is started with a `starting_response`, its first KPI is selected, and the KPI name is used as the
primary y-axis title while the response and weight still match it. The filter specification lists filters for the
Rows selection; lines starting with `---` are section titles in the app.
