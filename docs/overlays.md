# Overlays

The **Overlays** tab adjusts a base model with band relativities. Each feature's bands get a relativity, and the
overlay multiplies the base model's predictions by the relativity of each row's band. The suggestions come from the
training rows, and the adjustment is checked on test rows.

## Steps

1. **Data (optional):** type the path to a cleaned CSV and press *Load data* to calculate on a smaller in-sample set
   instead of the training rows. If that file has a `train_test` column, its rows with 1 are the test set (used first), and the
   rest are the rows the suggestions use.
2. **Read Model (optional):** read in a GBM `model.pkl` or Model bundle, or a GLM ratebook (`.xlsx`). These appear in
   the Base model list. For a ratebook, pick its objective and Tweedie variance power under Parameters, because a
   ratebook does not record them.
3. **Model & Bootstrap Samples:** choose the base model and the number of bootstrap samples (default 500).
4. **Calculate suggestions for all features** works through the base model's modelled features and the other data
   columns (Non-modelled). For each feature it groups the rows into bands and bootstraps each band's actual / expected
   (A/E) to get 90%, 95% and 99% ranges. Numeric features start with their specification banding, or a guessed one.
   Categorical features (text, category or boolean) use one band per level. Each band is then:
   - **Adjust:** its 99% range lies entirely above or below 1. The relativity is set to its A/E.
   - **Review:** only its 95% or 90% range lies entirely above or below 1. The relativity stays at 1 for you to decide.
   - **Keep at 1:** every range contains 1.
5. **Feature dropdown** (Modelled and Non-modelled groups, with Prev and Next) chooses which feature the chart, grid,
   balance check and save use. The chart shows each band's A/E with its ranges and the share of weight. Edit the
   **Relativity** column to change any band; the A/E-with-relativities plot below the grid updates.
6. **Check balance** shows the training A/E after the overlay and warns about relativities outside their 99% range
   or bands the rule says to adjust but left at 1.
7. **Save overlay** keeps it in the session. Names default to the feature (for example "NCD overlay"), with a number
   added if the name is taken. Saving is per feature, so save each one you want.

## Testing

To test on data the model has never seen, type the path to an out-of-sample CSV under **Path to cleaned CSV
(Out-of-sample test data)** and press *Load test data*. Every row of that file is then the test set, and the AvE plots
and metrics below use it instead of the test rows above. It must have the base model's features and the response.

**Overlays Plots on Test Data** (the dropdown) picks a saved overlay and shows its test-row plot: actual, base and
adjusted per row, with error bars (tick box above the plot) and the share of rows per band. Prev and Next step
through the overlays.

The **metrics table** compares the base model with all of its saved overlays applied together, on the test rows:
GINI, PoDE, RMSE and Within 5, 10 and 20%, with the change and whether it improved.

## Exports

- **Export report (.html):** one self-contained page for all saved overlays, with tabs for the metrics (per base
  model), the overlay details, the test-row plots and relativities (AvE Plots and Relativities), and the before and
  after A/E plots with their ranges (Bootstrapping). The report is titled "<KPI> - Overlays Report", using the KPI name of the overlays' response.
- **Export all adjustments (.xlsx):** one workbook with a tab per feature, listing each band's Feature, Band and
  Relativity.

## Rules and limits

- Overlays on the same base model are applied together. Two overlays on the same feature would both apply.
- Overlays are removed when their base model is removed by *Keep only this model*.
- A level that is unseen in training, or blank, uses a relativity of 1.
- Unsaved edits to the grid are lost when you switch features.
- Non-modelled columns with many distinct values make the calculation slower.
