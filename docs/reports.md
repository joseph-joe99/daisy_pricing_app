# Reports

Every export is one self-contained HTML page: it works offline, has no network requests, and includes its own
logo and charts. Each one is built by `daisy/report.py`.

- **Charts (Export charts (.html)):** the current Charts settings as tabs: Model Overview, Metrics, 1-way line and
  bar (Modelled and Non-modelled), SHAP, ALE, Lift Chart and Feature importance. With a `train_test` column, a Rows
  switch (All, Train, Test) redraws the plots and metrics for that split. The file is named "daisy_charts - <KPI>.html".
  The page is titled "<KPI> - Model Report" when a model is included, and "<KPI> - Report" otherwise.
  The Two-way views cover every pair of the top 5 features by importance, and the **Two-way top N** box sets how many (2 to 10).
  An **Error bars** tick box shows or hides the sigma error bars without rebuilding the page.
- **Scoring (Export scoring report (.html)):** titled "<KPI> - Scoring Report". See [scoring](scoring.md).
- **Overlays (Export report (.html)):** titled "<KPI> - Overlays Report", with these tabs:
  - **Metrics movement:** the base model against all of its overlays applied together, on the test rows.
  - **Overlay Details:** one table with a row per overlay (base model, feature, feature type, banding, response,
    training and test rows, A/E totals and bands changed), and the saved overlay names.
  - **AvE Plots and Relativities:** a dropdown picks each overlay to see its test-row plot and relativities.
  - **Bootstrapping:** a dropdown picks each overlay to see its A/E by band before and after the relativities, with
    the 99%, 95% and 90% ranges.
