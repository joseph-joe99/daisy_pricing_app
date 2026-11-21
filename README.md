# Daisy

Daisy is a Python / Shiny app for exploring insurance data and building, checking and scoring GLM and LightGBM
pricing models without writing code.

## What it does

- **Charts:** a response, KPI or model by band of any feature, with banding, caps, bins and filters. See [charts](docs/charts.md).
- **GLMs and GBMs:** build models, check metrics, lift, SHAP and importance, and export ratebooks. See [models](docs/models.md).
- **Overlays:** adjust a model with band relativities, bootstrapped on training data and tested on held-out data. See [overlays](docs/overlays.md).
- **Scoring:** score unseen data with a model from the session, a GBM `model.pkl`, or a GLM ratebook. See [scoring](docs/scoring.md).
- **Specifications:** labels, banding, caps, bins and monotonicity for each feature. See [specifications](docs/specifications.md).
- **Reports:** every export is one self-contained HTML page. See [reports](docs/reports.md).

## Install and run

```bash
python3.12 -m venv py312
source py312/bin/activate                   # Windows: .\py312\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
shiny run app.py --reload                   # demo: UK motor dataset (about 283,000 rows)
```

Other ways to start the app:

```bash
python -m daisy                             # this computer's address and localhost, port 8000
python -m daisy --host 127.0.0.1            # this computer only
shiny run app.py --host 0.0.0.0 --port 8000 # every address of this computer
```

To serve over HTTPS on one port, give a certificate and its key. The app then answers only on `https://` at that
port:

```bash
python -m daisy --port 8443 --cert certs/daisy.crt --key certs/daisy.key
```

`shiny run` does not take certificate options in this Shiny version, so use `python -m daisy` or `sample_script.py`
for HTTPS.

For a test certificate on this computer, create one with OpenSSL (the browser will warn that it is self-signed):

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/daisy.key -out certs/daisy.crt -days 365 -subj "/CN=localhost"
```

The app has no login, so anyone who can reach its address can see the data, and the Scoring tab reads files from
this computer. Only share it on a network you trust.

## Using your own data

```python
from daisy.app import make_app
app = make_app(df,
               dataset_name="mydata",
               starting_response="claims_cost",
               kpi_spec="mydata_kpi_spec.csv",
               feature_spec="mydata_feature_spec.csv",
               filter_spec="mydata_filter_spec.csv")
```

`sample_script.py` is a complete example: it reads a CSV and the three specification files, adds a `train_test`
column if there is none, and serves the app on port 8000. Fill in the paths and run `python sample_script.py`.

Your data needs:

- a response column, and optionally a weight (exposure) column;
- a `train_test` column (0 = train, 1 = test) for test views and GBMs;
- a `PostcodeArea` column for the map.

## Layout

```
daisy/
  app.py                 Shiny for Python UI and server
  __main__.py            python -m daisy, serve()
  specs.py filters.py    specification and filter handling
  banding.py summary.py  bands, caps, bins and one-way summaries
  glm.py gbm.py ale.py   models, SHAP and ALE
  overlay.py             overlay relativities and bootstrapping
  ratebook.py            ratebook tables and Excel
  scoring.py             out-of-time scoring
  charts.py maps.py      plotly figures
  report.py              HTML exports
  data.py                demo dataset and specifications (daisy/data/)
app.py                   demo app for `shiny run`
sample_script.py         your own data, start to finish
docs/                    guides to each part of the app
tests/                   test_core.py
```

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Licence

MIT. See `LICENSE`.
