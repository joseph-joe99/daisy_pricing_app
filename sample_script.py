# read libraries
import pandas as pd
import numpy as np
from daisy.__main__ import serve
from daisy.app import make_app

# read in the data
df = pd.read_csv("...Path_to_data.csv")

# create a train/test split column if it doesn't already exist
if "train_test" not in df.columns:
    df["train_test"] = np.random.default_rng(1).binomial(1, 0.2, len(df))

# specification files: give the paths (not DataFrames), so "Update specification" in Charts can save
# banding, minimum, maximum and bins back to the feature specification csv
kpi_spec = "...Path_to_kpi_spec.csv"
feature_spec = "...Path_to_feature_spec.csv"
filter_spec = "...Path_to_filter_spec.csv"

# create the app
app = make_app(
    df,
    dataset_name="df",
    starting_response="",
    kpi_spec=kpi_spec,
    feature_spec=feature_spec,
    filter_spec=filter_spec,
    title="Motor Insurance Pricing",
    starting_tab="Data",
)

if __name__ == "__main__":
    # hosts on the IP address of whoever runs this script (found automatically, printed to share) and on localhost;
    # pass host="127.0.0.1" to keep the app to this computer only.
    # For HTTPS on the same port, set CERT_FILE and KEY_FILE to a certificate and its private key.
    CERT_FILE = None  # e.g. "certs/daisy.crt"
    KEY_FILE = None  # e.g. "certs/daisy.key"
    serve(app, port=8000, launch_browser=True, ssl_certfile=CERT_FILE, ssl_keyfile=KEY_FILE)
