from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_demo_dataset() -> pd.DataFrame:
    """The UK motor demo dataset shipped with Daisy (about 283,000 policy rows)."""
    d = pd.read_csv(DATA_DIR / "insurance.csv")
    for c in d.columns[d.dtypes == object]:
        d[c] = d[c].astype("category")
    for c in d.columns:
        if str(d[c].dtype) in ("str", "string"):
            d[c] = d[c].astype("category")
    return d


def demo_spec_paths() -> dict:
    """Paths to the bundled KPI, feature and filter demo specification CSV files."""
    return {k: str(DATA_DIR / f"insurance_{k}_spec.csv") for k in ("kpi", "feature", "filter")}


def numerical_cols(d: pd.DataFrame):
    """Names of the numeric columns in a DataFrame, excluding boolean columns."""
    return [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]
