"""UK postcode-area choropleth (the repo ships only the area-level geometry)."""
from __future__ import annotations
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from .data import DATA_DIR


@lru_cache(maxsize=1)
def load_areas():
    """Postcode-area polygons as (GeoDataFrame, geojson dict). Needs geopandas."""
    import geopandas as gpd

    g = gpd.read_file(DATA_DIR / "areas_MappaR.gpkg", layer="areas_MappaR").to_crs(4326)
    g = g[["PostcodeArea", "population", "geometry"]]
    return g, json.loads(g.to_json())


def postcode_area_summary(
    d: pd.DataFrame,
    response: str,
    weight: str,
    total_filter: np.ndarray,
    area_col: str = "PostcodeArea",
    expected=None,
) -> pd.DataFrame:
    """Rows, weight, actual and rate by postcode area. With ``expected`` (the model's expected totals for each row of
    ``d``) it also gives the predicted rate and the A/E (actual / expected) by area."""
    mask = np.asarray(total_filter) == 1
    w = np.ones(len(d)) if weight == "N" else d[weight].to_numpy(float)
    mask &= ~np.isnan(w) & (w > 0)
    df = pd.DataFrame(
        {"area": d.loc[mask, area_col].astype(str).to_numpy(), "w": w[mask], "r": d.loc[mask, response].to_numpy(float)}
    )
    if expected is not None:
        df["e"] = np.asarray(expected, dtype=float)[mask]
    aggregates = dict(count=("w", "size"), weight=("w", "sum"), r_sum=("r", "sum"))
    if expected is not None:
        aggregates["e_sum"] = ("e", "sum")
    out = df.groupby("area").agg(**aggregates).reset_index()
    out["rate"] = out["r_sum"] / out["weight"]
    if expected is not None:
        out["predicted"] = out["e_sum"] / out["weight"]
        out["ae"] = out["r_sum"] / out["e_sum"].where(out["e_sum"] != 0)
    return out


AE_METRIC = "ae"  # A/E is shown on a diverging scale centred on 1


def area_choropleth(
    summary: pd.DataFrame,
    metric: str = "rate",
    title: str | None = None,
):
    """Choropleth map of postcode areas coloured by a chosen measure, with hover details for counts, weight, rate and predicted or actual-to-expected values."""
    import plotly.express as px

    _, gj = load_areas()
    hover = {"count": True, "weight": ":,.0f", "rate": ":.3f"}
    for column, fmt in (("predicted", ":.3f"), ("ae", ":.4f")):
        if column in summary.columns:
            hover[column] = fmt
    kw = dict(
        geojson=gj,
        locations="area",
        featureidkey="properties.PostcodeArea",
        color=metric,
        color_continuous_scale="Viridis",
        hover_data=hover,
        center={"lat": 54.5, "lon": -3.0},
        zoom=4.3,
        opacity=0.8,
    )
    if metric == AE_METRIC:
        kw.update(color_continuous_scale="RdBu_r", color_continuous_midpoint=1.0)
    fn = getattr(px, "choropleth_map", None)
    if fn is not None:
        fig = fn(summary, map_style="carto-positron", **kw)
    else:  # older plotly
        fig = px.choropleth_mapbox(summary, mapbox_style="carto-positron", **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=30 if title else 0, b=0), title=title, height=650)
    return fig
