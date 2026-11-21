"""Standalone builder of the self-contained HTML charts report (``daisy_charts.html``).

Everything that produces HTML lives here: the page template (CSS and JavaScript), the logo, the table
renderers and the section builders for the Model Overview, Metrics, 1-way line and bar, SHAP, ALE, Lift Chart and
Feature importance tabs.
Figures themselves come from :mod:`daisy.charts`. The page makes no network requests: Plotly is embedded and
the Daisy flower is read from the package ``images`` folder at export time and inlined into the page.
"""
from __future__ import annotations
import base64
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
from .banding import banding_guesser
from .charts import (
    ale_figure,
    importance_figure,
    importance_height,
    shap_heatmap_figure,
    lift_chart_figure,
    lift_table_frame,
    line_bar_figure,
    lorenz_figure,
    shap_flame_figure,
    stacked_shap_figure,
)
from .specs import (
    feature_label,
    feature_lookup,
    feature_monotonicity,
    feature_treatment_spec,
    numeric_or_none,
    spec_chart_limits,
)
from .summary import one_way_summary

IMAGES_DIR = Path(__file__).parent / "images"
LOGO_FILE = "daisy_flower.svg"

_SECTION_TITLES = {
    "overview": "Model Overview",
    "lines": "1-way line and bar",
    "shap": "SHAP",
    "ale": "ALE",
    "metrics": "Metrics",
    "lift": "Lift Chart",
    "importance": "Feature importance",
}
_GROUP_TITLES = {"all": "All features", "modelled": "Modelled", "non-modelled": "Non-modelled"}
_EXPORT_COLOURS = ["#24543f", "#d9623b", "#3b82a0", "#e9a83b", "#7b5ea7", "#8c6d46"]
_EXPORT_CSS = """
:root{--green-900:#173628;--green-700:#24543f;--green-500:#3f7f5f;--green-200:#cfe3d6;--green-100:#e3f0e6;
--cream:#f4f8f2;--gold:#e9a83b;--petal:#f7e9b0;--ink:#1f2d26;--muted:#5b6b61;--card:#ffffff;
--shadow:0 1px 2px rgba(23,54,40,.08),0 8px 24px rgba(23,54,40,.08)}
*{box-sizing:border-box}
body{margin:0;background:var(--cream);color:var(--ink);
font-family:"Inter","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;line-height:1.45}
header{background:linear-gradient(135deg,var(--green-900),var(--green-700) 60%,var(--green-500));color:#f4faf5;
padding:28px 24px 64px;border-bottom:4px solid var(--gold)}
.wrap{max-width:1200px;margin:0 auto}
.brand{display:flex;align-items:center;gap:16px}
.brand svg{width:52px;height:52px;flex:none;filter:drop-shadow(0 2px 4px rgba(0,0,0,.25))}
.brand h1{margin:0;font-size:1.65rem;letter-spacing:.02em}
.brand p{margin:2px 0 0;color:#cfe3d6;font-size:.95rem}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.chip{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.22);border-radius:999px;
padding:4px 12px;font-size:.82rem;color:#f4faf5}
main{margin-top:-44px;padding:0 24px 40px}
.sections{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
.section-list{display:flex;gap:8px;flex-wrap:wrap}
.views{margin-left:auto;display:flex;align-items:center;gap:8px;color:#f4faf5;font-size:.78rem;font-weight:700;
text-transform:uppercase;letter-spacing:.06em}
.seg{display:inline-flex;background:rgba(255,255,255,.92);border-radius:999px;padding:3px;box-shadow:var(--shadow)}
.seg button{appearance:none;border:0;background:transparent;cursor:pointer;font:inherit;font-size:.85rem;font-weight:700;
padding:6px 16px;border-radius:999px;color:var(--green-700);text-transform:none;letter-spacing:0}
.seg button:hover{background:var(--green-100)}
.seg button[aria-pressed="true"]{background:var(--green-700);color:#fff}
.section-btn{appearance:none;border:1.5px solid rgba(255,255,255,.35);cursor:pointer;font:inherit;font-weight:700;
padding:9px 20px;border-radius:999px;color:#f4faf5;background:rgba(23,54,40,.55);backdrop-filter:blur(2px);
transition:background .15s,color .15s}
.section-btn:hover{background:rgba(23,54,40,.8)}
.section-btn[aria-selected="true"]{background:var(--gold);border-color:var(--gold);color:var(--green-900)}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tab{appearance:none;border:0;cursor:pointer;font:inherit;font-weight:600;padding:10px 18px;color:var(--green-700);
background:var(--green-100);border-radius:12px 12px 0 0;transition:background .15s,color .15s}
.tab:hover{background:var(--green-200)}
.tab[aria-selected="true"]{background:var(--card);color:var(--green-900);box-shadow:0 -3px 0 var(--gold) inset}
.tab .count{margin-left:8px;font-size:.78rem;background:var(--green-700);color:#fff;border-radius:999px;padding:1px 8px}
.tab[aria-selected="true"] .count{background:var(--gold);color:var(--green-900)}
.panel{background:var(--card);border-radius:14px;box-shadow:var(--shadow);padding:22px}
.tabs:not([hidden]) + .panel{border-top-left-radius:0}
.note{margin:0 0 16px;padding:10px 14px;border-left:4px solid var(--gold);background:#fdf8e8;border-radius:0 8px 8px 0;
color:#5b4a14;font-size:.9rem}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;padding-bottom:18px;border-bottom:1px solid var(--green-100)}
.field{display:flex;flex-direction:column;gap:4px;min-width:260px;flex:1 1 260px;max-width:460px}
.field label{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
select{font:inherit;padding:10px 38px 10px 12px;border:1.5px solid var(--green-200);border-radius:10px;color:var(--ink);
background-color:#fff;background-image:linear-gradient(45deg,transparent 50%,#24543f 50%),linear-gradient(135deg,#24543f 50%,transparent 50%);
background-position:calc(100% - 20px) calc(50% + 1px),calc(100% - 14px) calc(50% + 1px);background-size:6px 6px;background-repeat:no-repeat;
appearance:none;cursor:pointer}
select:focus,button:focus-visible{outline:3px solid rgba(233,168,59,.55);outline-offset:1px}
.nav{display:flex;gap:8px}
.btn{font:inherit;font-weight:600;cursor:pointer;padding:10px 14px;border-radius:10px;border:1.5px solid var(--green-200);
background:#fff;color:var(--green-700)}
.btn:hover:not(:disabled){background:var(--green-100)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.meta{margin-left:auto;display:flex;gap:8px;align-items:center;color:var(--muted);font-size:.85rem}
.badge{background:var(--petal);color:#6a4a07;border-radius:8px;padding:3px 10px;font-weight:600}
.plot-title{margin:22px 0 0;font-size:1rem;color:var(--green-700)}
.plot{width:100%;margin-top:8px}
.empty,.error{padding:48px 12px;text-align:center;color:var(--muted)}
.table-wrap{overflow-x:auto}
table.metrics{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
table.metrics th{background:var(--green-700);color:#fff;text-align:right;padding:10px 14px;white-space:nowrap}
table.metrics td{text-align:right;padding:10px 14px;border-bottom:1px solid var(--green-100);white-space:nowrap}
table.metrics th:first-child,table.metrics td:first-child{text-align:left;font-weight:700;color:var(--green-900)}
table.metrics.text th,table.metrics.text td{text-align:left;white-space:normal}
.table-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:8px 28px;align-items:start}
table.metrics tbody tr:nth-child(even){background:#f8fbf7}
table.metrics tbody tr:hover{background:var(--green-100)}
footer{max-width:1200px;margin:0 auto;padding:0 24px 28px;color:var(--muted);font-size:.8rem}
@media (max-width:640px){header{padding:20px 16px 56px}main{padding:0 12px 24px}.meta{margin-left:0}}
@media print{.controls,.tabs,.sections{display:none}header{background:none;color:#000}}
"""

_EXPORT_JS = """
(function(){
  var DATA = JSON.parse(document.getElementById('daisy-data').textContent);
  function $(id){ return document.getElementById(id); }
  var sectionsEl=$('sections'), tabsEl=$('tabs'), controlsEl=$('controls'), selectEl=$('feature'),
      prevEl=$('prev'), nextEl=$('next'), plotsEl=$('plots'), emptyEl=$('empty'), noteEl=$('note'),
      staticEl=$('static'), bandEl=$('badge'), posEl=$('position');
  var cache = {};
  var viewsEl = $('views'), viewsWrap = $('views-wrap');
  var state = {section: 0, view: DATA.defaultView || DATA.views[0], group: DATA.sections.map(function(){ return 0; }), index: {}};
  var CONFIG = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']};

  function section(){ return DATA.sections[state.section]; }
  function groups(){ return section().groups || []; }
  function group(){ return groups()[state.group[state.section]] || null; }
  function key(){ return state.section + ':' + state.group[state.section]; }
  function viewOf(f){
    if (state.view === DATA.views[0]) { return f; }
    return (f.views && f.views[state.view]) || {error: 'Nothing to show for ' + state.view + ' rows.', figs: []};
  }
  function figure(f, v, i){
    var k = f.id + ':' + state.view + ':' + i;
    if (!cache[k]) { cache[k] = JSON.parse(v.figs[i].fig); }
    return cache[k];
  }
  function drawViews(){
    viewsEl.innerHTML = '';
    show(viewsWrap, DATA.views.length > 1);
    DATA.views.forEach(function(name){
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = name;
      b.setAttribute('aria-pressed', name === state.view ? 'true' : 'false');
      b.addEventListener('click', function(){ state.view = name; drawViews(); drawSection(); });
      viewsEl.appendChild(b);
    });
  }
  function show(el, on){ el.style.display = on ? '' : 'none'; }
  function purge(){
    Array.prototype.slice.call(plotsEl.children).forEach(function(c){ try { Plotly.purge(c); } catch (e) {} });
    plotsEl.innerHTML = '';
  }

  function drawSections(){
    sectionsEl.innerHTML = '';
    DATA.sections.forEach(function(s, i){
      var b = document.createElement('button');
      b.className = 'section-btn'; b.type = 'button'; b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', i === state.section ? 'true' : 'false');
      b.textContent = s.title;
      b.addEventListener('click', function(){ state.section = i; drawSection(); });
      sectionsEl.appendChild(b);
    });
  }
  function drawSection(){
    drawSections();
    var s = section();
    noteEl.textContent = s.note || ''; show(noteEl, !!s.note);
    if (s.kind === 'html') {
      purge(); show(tabsEl, false); show(controlsEl, false); show(plotsEl, false); show(emptyEl, false);
      staticEl.innerHTML = (s.views && s.views[state.view] !== undefined && state.view !== DATA.views[0]) ? s.views[state.view] : s.html;
      show(staticEl, true);
      return;
    }
    staticEl.innerHTML = ''; show(staticEl, false); show(plotsEl, true);
    drawTabs(); drawGroup();
  }
  function drawTabs(){
    tabsEl.innerHTML = '';
    var gs = groups();
    tabsEl.hidden = gs.length < 2; show(tabsEl, gs.length > 1);
    gs.forEach(function(g, i){
      var b = document.createElement('button');
      b.className = 'tab'; b.type = 'button'; b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', i === state.group[state.section] ? 'true' : 'false');
      b.appendChild(document.createTextNode(g.title));
      var c = document.createElement('span'); c.className = 'count'; c.textContent = g.features.length;
      b.appendChild(c);
      b.addEventListener('click', function(){ state.group[state.section] = i; drawTabs(); drawGroup(); });
      tabsEl.appendChild(b);
    });
  }
  function drawGroup(){
    var g = group(), has = !!g && g.features.length > 0;
    selectEl.innerHTML = '';
    if (has) { g.features.forEach(function(f, i){
      var o = document.createElement('option'); o.value = i; o.textContent = f.name; selectEl.appendChild(o);
    }); }
    selectEl.disabled = !has;
    show(controlsEl, has && !section().single); show(emptyEl, !has);
    emptyEl.textContent = (g && g.empty) || 'No features in this group.';
    if (has) { selectEl.value = state.index[key()] || 0; }
    drawPlot();
  }
  function drawPlot(){
    purge();
    var g = group(), n = g ? g.features.length : 0;
    prevEl.disabled = nextEl.disabled = n < 2;
    if (!n) { return; }
    var i = state.index[key()] || 0, f = g.features[i], v = viewOf(f);
    bandEl.textContent = v.badge || ''; show(bandEl, !!v.badge);
    posEl.textContent = (i + 1) + ' of ' + n;
    if (v.error) {
      var err = document.createElement('div'); err.className = 'error'; err.textContent = v.error;
      plotsEl.appendChild(err); return;
    }
    v.figs.forEach(function(item, j){
      if (item.title) {
        var h = document.createElement('h3'); h.className = 'plot-title'; h.textContent = item.title;
        plotsEl.appendChild(h);
      }
      var div = document.createElement('div'); div.className = 'plot';
      var fig = figure(f, v, j), showErrors = !!$('errbars') && $('errbars').checked;
      (fig.data || []).forEach(function(t){ if (t.error_y) { t.error_y.visible = showErrors; } });
      div.style.height = ((fig.layout && fig.layout.height) || 480) + 'px';
      plotsEl.appendChild(div);
      Plotly.react(div, fig.data, fig.layout, CONFIG);
    });
    if (v.html) {
      var table = document.createElement('div'); table.style.marginTop = '18px'; table.innerHTML = v.html;
      plotsEl.appendChild(table);
    }
  }
  function move(step){
    var n = group().features.length, k = key();
    state.index[k] = ((state.index[k] || 0) + step + n) % n;
    selectEl.value = state.index[k]; drawPlot();
  }
  selectEl.addEventListener('change', function(){ state.index[key()] = +selectEl.value; drawPlot(); });
  if ($('errbars')) { $('errbars').checked = !!DATA.errorBars; $('errbars').addEventListener('change', drawPlot); }
  prevEl.addEventListener('click', function(){ move(-1); });
  nextEl.addEventListener('click', function(){ move(1); });
  document.addEventListener('keydown', function(e){
    var g = group();
    if (e.target === selectEl || e.altKey || e.ctrlKey || e.metaKey || section().kind === 'html' || !g || !g.features.length) { return; }
    if (e.key === 'ArrowLeft') { move(-1); } else if (e.key === 'ArrowRight') { move(1); }
  });
  drawViews();
  drawSection();
})();
"""


# shown only if the logo file cannot be read (no web address involved)
_FALLBACK_LOGO = (
    '<svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="14" fill="#f7e9b0"/>'
    '<circle cx="20" cy="20" r="5.1" fill="#e9a83b"/></svg>'
)
_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def logo_html(path: str | Path | None = None) -> str:
    """The report logo, read from the ``images`` folder (or ``path``) and embedded in the page.

    SVG files are inlined as markup; raster images are embedded as data URIs. Nothing is fetched from the
    internet, and the exported report stays a single self-contained file.
    """
    file = Path(path) if path else IMAGES_DIR / LOGO_FILE
    try:
        if file.suffix.lower() == ".svg":
            svg = file.read_text(encoding="utf-8")
            svg = re.sub(r"<\?xml[^>]*\?>|<!DOCTYPE[^>]*>", "", svg).strip()
            return re.sub(r"""\sxmlns=(["'])[^"']*\1""", "", svg, count=1)  # namespace is implied inline in HTML
        mime = _IMAGE_TYPES[file.suffix.lower()]
        encoded = base64.b64encode(file.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{encoded}" alt="">'
    except (OSError, KeyError, UnicodeDecodeError):
        return _FALLBACK_LOGO


def _script_safe_json(value) -> str:
    """JSON that is safe to place inside a <script> element."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _style_export_figure(
    figure,
    x_label: str,
    height: int = 540,
) -> None:
    """Apply the Daisy look to a figure. Only the 'weight' bars and ordinary lines are recoloured."""

    def recolour_bar(trace):
        """Recolour exposure (secondary-axis) bars to the export accent colour."""
        if trace.type == "bar" and trace.yaxis == "y2":  # the exposure / weight bars
            trace.update(marker_color="rgba(233, 168, 59, 0.45)")

    figure.for_each_trace(recolour_bar)
    lines = [t for t in figure.data if t.type == "scatter" and not t.fill and (t.line is None or t.line.width != 0)]
    for i, trace in enumerate(lines):
        colour = _EXPORT_COLOURS[i % len(_EXPORT_COLOURS)]
        trace.update(line=dict(color=colour, width=2.5), marker=dict(color=colour, size=7))
    figure.for_each_trace(
        lambda t: t.update(fillcolor="rgba(63, 127, 95, 0.22)") if t.type == "scatter" and t.fill else None
    )
    figure.update_layout(
        title=None,
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color="#1f2d26"),
        margin=dict(l=60, r=60, t=70, b=60),
        hoverlabel=dict(bgcolor="#ffffff"),
    )
    figure.update_xaxes(title_text=x_label, gridcolor="#e3f0e6", linecolor="#cfe3d6")
    figure.update_yaxes(gridcolor="#e3f0e6", zerolinecolor="#cfe3d6")


def _feature_entry(
    index: int,
    name: str,
    figs: list[tuple[str | None, object]],
    badge: str | None = None,
    prefix: str = "plot",
    error: str | None = None,
) -> dict:
    """Report entry for one feature: its figures converted to JSON, with a badge and an optional error."""
    import plotly.io as pio

    return {
        "id": f"{prefix}-{index:03d}",
        "name": str(name),
        "badge": badge,
        "error": error,
        "figs": [{"title": title, "fig": pio.to_json(figure)} for title, figure in figs],
    }


def _error_entry(
    index: int,
    name: str,
    message: str,
    prefix: str,
) -> dict:
    """Report entry for a feature whose plots could not be built, carrying the error message."""
    return {"id": f"{prefix}-{index:03d}", "name": str(name), "badge": None, "error": message, "figs": []}


def _feature_limits(
    data: pd.DataFrame,
    feature: str,
    feature_spec: pd.DataFrame,
    selected_feature: str | None = None,
    selected_limits=None,
) -> tuple:
    """(minimum, maximum, bins) for a feature's 1-way plot: the chart setting for the selected feature,
    else what the specification applies to modelling (``banding_type``), else nothing."""
    if feature == selected_feature and selected_limits is not None:
        return tuple(selected_limits)
    if not pd.api.types.is_numeric_dtype(data[feature]):
        return None, None, None
    minimum, maximum = spec_chart_limits(feature_spec, feature)
    applied = feature_treatment_spec(feature_spec, feature) or {}
    return minimum, maximum, applied.get("bins")


def _feature_banding(
    data: pd.DataFrame,
    feature: str,
    feature_spec: pd.DataFrame,
    selected_feature: str | None = None,
    selected_banding: str = "0",
) -> str:
    """Banding used for a feature: the chart setting for the selected one, else spec, else a guess."""
    series = data[feature]
    if feature == selected_feature:
        banding = selected_banding.strip() or "0"
        if pd.api.types.is_numeric_dtype(series):
            try:
                float(banding)
            except ValueError:
                banding = "0"
        return banding
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if (feature_treatment_spec(feature_spec, feature) or {}).get("bins"):
            return "0"  # bins replace the banding, as in Charts
        configured = numeric_or_none(feature_lookup(feature_spec, feature).get("banding"))
        return str(configured if configured else banding_guesser(series))
    return "0"


def _banding_badge(banding: str) -> str:
    """Short banding label for a feature: 'none' for zero, otherwise the band width."""
    try:
        value = float(banding)
    except (TypeError, ValueError):
        return f"Banding: {banding}"
    return "Banding: none" if value == 0 else f"Banding: {value:g}"


def metrics_table_html(metrics: pd.DataFrame) -> str:
    """Render a metrics DataFrame (Dataset x metric) as a styled HTML table."""
    if metrics is None or metrics.empty:
        return '<div class="empty">No metrics available for the current filters.</div>'

    def fmt(
        column,
        value,
    ):
        """Format a metric value: four decimals (one for 'Within' columns), N/A for non-finite values."""
        if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
            if not np.isfinite(value):
                return "N/A"
            return f"{value:,.1f}" if str(column).startswith("Within ") else f"{value:,.4f}"
        return str(value)

    head = "".join(f"<th>{escape(str(c))}</th>" for c in metrics.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(fmt(c, v))}</td>" for c, v in zip(metrics.columns, row)) + "</tr>"
        for row in metrics.itertuples(index=False, name=None)
    )
    return f'<div class="table-wrap"><table class="metrics"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def overview_table_html(overview: pd.DataFrame) -> str:
    """Render the model overview (Statistic x Train / Test) as a styled HTML table."""
    if overview is None or overview.empty:
        return '<div class="empty">No model overview available for the current filters.</div>'

    def fmt(
        statistic,
        value,
    ):
        """Format an overview value: non-numeric values as text, non-finite numbers as N/A."""
        if not isinstance(value, (int, float, np.floating, np.integer)) or isinstance(value, bool):
            return str(value)
        if not np.isfinite(value):
            return "N/A"
        if statistic in ("Total Weight", "Total Response"):
            return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
        return f"{value:,.4f}"

    head = "".join(f"<th>{escape(str(c))}</th>" for c in overview.columns)
    rows = "".join(
        "<tr>"
        + f"<td>{escape(str(row[0]))}</td>"
        + "".join(f"<td>{escape(fmt(row[0], v))}</td>" for v in row[1:])
        + "</tr>"
        for row in overview.itertuples(index=False, name=None)
    )
    return f'<div class="table-wrap"><table class="metrics"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def text_table_html(
    table: pd.DataFrame,
    hover=None,
) -> str:
    """Render a table of text values (left aligned), e.g. GBM parameters or monotonicity.

    ``hover`` (a function of the first column's value) gives a hover text for each row's first cell."""
    if table is None or table.empty:
        return '<div class="empty">Nothing to show.</div>'
    head = "".join(f"<th>{escape(str(c))}</th>" for c in table.columns)

    def first_cell(value):
        """Render a first-column table cell, with an optional hover title."""
        title = f' title="{escape(hover(value), quote=True)}"' if hover else ""
        return f"<td{title}>{escape(str(value))}</td>"

    rows = "".join(
        "<tr>" + first_cell(row[0]) + "".join(f"<td>{escape(str(v))}</td>" for v in row[1:]) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    return (
        f'<div class="table-wrap"><table class="metrics text"><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _parameter_hover(name) -> str:
    """Hover text explaining a LightGBM parameter (from gbm.parameter_help)."""
    from .gbm import parameter_help

    return parameter_help(name)


def overview_section(
    overview: pd.DataFrame,
    note: str | None = None,
    tables: list[tuple[str, pd.DataFrame]] | None = None,
) -> dict:
    """Model overview tab: the summary table, then ``tables`` (title, DataFrame) side by side below it."""
    html = overview_table_html(overview)
    if tables:
        html += (
            '<div class="table-pair">'
            + "".join(
                f'<div><h3 class="plot-title">{escape(title)}</h3>'
                f'{text_table_html(table, hover=_parameter_hover if title == "GBM parameters" else None)}</div>'
                for title, table in tables
            )
            + "</div>"
        )
    return {
        "id": "overview",
        "title": _SECTION_TITLES["overview"],
        "kind": "html",
        "note": note,
        "html": html,
        "groups": [],
    }


def info_section(
    rows: list[tuple[str, str]],
    title: str = "Scoring data",
    section_id: str = "scoring-data",
    note: str | None = None,
) -> dict:
    """A section listing (item, value) details, e.g. the scoring data and the model used."""
    table = pd.DataFrame(rows, columns=["Item", "Value"])
    return {
        "id": section_id,
        "title": title,
        "kind": "html",
        "note": note,
        "html": text_table_html(table),
        "groups": [],
    }


def metrics_section(
    metrics: pd.DataFrame,
    note: str | None = None,
) -> dict:
    """Report section holding the metrics table."""
    return {
        "id": "metrics",
        "title": _SECTION_TITLES["metrics"],
        "kind": "html",
        "note": note,
        "html": metrics_table_html(metrics),
        "groups": [],
    }


def shap_section(
    data: pd.DataFrame,
    model,
    features: list[str],
    feature_spec: pd.DataFrame,
    weight: str,
    rows_mask,
    quantile: float = 0.05,
    note: str | None = None,
    pairs=None,
    selected_feature: str | None = None,
    selected_banding: str = "0",
    selected_limits=None,
) -> dict:
    """SHAP relativities and stacked mean SHAP for each feature of a GBM, one entry per feature.

    ``pairs`` is a list of (feature 1, feature 2): each adds a two-way SHAP heatmap, in a "Two-way" tab.
    ``selected_feature``, ``selected_banding`` and ``selected_limits`` are the Charts settings (banding, then
    (minimum, maximum, bins)) for the selected feature; the others use their specification, as the 1-way charts do."""
    mask = np.asarray(rows_mask) == 1
    entries = []
    for index, feature in enumerate(features, start=1):
        banding = _feature_banding(data, feature, feature_spec, selected_feature, selected_banding)
        minimum, maximum, bins = _feature_limits(data, feature, feature_spec, selected_feature, selected_limits)
        try:
            label = feature_label(feature_spec, feature)
            flame = shap_flame_figure(
                data,
                model,
                feature,
                weight,
                banding,
                mask,
                quantile=quantile,
                weight_label=weight,
                x_title=label,
                minimum=minimum,
                maximum=maximum,
                bins=bins,
            )
            stacked = stacked_shap_figure(
                data, model, feature, weight, banding, mask, x_title=label, minimum=minimum, maximum=maximum, bins=bins
            )
        except Exception as error:
            entries.append(
                {
                    "id": f"shap-{index:03d}",
                    "name": str(feature),
                    "badge": None,
                    "error": f"SHAP unavailable for {feature}: {error}",
                    "figs": [],
                }
            )
            continue
        _style_export_figure(flame, label, height=460)
        _style_export_figure(stacked, label, height=420)
        entries.append(
            _feature_entry(
                index,
                feature,
                [("SHAP relativity by band", flame), ("Stacked mean SHAP by band", stacked)],
                badge=_banding_badge(banding),
                prefix="shap",
            )
        )
    group = {
        "id": "shap",
        "title": getattr(model, "name", "GBM"),
        "features": entries,
        "empty": "No features with SHAP values. Rebuild the GBM with SHAP sampling switched on.",
    }
    groups = [group]
    if pairs:
        two_way = []
        for number, (first, second) in enumerate(pairs, start=1):
            try:
                heatmap = shap_heatmap_figure(
                    data,
                    model,
                    first,
                    second,
                    weight,
                    str(_feature_banding(data, first, feature_spec, selected_feature, selected_banding)),
                    str(_feature_banding(data, second, feature_spec, selected_feature, selected_banding)),
                    mask,
                    x_title=feature_label(feature_spec, first),
                    y_title=feature_label(feature_spec, second),
                    limits1=_feature_limits(data, first, feature_spec, selected_feature, selected_limits),
                    limits2=_feature_limits(data, second, feature_spec, selected_feature, selected_limits),
                )
                _style_export_figure(heatmap, feature_label(feature_spec, first), height=460)
                two_way.append(
                    _feature_entry(number, f"{first} x {second}", [("SHAP two-way by band", heatmap)], prefix="shap2")
                )
            except Exception as error:
                two_way.append(
                    {
                        "id": f"shap2-{number:03d}",
                        "name": f"{first} x {second}",
                        "badge": None,
                        "error": f"Two-way SHAP unavailable: {error}",
                        "figs": [],
                    }
                )
        groups.append(
            {"id": "shap-two-way", "title": "Two-way", "features": two_way, "empty": "No two-way SHAP views."}
        )
    return {"id": "shap", "title": _SECTION_TITLES["shap"], "kind": "plots", "note": note, "groups": groups}


def ale_section(
    data: pd.DataFrame,
    model,
    family: str,
    features: list[str],
    rows_mask,
    bins: int = 20,
    max_rows: int = 10000,
    note: str | None = None,
    feature_spec: pd.DataFrame | None = None,
    pairs=None,
) -> dict:
    """One-feature accumulated local effects for each feature of a fitted GLM or GBM.

    ``pairs`` is a list of (feature 1, feature 2): each adds a two-way ALE plot, in a "Two-way" tab."""
    from .ale import calculate_ale

    selected = np.asarray(rows_mask, dtype=bool).copy()
    if model.weight != "N":
        model_weights = pd.to_numeric(data[model.weight], errors="coerce").to_numpy(dtype=float)
        selected &= np.isfinite(model_weights) & (model_weights > 0)
    sample = data.loc[selected].copy()
    if len(sample) > int(max_rows):
        sample = sample.sample(n=int(max_rows), random_state=1)
    scale = "offset-free predictions" if family == "gbm" else "model predictions"
    summary = (
        f"ALE uses {len(sample):,} filtered rows, {int(bins)} bins and {scale}; " "observations are equally weighted."
    )
    entries = []
    for index, feature in enumerate(features, start=1):
        try:
            result = calculate_ale(sample, model, [feature], int(bins))
            label = feature_label(feature_spec, feature)
            figure = ale_figure(result, [feature], [label])
        except Exception as error:
            entries.append(
                {
                    "id": f"ale-{index:03d}",
                    "name": str(feature),
                    "badge": None,
                    "error": f"ALE unavailable for {feature}: {error}",
                    "figs": [],
                }
            )
            continue
        _style_export_figure(figure, label, height=500)
        entries.append(_feature_entry(index, feature, [(None, figure)], prefix="ale"))
    group = {"id": "ale", "title": getattr(model, "name", "Model"), "features": entries}
    groups = [group]
    if pairs:
        two_way = []
        for number, (first, second) in enumerate(pairs, start=1):
            try:
                result = calculate_ale(sample, model, [first, second], int(bins))
                figure = ale_figure(
                    result, [first, second], [feature_label(feature_spec, first), feature_label(feature_spec, second)]
                )
                _style_export_figure(figure, feature_label(feature_spec, first), height=500)
                two_way.append(_feature_entry(number, f"{first} x {second}", [(None, figure)], prefix="ale2"))
            except Exception as error:
                two_way.append(
                    {
                        "id": f"ale2-{number:03d}",
                        "name": f"{first} x {second}",
                        "badge": None,
                        "error": f"Two-way ALE unavailable: {error}",
                        "figs": [],
                    }
                )
        groups.append({"id": "ale-two-way", "title": "Two-way", "features": two_way, "empty": "No two-way ALE views."})
    return {
        "id": "ale",
        "title": _SECTION_TITLES["ale"],
        "kind": "plots",
        "note": f"{note + ' ' if note else ''}{summary}",
        "groups": groups,
    }


def _numeric_table_html(
    table: pd.DataFrame,
    decimals: dict[str, int] | None = None,
    default: int = 4,
) -> str:
    """Render a table whose numeric columns are formatted to ``decimals[column]`` places (``default`` otherwise)."""
    if table is None or table.empty:
        return '<div class="empty">Nothing to show.</div>'
    decimals = decimals or {}

    def fmt(
        column,
        value,
    ):
        """Format a numeric table cell with the column's decimal places. Non-finite numbers show as N/A."""
        if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
            return "N/A" if not np.isfinite(value) else f"{value:,.{decimals.get(column, default)}f}"
        return str(value)

    head = "".join(f"<th>{escape(str(c))}</th>" for c in table.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(fmt(c, v))}</td>" for c, v in zip(table.columns, row)) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    return f'<div class="table-wrap"><table class="metrics"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def _export_layout(
    figure,
    height: int,
    bottom: int = 60,
) -> None:
    """Apply the export styling (size, colours, fonts and margins) to a figure."""
    figure.update_layout(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color="#1f2d26"),
        margin=dict(l=60, r=60, t=60, b=bottom),
        hoverlabel=dict(bgcolor="#ffffff"),
    )
    figure.update_xaxes(gridcolor="#e3f0e6", linecolor="#cfe3d6")
    figure.update_yaxes(gridcolor="#e3f0e6", zerolinecolor="#cfe3d6")


def _style_lift_figure(figure) -> None:
    """Daisy colours by series name (every panel alike), so the legend matches both panels."""
    colours = {"Observed": _EXPORT_COLOURS[1], "Predicted": _EXPORT_COLOURS[0]}
    for trace in figure.data:
        if trace.type == "bar":
            trace.update(marker_color="rgba(233, 168, 59, 0.45)")
        elif trace.name in colours:
            trace.update(line=dict(color=colours[trace.name], width=2.5), marker=dict(color=colours[trace.name]))
        elif trace.name == "Portfolio mean":
            trace.update(line=dict(color="#5b6b61", dash="dot", width=1.5))
    _export_layout(figure, 520, bottom=100)


def _style_lorenz_figure(figure) -> None:
    """Style a Lorenz curve for export: the equality line is dotted."""
    for trace in figure.data:
        name = str(trace.name)
        if name.startswith("Equality"):
            trace.update(line=dict(color="#5b6b61", dash="dot", width=1.5))
        else:
            colour = _EXPORT_COLOURS[1] if name.startswith("Test") else _EXPORT_COLOURS[0]
            trace.update(line=dict(color=colour, width=2.5))
    figure.update_layout(title=None)
    _export_layout(figure, 460, bottom=100)


def _style_importance_figure(
    figure,
    rows: int,
) -> None:
    """Style a feature-importance bar chart for export, at the same height as the app."""
    figure.for_each_trace(lambda trace: trace.update(marker_color=_EXPORT_COLOURS[0]))
    _export_layout(figure, importance_height(rows))  # the same height as the app: every label fits
    figure.update_layout(margin=dict(l=140, r=40, t=30, b=60))
    figure.update_yaxes(automargin=True)  # the left margin grows for feature names longer than it


def _single_entry_section(
    section_id: str,
    group_title: str,
    entry: dict,
    note: str | None,
) -> dict:
    """Report section for a single entry, such as one plot set."""
    return {
        "id": section_id,
        "title": _SECTION_TITLES[section_id],
        "kind": "plots",
        "note": note,
        "single": True,
        "groups": [{"id": section_id, "title": group_title, "features": [entry]}],
    }


def lift_section(
    data: pd.DataFrame,
    model,
    weight: str,
    rows_mask,
    n_bins: int = 20,
    note: str | None = None,
) -> dict:
    """Lift Chart tab: binned actual vs predicted (train and test side by side), the Lorenz curve with its Gini
    and the actual-vs-expected table by bin, as in the app's Lift Chart tab."""
    name = getattr(model, "name", "Model")
    try:
        lift = lift_chart_figure(data, model, weight, rows_mask, n_bins)
        lorenz = lorenz_figure(data, model, weight, rows_mask)
        _style_lift_figure(lift)
        _style_lorenz_figure(lorenz)
        table = lift_table_frame(data, model, weight, rows_mask, n_bins)
        entry = _feature_entry(1, "Lift chart", [(None, lift), ("Lorenz curve", lorenz)], prefix="lift")
        entry["html"] = '<h3 class="plot-title">Actual vs expected by bin</h3>' + _numeric_table_html(
            table, {"Bin": 0, "Exposure": 2, "Actual": 2, "Predicted": 2, "A/E": 3, "Lift": 3}
        )
    except Exception as error:
        entry = _error_entry(1, "Lift chart", f"Lift chart unavailable: {error}", "lift")
    return _single_entry_section("lift", name, entry, note)


def importance_frame(
    data: pd.DataFrame,
    model,
    family: str,
) -> pd.DataFrame:
    """Feature importance, most important first: columns ``Feature`` and ``value`` (GBM gain %, or a GLM's
    mean |linear predictor contribution| x 100). ``value`` is also the column name for the table."""
    if family == "gbm":
        imp = model.importances[["Feature", "Gain", "Split"]].copy()
        return imp.sort_values("Gain", ascending=False).reset_index(drop=True)
    contributions = model.feature_contributions(data.iloc[model.pred_rows])
    mean_abs = contributions.abs().mean().sort_values(ascending=False) * 100
    return pd.DataFrame(
        {"Feature": [c.replace("glm_LP_", "") for c in mean_abs.index], "mean |LP| x 100": mean_abs.to_numpy()}
    )


def importance_section(
    data: pd.DataFrame,
    model,
    family: str,
    note: str | None = None,
) -> dict:
    """Feature importance tab: GBM gain %, or a GLM's mean |linear predictor contribution| x 100, as a bar chart
    with the table beneath it (the same as the app's Feature importance tab; it does not follow the row filter)."""
    name = getattr(model, "name", "Model")
    try:
        imp = importance_frame(data, model, family)
        value, axis = ("Gain", "gain %") if family == "gbm" else ("mean |LP| x 100", "mean |LP| x 100")
        figure = importance_figure(imp, x_title=axis, value=value)
        _style_importance_figure(figure, len(imp))  # every feature, as in the app
        entry = _feature_entry(1, "Feature importance", [(None, figure)], prefix="importance")
        entry["html"] = _numeric_table_html(imp)
    except Exception as error:
        entry = _error_entry(1, "Feature importance", f"Feature importance unavailable: {error}", "importance")
    return _single_entry_section("importance", name, entry, note)


def line_bar_section(
    data: pd.DataFrame,
    features: list[str],
    modelled: list[str] | None,
    feature_spec: pd.DataFrame,
    response: str,
    weight: str,
    add_cols: list[str],
    rows_mask,
    low_exposure: str,
    show_bars: bool,
    sigma_bars: bool,
    selected_feature: str | None = None,
    selected_banding: str = "0",
    valid_features: list[str] | None = None,
    y_title: str | None = None,
    per_unit_cols=(),
    selected_limits=None,
) -> dict:
    """The 1-way line and bar section. With a model: Modelled tab + Non-modelled tab (valid features only)."""
    modelled_set = set(modelled) if modelled is not None else set()
    valid_set = set(valid_features) if valid_features is not None else None
    group_ids = ["all"] if modelled is None else ["modelled", "non-modelled"]
    groups = {group_id: [] for group_id in group_ids}
    for index, feature in enumerate(features, start=1):
        if modelled is None:
            group = "all"
        elif feature in modelled_set:
            group = "modelled"
        elif valid_set is None or feature in valid_set:
            group = "non-modelled"
        else:
            continue  # non-modelled and not modelling-valid (e.g. response, weight, train/test flag)
        banding = _feature_banding(data, feature, feature_spec, selected_feature, selected_banding)
        try:
            group_by = "__daisy_chart_group__"
            while group_by in data.columns:
                group_by += "_"
            chart_data = data.assign(**{group_by: data[feature]})
            minimum, maximum, bins = _feature_limits(data, feature, feature_spec, selected_feature, selected_limits)
            summary = one_way_summary(
                chart_data,
                response,
                weight,
                group_by,
                add_cols,
                banding=banding,
                group_low_exposure=low_exposure,
                total_filter=rows_mask,
                per_unit_cols=per_unit_cols,
                minimum=minimum,
                maximum=maximum,
                bins=bins,
            )
            figure = line_bar_figure(
                summary,
                group_by,
                response,
                add_cols,
                show_bars=show_bars,
                sigma_bars=True,
                title=feature,  # the page's tickbox shows them
                y_title=y_title,
                weight_label=weight,
                x_title=feature_label(feature_spec, feature),
            )
            _style_export_figure(figure, feature_label(feature_spec, feature))
            # the badge above each 1-way chart: its banding and the monotonicity constraint from the specification
            badge = f"{_banding_badge(banding)} \u00b7 Monotonicity: {feature_monotonicity(feature_spec, feature) or 'none'}"
            groups[group].append(_feature_entry(index, feature, [(None, figure)], badge=badge))
        except Exception as error:
            groups[group].append(_error_entry(index, feature, f"Chart unavailable for {feature}: {error}", "plot"))
    return {
        "id": "lines",
        "title": _SECTION_TITLES["lines"],
        "kind": "plots",
        "note": None,
        "groups": [{"id": g, "title": _GROUP_TITLES[g], "features": groups[g]} for g in group_ids],
    }


def _merge_views(by_view: dict) -> list[dict]:
    """Fold per-view section lists into the first view's sections (each feature gets a ``views`` map)."""
    names = list(by_view)
    base = by_view[names[0]]
    for name in names[1:]:
        for base_section, view_section in zip(base, by_view[name]):
            if view_section is base_section:
                continue
            if base_section["kind"] == "html":
                base_section.setdefault("views", {})[name] = view_section["html"]
                continue
            for base_group, view_group in zip(base_section["groups"], view_section["groups"]):
                for base_feature, view_feature in zip(base_group["features"], view_group["features"]):
                    base_feature.setdefault("views", {})[name] = {
                        "badge": view_feature.get("badge"),
                        "error": view_feature.get("error"),
                        "figs": view_feature["figs"],
                        "html": view_feature.get("html"),
                    }
    return base


def render_report_html(
    sections: list[dict],
    response: str,
    weight: str,
    subtitle: str,
    model_name: str | None = None,
    views: list[str] | None = None,
    default_view: str | None = None,
    logo_path: str | Path | None = None,
    app_title: str = "Daisy",
    error_bars: bool = True,
) -> bytes:
    """Wrap sections into one self-contained, Daisy-themed HTML page (UTF-8 bytes).

    ``logo_path`` overrides the logo read from the ``images`` folder; ``app_title`` is the title of the app,
    used as the report title."""
    from plotly.offline import get_plotlyjs

    texts = [f"Response: {response}", f"Weight: {'none (N)' if weight == 'N' else weight}"]
    if model_name:
        texts.append(f"Model: {model_name}")
    texts.append(f"Generated {datetime.now():%d %b %Y %H:%M}")
    chips = "".join(f'<span class="chip">{escape(text)}</span>' for text in texts)
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(app_title)}</title>"
        f"<style>{_EXPORT_CSS}</style></head><body>"
        f'<header><div class="wrap"><div class="brand">{logo_html(logo_path)}'
        f"<div><h1>{escape(app_title)}</h1><p>{escape(subtitle)}</p></div></div>"
        f'<div class="chips">{chips}</div></div></header>'
        '<main class="wrap"><div class="sections"><div id="sections" class="section-list" role="tablist"></div>'
        '<div class="views" id="views-wrap"><span>Rows</span><div class="seg" id="views"></div></div></div>'
        '<div class="tabs" id="tabs" role="tablist"></div>'
        '<section class="panel"><p class="note" id="note" style="display:none"></p>'
        '<div class="controls" id="controls">'
        '<div class="field"><label for="feature">Feature</label><select id="feature"></select></div>'
        '<div class="field"><label><input type="checkbox" id="errbars"> Error bars</label></div>'
        '<div class="nav"><button class="btn" id="prev" type="button" aria-label="Previous feature">&larr; Prev</button>'
        '<button class="btn" id="next" type="button" aria-label="Next feature">Next &rarr;</button></div>'
        '<div class="meta"><span class="badge" id="badge"></span><span id="position"></span></div></div>'
        '<div id="plots"></div><div class="empty" id="empty" style="display:none"></div>'
        '<div id="static" style="display:none"></div></section></main>'
        "<footer>Tip: use the &larr; / &rarr; keys to step through features. Charts reflect the filters and "
        "settings that were active when exported.</footer>"
        f"<script>{get_plotlyjs()}</script>"
        f'<script id="daisy-data" type="application/json">{_script_safe_json({"sections": sections, "views": views or ["All"], "defaultView": default_view, "errorBars": bool(error_bars)})}</script>'
        f"<script>{_EXPORT_JS}</script></body></html>"
    )
    return html.encode("utf-8")


def export_line_bar_plots(
    data: pd.DataFrame,
    features: list[str],
    modelled: list[str] | None,
    feature_spec: pd.DataFrame,
    response: str,
    weight: str,
    add_cols: list[str],
    rows_mask,
    low_exposure: str,
    show_bars: bool,
    sigma_bars: bool,
    selected_feature: str | None = None,
    selected_banding: str = "0",
    valid_features: list[str] | None = None,
    extra_sections: list | None = None,
    model_name: str | None = None,
    view_masks: dict | None = None,
    default_view: str = "All",
    y_title: str | None = None,
    leading_sections: list | None = None,
    app_title: str = "Daisy",
    per_unit_cols=(),
    selected_limits=None,
) -> bytes:
    """Build one self-contained HTML page (UTF-8 bytes) with the 1-way plots and any extra sections.

    The first tab holds the 1-way line and bar plots: with a model, a Modelled tab and a Non-modelled
    tab (restricted to ``valid_features`` when given), otherwise a single "All features" tab. Each has a
    feature drop-down. ``extra_sections`` (SHAP, ALE, Metrics - see :func:`shap_section`,
    :func:`ale_section`, :func:`metrics_section`) become further top-level tabs; each item is either a
    finished section or a callable ``f(rows_mask) -> section`` that is evaluated once per view.

    ``view_masks`` maps a view name ("All", "Train", "Test") to its row mask; the page then offers a
    Rows switch and every plot is drawn for each view. The first entry is the base view and
    ``default_view`` the one shown on opening. Without it, ``rows_mask`` is used for a single view.
    ``app_title`` is the title of the app, used as the report title. ``y_title`` (the KPI name) is the
    primary y-axis title of the 1-way plots. ``leading_sections`` are
    placed before the 1-way plots (e.g. the model overview, then Metrics); like ``extra_sections`` each item is a
    finished section or a callable ``f(rows_mask) -> section``.
    ``per_unit_cols`` names the response / prediction columns that are per unit of weight rather than totals.
    """
    view_masks = dict(view_masks) if view_masks else {"All": rows_mask}
    by_view = {}
    for view, mask in view_masks.items():
        line_section = line_bar_section(
            data,
            features,
            modelled,
            feature_spec,
            response,
            weight,
            add_cols,
            mask,
            low_exposure,
            show_bars,
            sigma_bars,
            selected_feature,
            selected_banding,
            valid_features,
            y_title,
            per_unit_cols,
            selected_limits,
        )
        by_view[view] = (
            [item(mask) if callable(item) else item for item in leading_sections or []]
            + [line_section]
            + [item(mask) if callable(item) else item for item in extra_sections or []]
        )
    sections = _merge_views(by_view)
    line_groups = next(section for section in sections if section["id"] == "lines")["groups"]
    counts = {g["id"]: len(g["features"]) for g in line_groups}
    subtitle = (
        f"{counts['all']} features"
        if "all" in counts
        else f"{counts['modelled']} modelled \u00b7 {counts['non-modelled']} non-modelled features"
    )
    names = list(view_masks)
    return render_report_html(
        sections,
        response,
        weight,
        f"Charts export \u00b7 {subtitle}",
        model_name,
        views=names,
        default_view=default_view if default_view in names else names[0],
        app_title=app_title,
        error_bars=sigma_bars,
    )


def format_metric_value(
    value,
    metric: str,
) -> str:
    """A metric value as shown in the metrics comparison tables."""
    if not np.isfinite(value):
        return "N/A"
    if metric.startswith("Within "):
        return f"{value:.1f}%"
    return f"{value:,.5g}"


def format_metric_change(
    value,
    metric: str,
) -> str:
    """A metric change as shown in the metrics comparison tables (% for GINI, RMSE and the Within bands)."""
    if not np.isfinite(value):
        return "N/A"
    if metric == "PoDE":
        return f"{value:+.4f}"
    if metric in {"GINI", "RMSE"} or metric.startswith("Within "):
        return f"{value:+.1f}%"
    return format_metric_value(value, metric)


_STATUS_COLOURS = {"Improved": "#1e7b4a", "Deteriorated": "#b03a2e"}


def comparison_table_html(
    comparison: pd.DataFrame,
    new_label: str,
    old_label: str,
) -> str:
    """The metrics comparison (new against old, with the change and whether it improved) as an HTML table."""
    headers = ["Dataset", "Metric", new_label, old_label, "Change", "Result"]
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    rows = []
    for row in comparison.itertuples(index=False):
        colour = _STATUS_COLOURS.get(row.Status, "inherit")
        rows.append(
            f"<tr><td>{escape(str(row.Dataset))}</td><td>{escape(str(row.Metric))}</td>"
            f"<td>{escape(format_metric_value(row.New, row.Metric))}</td>"
            f"<td>{escape(format_metric_value(row.Old, row.Metric))}</td>"
            f'<td style="color:{colour}">{escape(format_metric_change(row.Change, row.Metric))}</td>'
            f'<td style="color:{colour}">{escape(str(row.Status))}</td></tr>'
        )
    return (
        f'<div class="table-wrap"><table class="metrics"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def overlay_report_html(
    overlays: list[dict],
    metrics_groups: list[dict],
    weight: str,
    subtitle: str,
    app_title: str = "Daisy",
) -> bytes:
    """One self-contained HTML report with every saved overlay and the metrics movement on the test rows.

    ``metrics_groups`` (one per base model) holds: base_name, overlays (their names), comparison (DataFrame),
    new_label and old_label. A dropdown picks an overlay to see its details, test-rows plot and relativities.
    Each item of ``overlays`` holds: name, base_name, feature, response, info_rows, relativities (DataFrame),
    figure and plot_note."""
    metrics_html = []
    for group in metrics_groups:
        names = ", ".join(escape(str(n)) for n in group["overlays"])
        metrics_html.append(
            f'<h3 class="plot-title">{escape(group["base_name"])}: {names}</h3>'
            f'{comparison_table_html(group["comparison"], group["new_label"], group["old_label"])}'
        )
    sections = [
        {
            "id": "overlay-metrics",
            "title": "Metrics movement with all overlays",
            "kind": "html",
            "note": "Each base model's overlays are applied together. Change is the with-overlays value minus the "
            "base value; GINI and RMSE changes are in %.",
            "html": "".join(metrics_html) or '<div class="empty">No metrics available.</div>',
            "groups": [],
        }
    ]
    # one table with a row per overlay (its details, without the rule)
    details = pd.DataFrame([dict(item["info_rows"]) for item in overlays]) if overlays else pd.DataFrame()
    sections.append(
        {
            "id": "overlay-features",
            "title": "Overlay Details",
            "kind": "html",
            "note": None,
            "html": text_table_html(details) if not details.empty else '<div class="empty">No overlays.</div>',
            "groups": [],
        }
    )
    entries = []
    for index, item in enumerate(overlays, start=1):
        entry = _feature_entry(
            index, item["name"], [("Test rows", item["figure"])], badge=item["base_name"], prefix="overlay"
        )
        note = f'<p class="note">{escape(item["plot_note"])}</p>' if item.get("plot_note") else ""
        entry["html"] = f'{note}<h3 class="plot-title">Relativities</h3>{text_table_html(item["relativities"])}'
        entries.append(entry)
    sections.append(
        {
            "id": "overlays",
            "title": "AvE Plots and Relativities",
            "kind": "plots",
            "single": False,
            "note": "Choose an overlay in the dropdown (or use Prev and Next) to see its details, plot and "
            "relativities.",
            "groups": [{"id": "overlays", "title": "Saved overlays", "features": entries}],
        }
    )
    ae_entries = []
    for index, item in enumerate(overlays, start=1):
        if item.get("ae_figures"):
            figs = list(zip(["A/E by band before relativities", "A/E by band after relativities"], item["ae_figures"]))
            ae_entries.append(_feature_entry(index, item["name"], figs, badge=item["base_name"], prefix="overlay-ae"))
    if ae_entries:
        sections.append(
            {
                "id": "overlay-ae",
                "title": "Bootstrapping",
                "kind": "plots",
                "single": False,
                "note": "A/E by band on the training rows the suggestions use, before and after the "
                "relativities. Choose an overlay in the dropdown.",
                "groups": [{"id": "overlay-ae", "title": "Saved overlays", "features": ae_entries}],
            }
        )
    responses = sorted({str(item["response"]) for item in overlays})
    return render_report_html(
        sections, response=", ".join(responses), weight=weight, subtitle=subtitle, app_title=app_title, error_bars=False
    )
