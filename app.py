import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Panama Electricity Demand", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

DATA = Path(__file__).parent / "data"

# ------------------------------------------------------------------ design tokens
INK, MUTED, LINE, GRID, ACCENT = "#0F172A", "#64748B", "#E2E8F0", "#EEF2F7", "#2563EB"
FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
FAMILY_COLORS = {
    "Baseline": "#94A3B8", "ML klasik": "#2563EB", "Neural net": "#7C3AED",
    "Foundation model": "#0D9488", "Statistik": "#D97706",
}
DAYTYPE_COLORS = {"Weekday": "#2563EB", "Weekend": "#0D9488", "Holiday": "#D97706"}
SERIES = ["#2563EB", "#0D9488", "#7C3AED", "#D97706", "#DB2777", "#0891B2", "#65A30D", "#475569"]
SEQUENTIAL = [[0, "#EFF6FF"], [0.5, "#60A5FA"], [1, "#1E3A8A"]]
DIVERGING = [[0, "#B91C1C"], [0.5, "#F8FAFC"], [1, "#1D4ED8"]]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _stretch():
    # Streamlit >= 1.50 memakai width="stretch"; versi lama memakai use_container_width
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


STRETCH = _stretch()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button,
.stApp input, .stApp textarea, .stApp [data-baseweb="tab"], .stApp [data-testid="stMarkdownContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background: #FFFFFF; }
.block-container { max-width: 1360px; padding: 1.4rem 2.2rem 3rem; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
h1, h2, h3, h4 { color: #0F172A; letter-spacing: -0.01em; }

/* top bar */
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
          padding: 18px 22px; margin-bottom: 14px; border: 1px solid #E2E8F0; border-radius: 16px;
          background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 55%, #FDF4FF 100%); }
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark { width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg,#2563EB,#7C3AED); box-shadow: 0 4px 12px rgba(37,99,235,.35); display: flex;
              align-items: center; justify-content: center; }
.brand-title { font-size: 1.2rem; font-weight: 650; color: #0F172A; line-height: 1.2; margin: 0; }
.brand-sub { font-size: .82rem; color: #64748B; margin: 2px 0 0; }
.pill { display: inline-block; padding: 4px 11px; border: 1px solid #E2E8F0; border-radius: 999px;
        font-size: .76rem; color: #475569; background: #FFFFFF; margin-left: 6px; }
.pill b { color: #0F172A; font-weight: 600; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; font-weight: 500; color: #64748B; background: transparent; }
.stTabs [aria-selected="true"] { color: #0F172A; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2563EB; height: 2px; }
.stTabs [data-baseweb="tab-border"] { background-color: transparent; }

/* KPI cards */
.kpi { background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #2563EB; border-radius: 12px; padding: 16px 18px; height: 100%;
       box-shadow: 0 1px 2px rgba(15,23,42,.04); transition: box-shadow .15s, transform .15s; }
.kpi:hover { box-shadow: 0 6px 18px rgba(15,23,42,.09); transform: translateY(-1px); }
.kpi-label { font-size: .7rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: #64748B; }
.kpi-value { font-size: 1.7rem; font-weight: 650; color: #0F172A; margin-top: 8px; line-height: 1.1;
             font-variant-numeric: tabular-nums; }
.kpi-value.small { font-size: 1.02rem; line-height: 1.35; margin-top: 10px; font-weight: 600; }
.kpi-sub { font-size: .8rem; color: #64748B; margin-top: 8px; min-height: 1.3em; }
.chip { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .72rem; font-weight: 600; margin-right: 6px; }
.chip.good { background: #DCFCE7; color: #166534; }
.chip.bad { background: #FEE2E2; color: #991B1B; }
.chip.neutral { background: #F1F5F9; color: #475569; }

/* cards (bordered containers) */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
.card-title { font-size: .95rem; font-weight: 600; color: #0F172A; margin: 2px 0 0; }
.card-sub { font-size: .8rem; color: #64748B; margin: 2px 0 6px; }
.section { font-size: 1.05rem; font-weight: 650; color: #0F172A; margin: 22px 0 2px; }
.section-sub { font-size: .85rem; color: #64748B; margin: 0 0 10px; }
.spacer { height: 14px; }
.about li { margin-bottom: 4px; color: #334155; }
.stDownloadButton button { border-radius: 8px; border: 1px solid #E2E8F0; background: #FFFFFF; color: #0F172A; font-weight: 500; }
.stDownloadButton button:hover { border-color: #2563EB; color: #2563EB; }
</style>
""", unsafe_allow_html=True)

esc = html.escape


# ------------------------------------------------------------------ helpers
_plot_n = [0]


def plot(fig):
    _plot_n[0] += 1
    st.plotly_chart(fig, key=f"plot_{_plot_n[0]}", theme=None, config={"displayModeBar": False}, **STRETCH)


def table(df, **kw):
    st.dataframe(df, hide_index=True, **STRETCH, **kw)


def style(fig, height=360, legend=True, hover="x unified"):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color="#334155"),
        hovermode=hover, showlegend=legend,
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color="white", family=FONT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="",
                    font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE, tickcolor=LINE, ticks="outside")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


def kpi(col, label, value, sub="", chip=None, kind="neutral", small=False):
    chip_html = f'<span class="chip {kind}">{esc(chip)}</span>' if chip else ""
    cls = "kpi-value small" if small else "kpi-value"
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{esc(label)}</div><div class="{cls}">{value}</div>'
        f'<div class="kpi-sub">{chip_html}{sub}</div></div>', unsafe_allow_html=True)


def card_title(title, sub=None):
    st.markdown(f'<div class="card-title">{esc(title)}</div>' + (f'<div class="card-sub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f'<div class="section">{esc(title)}</div>' + (f'<div class="section-sub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def spacer():
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_csv(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def prepare():
    cmp, hor, fc = load_csv("model_comparison.csv"), load_csv("metrics_by_horizon.csv"), load_csv("forecast_outputs_test.csv")
    wx, eda = load_csv("stage1_weather_metrics.csv"), load_csv("eda_dataset.csv")
    wf = load_csv("stage1_weather_forecast_test.csv")
    if not wf.empty:
        wf["forecast_origin"] = pd.to_datetime(wf["forecast_origin"], errors="coerce")
        wf["target_datetime"] = pd.to_datetime(wf["target_datetime"], errors="coerce")

    for c in cmp.columns:
        if c not in ("Model", "Stage 1", "Keluarga"):
            cmp[c] = pd.to_numeric(cmp[c], errors="coerce")
    for c in ("Horizon_h", "MAE", "RMSE"):
        if c in hor:
            hor[c] = pd.to_numeric(hor[c], errors="coerce")
    if not fc.empty:
        fc["forecast_origin"] = pd.to_datetime(fc["forecast_origin"], errors="coerce")
        fc["target_datetime"] = pd.to_datetime(fc["target_datetime"], errors="coerce")
        for c in fc.columns:
            if c not in ("forecast_origin", "target_datetime"):
                fc[c] = pd.to_numeric(fc[c], errors="coerce")
        fc = fc.dropna(subset=["forecast_origin", "target_datetime"]).sort_values("target_datetime").reset_index(drop=True)
    if not eda.empty and "datetime" in eda:
        eda["datetime"] = pd.to_datetime(eda["datetime"], errors="coerce")
        for c in eda.columns:
            if c != "datetime":
                eda[c] = pd.to_numeric(eda[c], errors="coerce")
        eda = eda.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        eda["year"] = eda["datetime"].dt.year
        eda["month"] = eda["datetime"].dt.month
        eda["hour"] = eda["datetime"].dt.hour
        eda["dow"] = eda["datetime"].dt.dayofweek
        hol = eda["holiday"] if "holiday" in eda else 0
        eda["day_type"] = np.where(hol == 1, "Holiday", np.where(eda["dow"] >= 5, "Weekend", "Weekday"))
    return cmp, hor, fc, wx, eda, wf


cmp, hor, fc, wx, eda, wf = prepare()
if cmp.empty or fc.empty:
    st.error("Output notebook belum lengkap. Pastikan CSV hasil notebook ada di folder data/.")
    st.stop()

BASE_COLS = ("forecast_origin", "target_datetime", "horizon", "actual_nat_demand")
model_cols = [c for c in fc.columns if c not in BASE_COLS]
rank = cmp.sort_values("RMSE_test").reset_index(drop=True)
best = rank.iloc[0]
best_model = str(best["Model"])
naive_rows = cmp[cmp["Model"].str.startswith("Naive 24")]
naive = naive_rows.iloc[0] if not naive_rows.empty else None
naive_col = str(naive["Model"]) if naive is not None else None


def vs_naive(col):
    if naive is None or not np.isfinite(naive[col]) or naive[col] == 0:
        return None, "neutral"
    d = (best[col] / naive[col] - 1) * 100
    return f"{d:+.1f}% vs naive 24h", ("good" if d < -0.05 else "bad" if d > 0.05 else "neutral")


# ------------------------------------------------------------------ top bar
period = f"{fc['target_datetime'].min():%d %b %Y} – {fc['target_datetime'].max():%d %b %Y}"
BOLT = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/></svg>')
st.markdown(f"""
<div class="topbar">
  <div class="brand"><div class="brand-mark">{BOLT}</div>
    <div><div class="brand-title">Panama Electricity Demand</div>
    <div class="brand-sub">24-hour ahead forecasting · weather-informed two-stage models</div></div></div>
  <div><span class="pill">Test <b>{period}</b></span><span class="pill"><b>{cmp['Model'].nunique()}</b> models</span>
  <span class="pill"><b>{fc['forecast_origin'].nunique():,}</b> forecast origins</span></div>
</div>
""", unsafe_allow_html=True)

tab_ov, tab_mod, tab_fc, tab_err, tab_eda, tab_wx, tab_about = st.tabs(
    ["Overview", "Models", "Forecasts", "Error analysis", "Data explorer", "Weather (Stage 1)", "About"])

# ================================================================== OVERVIEW
with tab_ov:
    spacer()
    k = st.columns(5)
    kpi(k[0], "Best model", esc(best_model), sub=f"{esc(str(best['Keluarga']))} · Stage 1: {esc(str(best['Stage 1']))}", small=True)
    txt, kind = vs_naive("RMSE_test")
    kpi(k[1], "RMSE", f"{best['RMSE_test']:,.1f}", sub="MWh", chip=txt, kind=kind)
    txt, kind = vs_naive("MAE_test")
    kpi(k[2], "MAE", f"{best['MAE_test']:,.1f}", sub="MWh", chip=txt, kind=kind)
    txt, kind = vs_naive("MAPE_test (%)")
    kpi(k[3], "MAPE", f"{best['MAPE_test (%)']:,.2f}%", chip=txt, kind=kind)
    kpi(k[4], "R²", f"{best['R2_test']:.3f}", sub=(f"Naive 24h: {naive['R2_test']:.3f}" if naive is not None else ""))
    spacer()

    _ae = (fc[best_model] - fc["actual_nat_demand"]).abs()
    _bh = _ae.groupby(fc["target_datetime"].dt.hour).mean()
    _bd = _ae.groupby(fc["target_datetime"].dt.date).mean()
    ik = st.columns(4)
    kpi(ik[0], "Easiest hour", f"{int(_bh.idxmin()):02d}:00", sub=f"MAE {_bh.min():,.1f} MWh", small=True)
    kpi(ik[1], "Hardest hour", f"{int(_bh.idxmax()):02d}:00", sub=f"MAE {_bh.max():,.1f} MWh", small=True)
    kpi(ik[2], "Best day", f"{_bd.idxmin():%d %b %Y}", sub=f"MAE {_bd.min():,.1f} MWh", small=True)
    kpi(ik[3], "Worst day", f"{_bd.idxmax():%d %b %Y}", sub=f"MAE {_bd.max():,.1f} MWh", small=True)
    spacer()

    left, right = st.columns([2.2, 1])
    with left:
        with st.container(border=True):
            h1, h2 = st.columns([1.6, 1])
            with h1:
                card_title("Actual vs forecast", best_model)
            win = h2.radio("Window", ["1 week", "1 month", "Full test"], index=1, horizontal=True,
                           label_visibility="collapsed", key="ov_window")
            d = fc[["target_datetime", "actual_nat_demand", best_model]].dropna()
            if win != "Full test":
                d = d.head(24 * (7 if win == "1 week" else 30))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["target_datetime"], y=d["actual_nat_demand"], name="Actual", mode="lines",
                                     line=dict(color=INK, width=1.6)))
            fig.add_trace(go.Scatter(x=d["target_datetime"], y=d[best_model], name="Forecast", mode="lines",
                                     line=dict(color=ACCENT, width=1.6)))
            marker = pd.Timestamp("2020-03-01")
            if d["target_datetime"].min() < marker < d["target_datetime"].max():
                fig.add_shape(type="line", x0=marker.isoformat(), x1=marker.isoformat(), xref="x", y0=0, y1=1, yref="paper",
                              line=dict(color=MUTED, width=1, dash="dot"))
                fig.add_annotation(x=marker.isoformat(), y=1, yref="paper", text="Mar 2020", showarrow=False,
                                   xanchor="left", yanchor="bottom", font=dict(size=10, color=MUTED))
            style(fig, height=350)
            fig.update_yaxes(title_text="Demand (MWh)")
            plot(fig)
    with right:
        with st.container(border=True):
            card_title("RMSE by model", "Top 8 · lower is better")
            top = rank.head(8).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["RMSE_test"], y=top["Model"], orientation="h",
                marker_color=[FAMILY_COLORS.get(str(f), ACCENT) for f in top["Keluarga"]],
                text=[f"{v:,.1f}" for v in top["RMSE_test"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>RMSE %{x:,.2f}<extra></extra>"))
            style(fig, height=350, legend=False, hover="closest")
            fig.update_xaxes(visible=False)
            fig.update_yaxes(automargin=True)
            plot(fig)

    spacer()
    with st.container(border=True):
        card_title("Leaderboard", "Ranked by test RMSE · error metrics in MWh")
        lb = rank.copy()
        lb.insert(0, "#", range(1, len(lb) + 1))
        cols = [c for c in ["#", "Model", "Keluarga", "Stage 1", "RMSE_test", "MAE_test", "MAPE_test (%)", "R2_test"] if c in lb]
        cc = st.column_config
        table(lb[cols], column_config={
            "#": cc.NumberColumn("#", width="small"),
            "Keluarga": cc.TextColumn("Family"),
            "RMSE_test": cc.ProgressColumn("RMSE", format="%.2f", min_value=0.0, max_value=float(lb["RMSE_test"].max())),
            "MAE_test": cc.NumberColumn("MAE", format="%.2f"),
            "MAPE_test (%)": cc.NumberColumn("MAPE (%)", format="%.2f"),
            "R2_test": cc.NumberColumn("R²", format="%.3f"),
        })

# ================================================================== MODELS
with tab_mod:
    spacer()
    metric_map = {"RMSE": "RMSE_test", "MAE": "MAE_test", "MAPE": "MAPE_test (%)", "R²": "R2_test"}
    mlabel = st.radio("Metric", list(metric_map), horizontal=True, key="mod_metric", label_visibility="collapsed")
    mcol = metric_map[mlabel]
    higher_better = mcol == "R2_test"

    a, b = st.columns([1.25, 1])
    with a:
        with st.container(border=True):
            card_title(f"{mlabel} by model", "Higher is better" if higher_better else "Lower is better")
            order = cmp.sort_values(mcol, ascending=not higher_better)
            fig = px.bar(order, x=mcol, y="Model", color="Keluarga", orientation="h", color_discrete_map=FAMILY_COLORS,
                         category_orders={"Model": list(order["Model"][::-1])})
            style(fig, height=max(380, 30 * len(order)), hover="closest")
            fig.update_xaxes(title_text=mlabel)
            fig.update_yaxes(title_text="", automargin=True)
            plot(fig)
    with b:
        with st.container(border=True):
            card_title("Stability across 2020", "RMSE before vs after March 2020 · top 8 models")
            if {"RMSE_JanFeb20", "RMSE_MarJun20"}.issubset(cmp.columns):
                s = cmp.sort_values("RMSE_test").head(8)
                sm = s.melt(id_vars="Model", value_vars=["RMSE_JanFeb20", "RMSE_MarJun20"], var_name="Period", value_name="RMSE")
                sm["Period"] = sm["Period"].map({"RMSE_JanFeb20": "Jan–Feb 2020", "RMSE_MarJun20": "Mar–Jun 2020"})
                fig = px.bar(sm, x="RMSE", y="Model", color="Period", orientation="h", barmode="group",
                             color_discrete_map={"Jan–Feb 2020": "#94A3B8", "Mar–Jun 2020": ACCENT},
                             category_orders={"Model": list(s["Model"][::-1])})
                style(fig, height=max(380, 30 * len(order)), hover="closest")
                fig.update_yaxes(title_text="", automargin=True)
                plot(fig)
            else:
                st.info("Kolom RMSE per periode tidak tersedia.")

    spacer()
    with st.container(border=True):
        h1, h2 = st.columns([2.2, 1])
        with h1:
            card_title("Error by forecast horizon", "How accuracy changes from 1 to 24 hours ahead")
        hmetric = h2.radio("Horizon metric", ["RMSE", "MAE"], horizontal=True, key="mod_hmetric", label_visibility="collapsed")
        if hor.empty:
            st.info("Metrik per horizon belum tersedia.")
        else:
            default = [m for m in list(rank["Model"].head(5)) + ([naive_col] if naive_col else []) if m in set(hor["Model"])]
            chosen = st.multiselect("Models", sorted(hor["Model"].unique()), default=list(dict.fromkeys(default)), key="mod_hmodels")
            hh = hor[hor["Model"].isin(chosen)]
            if hh.empty:
                st.info("Pilih minimal satu model.")
            else:
                fig = px.line(hh, x="Horizon_h", y=hmetric, color="Model", markers=True, color_discrete_sequence=SERIES)
                fig.update_traces(line=dict(width=2), marker=dict(size=5))
                style(fig, height=380)
                fig.update_xaxes(title_text="Horizon (hours ahead)", dtick=2)
                fig.update_yaxes(title_text=f"{hmetric} (MWh)")
                plot(fig)

    with st.expander("Full metrics table"):
        table(cmp.sort_values("RMSE_test").round(4))
        st.download_button("Download model_comparison.csv", cmp.to_csv(index=False).encode(), "model_comparison.csv", "text/csv", key="dl_cmp")

# ================================================================== FORECASTS
with tab_fc:
    spacer()
    origin_dates = sorted(fc["forecast_origin"].dt.date.unique())
    if "fc_date" not in st.session_state:
        st.session_state["fc_date"] = origin_dates[-1]

    def _shift(step):
        cur = st.session_state["fc_date"]
        i = origin_dates.index(cur) if cur in origin_dates else min(range(len(origin_dates)), key=lambda j: abs((origin_dates[j] - cur).days))
        st.session_state["fc_date"] = origin_dates[min(max(i + step, 0), len(origin_dates) - 1)]

    c1, c2, c3 = st.columns([1.6, 1, 1.2])
    model_sel = c1.selectbox("Model", model_cols, index=model_cols.index(best_model) if best_model in model_cols else 0, key="fc_model")
    picked = c2.date_input("Forecast origin", min_value=origin_dates[0], max_value=origin_dates[-1], key="fc_date")
    c3.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    b1, b2 = c3.columns(2)
    b1.button("◀ Prev", on_click=_shift, args=(-1,), key="fc_prev", **STRETCH)
    b2.button("Next ▶", on_click=_shift, args=(1,), key="fc_next", **STRETCH)
    overlay = st.multiselect("Compare with", [m for m in model_cols if m != model_sel],
                             default=[naive_col] if naive_col and naive_col != model_sel and naive_col in model_cols else [], key="fc_overlay")
    origin = picked if picked in origin_dates else min(origin_dates, key=lambda x: abs((x - picked).days))
    day = fc[fc["forecast_origin"].dt.date == origin].sort_values("target_datetime")
    spacer()

    if day.empty:
        st.info("Tidak ada prakiraan untuk origin ini.")
    else:
        err = day[model_sel] - day["actual_nat_demand"]
        worst = err.abs().idxmax()
        k = st.columns(4)
        kpi(k[0], "MAE", f"{err.abs().mean():,.1f}", sub="MWh · this origin")
        kpi(k[1], "RMSE", f"{np.sqrt((err ** 2).mean()):,.1f}", sub="MWh · this origin")
        kpi(k[2], "Bias", f"{err.mean():+,.1f}", sub="MWh · + means over-forecast")
        kpi(k[3], "Largest miss", f"{abs(err[worst]):,.1f}", sub=f"MWh · at {day.loc[worst, 'target_datetime']:%H:%M}")
        spacer()

        with st.container(border=True):
            card_title(f"{model_sel}", f"24 hours from {origin:%A, %d %B %Y}")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
            fig.add_trace(go.Scatter(x=day["target_datetime"], y=day["actual_nat_demand"], name="Actual", mode="lines",
                                     line=dict(color=INK, width=2.2)), row=1, col=1)
            for _k, _m in enumerate(overlay):
                fig.add_trace(go.Scatter(x=day["target_datetime"], y=day[_m], name=_m, mode="lines",
                                         line=dict(color=SERIES[(_k + 2) % len(SERIES)], width=1.5, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=day["target_datetime"], y=day[model_sel], name="Forecast", mode="lines+markers",
                                     line=dict(color=ACCENT, width=2.2), marker=dict(size=5)), row=1, col=1)
            fig.add_trace(go.Bar(x=day["target_datetime"], y=err, name="Error", showlegend=False,
                                 marker_color=["#D97706" if v > 0 else ACCENT for v in err],
                                 hovertemplate="%{y:+,.1f} MWh<extra>Error</extra>"), row=2, col=1)
            style(fig, height=500)
            fig.update_yaxes(title_text="Demand (MWh)", row=1, col=1)
            fig.update_yaxes(title_text="Error", row=2, col=1)
            plot(fig)

        with st.expander("Hourly detail"):
            detail = day[["horizon", "target_datetime", "actual_nat_demand", model_sel]].copy()
            detail["error"] = detail[model_sel] - detail["actual_nat_demand"]
            detail["abs_error"] = detail["error"].abs()
            detail = detail.rename(columns={"actual_nat_demand": "actual", model_sel: "forecast"})
            num_cols = ["actual", "forecast", "error", "abs_error"]
            detail[num_cols] = detail[num_cols].round(2)
            table(detail, column_config={
                "horizon": st.column_config.NumberColumn("Horizon (h)"),
                "target_datetime": st.column_config.DatetimeColumn("Target time", format="DD MMM, HH:mm"),
            })
            st.download_button("Download this forecast", detail.to_csv(index=False).encode(), "forecast_24h.csv", "text/csv", key="dl_fc")

# ================================================================== ERROR ANALYSIS
with tab_err:
    spacer()
    emodel = st.selectbox("Model", model_cols, index=model_cols.index(best_model) if best_model in model_cols else 0, key="err_model")
    ed = fc[["forecast_origin", "target_datetime", "actual_nat_demand", emodel]].dropna().copy()
    ed["err"] = ed[emodel] - ed["actual_nat_demand"]
    ed["abs_err"] = ed["err"].abs()
    ed["ape"] = ed["abs_err"] / ed["actual_nat_demand"].where(ed["actual_nat_demand"].abs() > 1e-6) * 100
    ed["hour"] = ed["target_datetime"].dt.hour
    ed["dow"] = ed["target_datetime"].dt.dayofweek
    ed["date"] = ed["target_datetime"].dt.normalize()
    k = st.columns(4)
    kpi(k[0], "Bias", f"{ed['err'].mean():+,.1f}", sub="MWh · + means over-forecast")
    kpi(k[1], "P90 abs. error", f"{ed['abs_err'].quantile(0.9):,.1f}", sub="MWh · 9 of 10 hours are below this")
    kpi(k[2], "Within ±5%", f"{(ed['ape'] <= 5).mean() * 100:,.1f}%", sub="of hourly forecasts")
    _w = ed["abs_err"].idxmax()
    kpi(k[3], "Largest miss", f"{ed.loc[_w, 'abs_err']:,.1f}", sub=f"MWh · {ed.loc[_w, 'target_datetime']:%d %b %Y %H:%M}")
    spacer()

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            card_title("Bias by hour and weekday", "Blue = over-forecast · red = under-forecast")
            pv = ed.pivot_table(index="dow", columns="hour", values="err", aggfunc="mean").sort_index()
            lim = float(np.nanmax(np.abs(pv.values))) or 1.0
            fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{h:02d}" for h in pv.columns], y=[DAYS[i] for i in pv.index],
                                       colorscale=DIVERGING, zmin=-lim, zmax=lim, colorbar=dict(title="MWh", thickness=10, len=0.9),
                                       hovertemplate="%{y} %{x}:00<br>bias %{z:+,.1f} MWh<extra></extra>"))
            style(fig, height=320, legend=False, hover="closest")
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(title_text="Target hour")
            plot(fig)
    with b:
        with st.container(border=True):
            card_title("Daily MAPE calendar", "Each cell is one forecast day · darker = harder day")
            dd = ed.groupby("date", as_index=False)["ape"].mean()
            dd["week"] = dd["date"].dt.to_period("W-SUN").dt.start_time
            dd["dow"] = dd["date"].dt.dayofweek
            pc = dd.pivot_table(index="dow", columns="week", values="ape", aggfunc="mean").reindex(range(7))
            fig = go.Figure(go.Heatmap(z=pc.values, x=[f"{w:%d %b}" for w in pc.columns], y=DAYS, colorscale=SEQUENTIAL,
                                       colorbar=dict(title="MAPE %", thickness=10, len=0.9), xgap=2, ygap=2,
                                       hovertemplate="Week of %{x} · %{y}<br>MAPE %{z:.2f}%<extra></extra>"))
            style(fig, height=320, legend=False, hover="closest")
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(title_text="Week starting")
            plot(fig)

    spacer()
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            card_title("Actual vs predicted", "Points on the dashed line are perfect forecasts")
            sm = ed.sample(min(len(ed), 3000), random_state=1)
            lo, hi = float(min(ed["actual_nat_demand"].min(), ed[emodel].min())), float(max(ed["actual_nat_demand"].max(), ed[emodel].max()))
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=sm["actual_nat_demand"], y=sm[emodel], mode="markers", name="Hours",
                                       marker=dict(size=4, color=ACCENT, opacity=0.35)))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal", line=dict(color=INK, width=1.2, dash="dash")))
            style(fig, height=340, hover="closest")
            fig.update_xaxes(title_text="Actual (MWh)")
            fig.update_yaxes(title_text="Predicted (MWh)")
            plot(fig)
    with b:
        with st.container(border=True):
            card_title("Error distribution", "Forecast minus actual, all test hours")
            fig = go.Figure(go.Histogram(x=ed["err"], nbinsx=60, marker=dict(color=ACCENT, line=dict(color="white", width=0.5))))
            fig.add_vline(x=0, line_color=INK, line_width=1)
            style(fig, height=340, legend=False, hover="closest")
            fig.update_xaxes(title_text="Error (MWh)")
            fig.update_yaxes(title_text="Hours")
            plot(fig)

    spacer()
    with st.container(border=True):
        card_title("Hardest forecast days", "Top 10 by daily MAE")
        wd = ed.groupby("date").agg(MAE=("abs_err", "mean"), RMSE=("err", lambda v: float(np.sqrt((v ** 2).mean()))),
                                    Bias=("err", "mean"), MAPE=("ape", "mean")).sort_values("MAE", ascending=False).head(10).reset_index()
        wd["date"] = wd["date"].dt.strftime("%a, %d %b %Y")
        table(wd.rename(columns={"date": "Day", "MAPE": "MAPE (%)"}).round(2))

# ================================================================== DATA EXPLORER
with tab_eda:
    spacer()
    if eda.empty or "nat_demand" not in eda:
        st.info("Dataset EDA belum tersedia di folder data/.")
    else:
        dmin, dmax = eda["datetime"].min().date(), eda["datetime"].max().date()
        f1, f2 = st.columns([1.2, 3])
        rng = f1.date_input("Period", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="eda_range")
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            e = eda[(eda["datetime"].dt.date >= rng[0]) & (eda["datetime"].dt.date <= rng[1])].copy()
        else:
            e = eda.copy()
        if e.empty:
            st.info("Tidak ada data pada periode ini.")
        else:
            dm = e["nat_demand"]
            k = st.columns(5)
            kpi(k[0], "Observations", f"{len(e):,}", sub="hourly records")
            kpi(k[1], "Mean demand", f"{dm.mean():,.1f}", sub="MWh")
            kpi(k[2], "Median", f"{dm.median():,.1f}", sub="MWh")
            kpi(k[3], "Std. deviation", f"{dm.std():,.1f}", sub="MWh")
            kpi(k[4], "Peak", f"{dm.max():,.1f}", sub=f"MWh · {e.loc[dm.idxmax(), 'datetime']:%d %b %Y %H:%M}")
            spacer()

            t_trend, t_pat, t_dist, t_wx, t_cor = st.tabs(["Trend", "Patterns", "Distribution", "Weather link", "Correlation"])

            with t_trend:
                spacer()
                a, b = st.columns([2.2, 1])
                with a:
                    with st.container(border=True):
                        h1, h2 = st.columns([1, 1.3])
                        with h1:
                            card_title("Demand over time")
                        agg = h2.radio("Aggregation", ["Hourly", "Daily", "Weekly", "Monthly"], index=1, horizontal=True,
                                       key="eda_agg", label_visibility="collapsed")
                        ser = e.set_index("datetime")["nat_demand"]
                        if agg == "Daily":
                            ser = ser.resample("D").mean()
                        elif agg == "Weekly":
                            ser = ser.resample("W").mean()
                        elif agg == "Monthly":
                            ser = ser.resample("MS").mean()
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ser.index, y=ser.values, name=agg, mode="lines", line=dict(color=ACCENT, width=1.2)))
                        if agg in ("Hourly", "Daily"):
                            sm = ser.rolling(24 if agg == "Hourly" else 30, min_periods=1).mean()
                            fig.add_trace(go.Scatter(x=sm.index, y=sm.values, name="Moving average", mode="lines",
                                                     line=dict(color=INK, width=2)))
                        style(fig, height=340)
                        fig.update_yaxes(title_text="Demand (MWh)")
                        plot(fig)
                with b:
                    with st.container(border=True):
                        card_title("Annual average")
                        yr = e.groupby("year", as_index=False)["nat_demand"].mean()
                        fig = go.Figure(go.Bar(x=yr["year"].astype(str), y=yr["nat_demand"], marker_color=ACCENT,
                                               hovertemplate="%{x}: %{y:,.1f} MWh<extra></extra>"))
                        style(fig, height=340, legend=False, hover="closest")
                        fig.update_yaxes(title_text="MWh")
                        plot(fig)

            with t_pat:
                spacer()
                with st.container(border=True):
                    card_title("Average demand by hour and weekday", "Darker means higher demand")
                    pv = e.pivot_table(index="dow", columns="hour", values="nat_demand", aggfunc="mean").sort_index()
                    fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{h:02d}" for h in pv.columns], y=[DAYS[i] for i in pv.index],
                                               colorscale=SEQUENTIAL, colorbar=dict(title="MWh", thickness=10, len=0.9),
                                               hovertemplate="%{y} %{x}:00<br>%{z:,.1f} MWh<extra></extra>"))
                    style(fig, height=300, legend=False, hover="closest")
                    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
                    fig.update_xaxes(title_text="Hour of day")
                    plot(fig)
                spacer()
                a, b = st.columns(2)
                with a:
                    with st.container(border=True):
                        card_title("Seasonality by month")
                        mo = e.groupby("month", as_index=False)["nat_demand"].mean()
                        fig = go.Figure(go.Scatter(x=mo["month"], y=mo["nat_demand"], mode="lines+markers",
                                                   line=dict(color=ACCENT, width=2.4), marker=dict(size=7),
                                                   hovertemplate="Month %{x}: %{y:,.1f} MWh<extra></extra>"))
                        style(fig, height=300, legend=False, hover="closest")
                        fig.update_xaxes(title_text="Month", dtick=1)
                        fig.update_yaxes(title_text="Average demand (MWh)")
                        plot(fig)
                with b:
                    with st.container(border=True):
                        card_title("Weekday, weekend and holiday")
                        dt = e.groupby("day_type", as_index=False)["nat_demand"].mean()
                        fig = go.Figure(go.Bar(x=dt["day_type"], y=dt["nat_demand"],
                                               marker_color=[DAYTYPE_COLORS.get(x, ACCENT) for x in dt["day_type"]],
                                               text=[f"{v:,.1f}" for v in dt["nat_demand"]], textposition="outside", cliponaxis=False,
                                               hovertemplate="%{x}: %{y:,.1f} MWh<extra></extra>"))
                        style(fig, height=300, legend=False, hover="closest")
                        fig.update_yaxes(title_text="Average demand (MWh)")
                        plot(fig)

            with t_dist:
                spacer()
                a, b = st.columns(2)
                with a:
                    with st.container(border=True):
                        card_title("Demand distribution")
                        fig = go.Figure(go.Histogram(x=e["nat_demand"], nbinsx=50, marker=dict(color=ACCENT, line=dict(color="white", width=0.6))))
                        style(fig, height=320, legend=False, hover="closest")
                        fig.update_xaxes(title_text="Demand (MWh)")
                        fig.update_yaxes(title_text="Hours")
                        plot(fig)
                with b:
                    with st.container(border=True):
                        card_title("Spread by day type")
                        fig = px.box(e, x="day_type", y="nat_demand", color="day_type", color_discrete_map=DAYTYPE_COLORS,
                                     category_orders={"day_type": ["Weekday", "Weekend", "Holiday"]})
                        style(fig, height=320, legend=False, hover="closest")
                        fig.update_xaxes(title_text="")
                        fig.update_yaxes(title_text="Demand (MWh)")
                        plot(fig)
                with st.expander("Descriptive statistics"):
                    stats = e.drop(columns=["year", "month", "hour", "dow"], errors="ignore").select_dtypes(include="number")
                    st.dataframe(stats.describe().T.round(3), **STRETCH)

            with t_wx:
                spacer()
                wcols = [c for c in e.columns if c not in ("datetime", "nat_demand", "holiday", "school", "year", "month", "hour", "dow")
                         and pd.api.types.is_numeric_dtype(e[c])]
                if not wcols:
                    st.info("Kolom cuaca numerik tidak ditemukan.")
                else:
                    with st.container(border=True):
                        h1, h2 = st.columns([2.5, 1])
                        with h1:
                            card_title("Weather variable vs demand", "Dots are a sample of hours; the line is the mean demand per value bin")
                        var = h2.selectbox("Variable", wcols, index=wcols.index("T2M_toc") if "T2M_toc" in wcols else 0,
                                           key="eda_var", label_visibility="collapsed")
                        full = e[[var, "nat_demand"]].dropna()
                        samp = full.sample(min(len(full), 4000), random_state=42) if len(full) else full
                        bins = pd.cut(full[var], bins=20)
                        bm = full.groupby(bins, observed=True).agg(x=(var, "mean"), y=("nat_demand", "mean")).reset_index(drop=True)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=samp[var], y=samp["nat_demand"], mode="markers", name="Hours",
                                                 marker=dict(size=5, color="#94A3B8", opacity=0.3)))
                        fig.add_trace(go.Scatter(x=bm["x"], y=bm["y"], mode="lines+markers", name="Mean per bin",
                                                 line=dict(color=ACCENT, width=3), marker=dict(size=6)))
                        style(fig, height=400, hover="closest")
                        fig.update_xaxes(title_text=var)
                        fig.update_yaxes(title_text="Demand (MWh)")
                        plot(fig)

            with t_cor:
                spacer()
                ccols = [c for c in e.select_dtypes(include="number").columns if c not in ("year", "month", "hour", "dow")]
                method = st.radio("Method", ["Pearson", "Spearman"], horizontal=True, key="eda_corr", label_visibility="collapsed")
                corr = e[ccols].corr(method=method.lower())
                a, b = st.columns([1.5, 1])
                with a:
                    with st.container(border=True):
                        card_title(f"{method} correlation matrix")
                        fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale=DIVERGING,
                                                   zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
                                                   textfont=dict(size=9), colorbar=dict(thickness=10, len=0.9),
                                                   hovertemplate="%{y} × %{x}<br>%{z:.3f}<extra></extra>"))
                        style(fig, height=max(440, 26 * len(ccols)), legend=False, hover="closest")
                        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
                        plot(fig)
                with b:
                    with st.container(border=True):
                        card_title("Correlation with demand")
                        cr = corr["nat_demand"].drop("nat_demand").sort_values(key=lambda x: x.abs())
                        fig = go.Figure(go.Bar(x=cr.values, y=cr.index, orientation="h",
                                               marker_color=[ACCENT if v >= 0 else "#B91C1C" for v in cr.values],
                                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
                        style(fig, height=max(440, 26 * len(ccols)), legend=False, hover="closest")
                        fig.update_xaxes(range=[-1, 1], zeroline=True, zerolinecolor=LINE)
                        fig.update_yaxes(automargin=True)
                        plot(fig)
                with st.expander("Data quality and preview"):
                    miss = eda.isna().sum().reset_index()
                    miss.columns = ["Variable", "Missing"]
                    miss = miss[miss["Missing"] > 0]
                    if miss.empty:
                        st.success("Tidak ada missing value pada dataset ini.")
                    else:
                        table(miss)
                    table(e.head(100))
                    st.download_button("Download filtered data", e.to_csv(index=False).encode(), "eda_filtered.csv", "text/csv", key="dl_eda")

# ================================================================== WEATHER STAGE 1
with tab_wx:
    spacer()
    need = {"Stage1", "Split", "Variable", "MAE", "RMSE"}
    if wx.empty or not need.issubset(wx.columns):
        st.info("Metrik cuaca Stage 1 belum diekspor.")
    else:
        for c in ("MAE", "RMSE"):
            wx[c] = pd.to_numeric(wx[c], errors="coerce")
        with st.container(border=True):
            h1, h2 = st.columns([2.5, 1])
            with h1:
                card_title("Weather forecast error by variable", "Each panel has its own scale because the variables use different units")
            wm = h2.radio("Weather metric", ["RMSE", "MAE"], horizontal=True, key="wx_metric", label_visibility="collapsed")
            fig = px.bar(wx, x="Stage1", y=wm, color="Split", barmode="group", facet_col="Variable", facet_col_wrap=4,
                         facet_row_spacing=0.12, color_discrete_map={"val": "#94A3B8", "test": ACCENT})
            fig.for_each_annotation(lambda an: an.update(text=an.text.split("=")[-1]))
            style(fig, height=620, hover="closest")
            fig.update_yaxes(matches=None, showticklabels=True, title_text="")
            fig.update_xaxes(title_text="")
            plot(fig)
        if not wf.empty:
            spacer()
            wvars = sorted({c.split("__")[0] for c in wf.columns if c.endswith("__actual")})
            wdates = sorted(wf["forecast_origin"].dt.date.unique())
            if wvars and wdates:
                with st.container(border=True):
                    h1, h2, h3 = st.columns([2, 1, 1])
                    with h1:
                        card_title("Actual vs forecast weather", "Stage 1 LSTM · 24 hours from one forecast origin")
                    wv = h2.selectbox("Weather variable", wvars, index=wvars.index("T2M_toc") if "T2M_toc" in wvars else 0,
                                      key="wxf_var", label_visibility="collapsed")
                    wo = h3.selectbox("Forecast origin", wdates, index=len(wdates) - 1, key="wxf_date", label_visibility="collapsed")
                    g = wf[wf["forecast_origin"].dt.date == wo].sort_values("target_datetime")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=g["target_datetime"], y=g[f"{wv}__actual"], name="Actual", mode="lines", line=dict(color=INK, width=2.2)))
                    fig.add_trace(go.Scatter(x=g["target_datetime"], y=g[f"{wv}__forecast"], name="Forecast", mode="lines+markers",
                                             line=dict(color=ACCENT, width=2.2), marker=dict(size=5)))
                    style(fig, height=340)
                    fig.update_yaxes(title_text=wv)
                    plot(fig)
        with st.expander("Stage 1 metrics table"):
            table(wx.round(5))
            st.download_button("Download stage1_weather_metrics.csv", wx.to_csv(index=False).encode(),
                               "stage1_weather_metrics.csv", "text/csv", key="dl_wx")

# ================================================================== ABOUT
with tab_about:
    spacer()
    a, b = st.columns([1.5, 1])
    with a:
        with st.container(border=True):
            card_title("Forecast design")
            st.markdown(f"""
<ul class="about">
<li><b>Horizon:</b> 24 hours ahead, one forecast origin per day at 00:00</li>
<li><b>Context:</b> 168 previous hours (7 days) of demand</li>
<li><b>Stage 1:</b> LSTM weather forecast for the next 24 hours</li>
<li><b>Stage 2:</b> demand model using demand history, calendar and the Stage 1 weather forecast</li>
<li><b>Split:</b> time-ordered, no shuffling · test {period}</li>
<li><b>Metrics:</b> RMSE, MAE, MAPE, R² overall and per horizon</li>
</ul>""", unsafe_allow_html=True)
        spacer()
        with st.container(border=True):
            card_title("Models by family")
            fam = cmp.groupby("Keluarga")["Model"].apply(lambda s: ", ".join(s)).reset_index()
            fam.columns = ["Family", "Models"]
            table(fam)
    with b:
        with st.container(border=True):
            card_title("Downloads", "CSV exported from the notebook")
            for fname, label in [("model_comparison.csv", "Model comparison"), ("metrics_by_horizon.csv", "Metrics by horizon"),
                                 ("stage1_weather_forecast_test.csv", "Stage 1 weather forecasts"),
                                 ("forecast_outputs_test.csv", "Test forecasts"), ("eda_dataset.csv", "Dataset"),
                                 ("stage1_weather_metrics.csv", "Stage 1 weather metrics")]:
                p = DATA / fname
                if p.exists():
                    st.download_button(label, p.read_bytes(), fname, "text/csv", key=f"dl_about_{fname}")
        spacer()
        if not eda.empty:
            with st.container(border=True):
                card_title("Dataset")
                st.markdown(f"**{len(eda):,}** hourly rows  \n{eda['datetime'].min():%d %b %Y} – {eda['datetime'].max():%d %b %Y}")

st.caption("Generated from completed notebook outputs · models are not retrained by this app")
