"""Streamlit dashboard for the business simulator.

Run locally:   streamlit run app.py

How Streamlit works, in one paragraph: the whole script reruns from top to
bottom every time the user changes a widget. Widgets return their current
value, so the script reads like a normal program. Slow work is wrapped in
@st.cache_data, which skips the work when the inputs haven't changed.

Flow of this page:
    template YAML  ->  Business  ->  edits from the "Your numbers" table
    ->  edited Business  ->  simulate / tornado / compare  ->  cards and charts
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from pydantic import ValidationError

from simulator import (
    Business,
    Comparison,
    SimulationResult,
    Summary,
    Tornado,
    compare,
    get_plan,
    period_table,
    simulate,
    summarize,
    tornado,
)
from simulator.editing import inputs_frame, read_frame
from simulator.formatting import input_label, money
from simulator.metrics import cash_runout_period

TEMPLATES = Path(__file__).parent / "templates"
ACCENT, PLAN_B, BAD, MUTED = "#b06a12", "#2b6a8c", "#c0432f", "#8a99a8"
FONT = "IBM Plex Sans, Segoe UI, sans-serif"

FAVICON = Path(__file__).parent / "assets" / "favicon.png"

st.set_page_config(page_title="Business Simulator", page_icon=str(FAVICON), layout="wide")

# Fonts and a few card styles. Colours use translucent greys so the page
# works in both Streamlit's light and dark themes.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; letter-spacing: -0.01em; }
.block-container { padding-top: 2.2rem; max-width: 1360px; }
.lead { font-size: 1.05rem; line-height: 1.6; max-width: 72ch; margin: 0.2rem 0 1rem; }
.lead b { font-weight: 600; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 0.6rem; }
@media (max-width: 1500px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { border: 1px solid rgba(128,128,128,.28); border-radius: 10px; padding: 12px 14px; background: rgba(128,128,128,.05);
       display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.kpi .k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .07em; opacity: .7; font-weight: 500; }
.kpi .v { font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem; font-weight: 500; letter-spacing: -0.02em; overflow-wrap: anywhere; }
.kpi .d { font-size: 0.8rem; opacity: .75; }
.pill { align-self: flex-start; font-size: 0.72rem; font-weight: 600; border-radius: 999px; padding: 1px 9px; margin-top: 4px; }
.pill.good { color: #2b8a57; background: rgba(43,138,87,.14); }
.pill.warn { color: #b06a12; background: rgba(176,106,18,.16); }
.pill.bad  { color: #c0432f; background: rgba(192,67,47,.14); }
.verdict { border-radius: 10px; padding: 12px 14px; background: rgba(43,106,140,.12); margin: 0.4rem 0 0.8rem; line-height: 1.55; }
.verdict b { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ---- loading and validating ---------------------------------------------

def list_templates() -> dict[str, str]:
    """Template display name -> YAML text, for every file in templates/."""
    found: dict[str, str] = {}
    for path in sorted(TEMPLATES.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        found[str(data.get("name", path.stem))] = text
    return found


def describe_errors(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors():
        where = " → ".join(str(part) for part in err["loc"]) or "file"
        out.append(f"**{where}**: {err['msg'].removeprefix('Value error, ')}")
    return out


def parse(text: str) -> tuple[Business | None, list[str]]:
    """Validate YAML text. Returns the business, or a list of readable errors."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return None, [f"The YAML can't be read: {e}"]
    try:
        return Business.model_validate(data), []
    except ValidationError as e:
        return None, describe_errors(e)


# ---- cached runs ------------------------------------------------------------
# Cached on JSON text rather than the Business object: text is easy for
# Streamlit to hash, and identical text always means an identical business.

@st.cache_data(show_spinner="Running scenarios…")
def run_base(text: str, n: int, seed: int) -> tuple[SimulationResult, Summary, Tornado]:
    business = Business.model_validate(yaml.safe_load(text))
    result = simulate(business, n=n, seed=seed)
    return result, summarize(result), tornado(business)


@st.cache_data(show_spinner="Comparing plans…")
def run_compare(text: str, n: int, seed: int, plan_name: str) -> Comparison:
    business = Business.model_validate(yaml.safe_load(text))
    return compare(business, get_plan(business, plan_name), n=n, seed=seed)


# ---- charts --------------------------------------------------------------

def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(t=36, b=10, l=10, r=10), font=dict(family=FONT),
        legend=dict(orientation="h", y=1.12, x=0), hoverlabel=dict(font_family=FONT),
    )
    return fig


def profit_histogram(a: SimulationResult, b: SimulationResult | None = None, s: Summary | None = None) -> go.Figure:
    fig = go.Figure()
    # One shared set of bins, so the two plans' bars line up and compare fairly.
    both = a.total_profit if b is None else np.concatenate([a.total_profit, b.total_profit])
    edges = np.histogram_bin_edges(both, bins=60)
    width = float(edges[1] - edges[0])
    hover = "Profit about %{x:,.0f}<br>%{y} scenarios<extra>%{fullData.name}</extra>"
    fig.add_histogram(
        x=a.total_profit, xbins=dict(start=float(edges[0]), end=float(edges[-1]), size=width),
        name="Current plan", marker_color=ACCENT, opacity=0.8, hovertemplate=hover,
    )
    if b is not None:
        # Plan B as an outline on top, so neither plan hides the other.
        counts, _ = np.histogram(b.total_profit, bins=edges)
        centres = (edges[:-1] + edges[1:]) / 2
        fig.add_scatter(
            mode="lines", x=centres, y=counts, line_shape="hvh", line=dict(color=PLAN_B, width=2.2),
            name=b.plan or "Plan B", hovertemplate=hover,
        )
    fig.add_vline(x=0, line_dash="dash", line_color=BAD, annotation_text="Zero profit", annotation_position="top left")
    if s is not None:
        for label, value in (("P10", s.total_profit.p10), ("Median", s.total_profit.p50), ("P90", s.total_profit.p90)):
            fig.add_vline(x=value, line_dash="dot", line_color=MUTED, annotation_text=label, annotation_position="top")
    fig.update_layout(barmode="overlay", xaxis_title=f"Total profit over the horizon ({a.currency})", yaxis_title="Scenarios")
    fig.update_xaxes(tickformat="~s")
    return style(fig, 380)


def cash_fan(a: SimulationResult, b: SimulationResult | None = None) -> go.Figure:
    t = period_table(a)
    start = float(a.starting_cash.mean())
    x = [0, *t["period"]]
    hover = "%{y:,.0f}<extra>%{fullData.name}</extra>"
    fig = go.Figure()
    fig.add_scatter(mode="lines", x=x, y=[start, *t["cash_p90"]], line=dict(width=0), name="P90 (good case)", hovertemplate=hover, showlegend=False)
    fig.add_scatter(mode="lines", 
        x=x, y=[start, *t["cash_p10"]], fill="tonexty", fillcolor="rgba(176,106,18,0.18)",
        line=dict(width=0), name="P10 to P90", hovertemplate="%{y:,.0f}<extra>P10 (bad case)</extra>",
    )
    fig.add_scatter(mode="lines", x=x, y=[start, *t["cash_p50"]], line=dict(color=ACCENT, width=3), name="Median", hovertemplate=hover)
    if b is not None:
        tb = period_table(b)
        fig.add_scatter(mode="lines", 
            x=x, y=[float(b.starting_cash.mean()), *tb["cash_p50"]],
            line=dict(color=PLAN_B, width=2, dash="dash"), name=f"{b.plan} median", hovertemplate=hover,
        )
    fig.add_hline(y=0, line_dash="dash", line_color=BAD)
    fig.update_layout(hovermode="x unified", xaxis_title=f"{a.period.title()} (0 = start)", yaxis_title=f"Cash ({a.currency})")
    fig.update_yaxes(tickformat="~s")
    return style(fig, 380)


def tornado_chart(t: Tornado, top: int = 10) -> go.Figure:
    rows = t.table.head(top).iloc[::-1]  # reversed: Plotly draws the first row at the bottom
    labels = [input_label(p) for p in rows["input"]]
    fig = go.Figure()
    for col, name, colour, edge in (("profit_at_low", "Input at low", MUTED, "low"), ("profit_at_high", "Input at high", ACCENT, "high")):
        fig.add_bar(
            y=labels, x=rows[col] - t.base_profit, base=t.base_profit, orientation="h", name=name, marker_color=colour,
            customdata=np.stack([rows[edge], rows[col]], axis=1),
            hovertemplate="%{y}<br>at %{customdata[0]:,.4g}: profit %{customdata[1]:,.0f}<extra></extra>",
        )
    fig.add_vline(x=t.base_profit, line_color=MUTED, annotation_text="Base case", annotation_position="top")
    fig.update_layout(barmode="overlay", xaxis_title="Total profit")
    fig.update_xaxes(tickformat="~s")
    return style(fig, max(280, 44 * len(rows) + 90))


# ---- small HTML pieces -------------------------------------------------------

def kpi(label: str, value: str, detail: str, pill: tuple[str, str] | None = None) -> str:
    tag = f'<span class="pill {pill[0]}">{pill[1]}</span>' if pill else ""
    return f'<div class="kpi"><span class="k">{label}</span><span class="v">{value}</span><span class="d">{detail}</span>{tag}</div>'


def risk_pill(p: float) -> tuple[str, str]:
    if p < 0.10:
        return ("good", "Low risk")
    return ("warn", "Watch this") if p < 0.30 else ("bad", "High risk")


# ---- sidebar ---------------------------------------------------------------

templates = list_templates()
with st.sidebar:
    st.header("Business")
    choice = st.selectbox("Template", [*templates, "Upload my own"])
    if choice == "Upload my own":
        upload = st.file_uploader("Business file (YAML or JSON)", type=["yaml", "yml", "json"])
        source = upload.getvalue().decode("utf-8") if upload else next(iter(templates.values()))
        if not upload:
            st.caption("Showing the first template until you upload a file.")
    else:
        source = templates[choice]
    st.header("Simulation")
    n = st.select_slider("Scenarios", options=[1000, 5000, 20000, 50000], value=5000,
                         help="More scenarios give steadier numbers but take longer.")
    seed = int(st.number_input("Random seed", min_value=0, value=42, step=1,
                               help="The same seed always gives the same results. Change it to check the numbers are stable."))

# ---- page -------------------------------------------------------------------

st.title("Business Simulator")
st.caption("Describe a business with honest ranges instead of single guesses. The simulator plays out thousands of possible futures and shows the odds.")

left, right = st.columns([1.35, 1.6], gap="large")

with left:
    st.subheader("Your numbers")
    with st.expander("Advanced: edit the full definition (YAML)"):
        st.caption("Streams, seasonality, capacity and plans live here. The table below updates from it.")
        text = st.text_area("YAML", value=source, height=420, key=f"yaml::{choice}", label_visibility="collapsed")

    base_business, errors = parse(text)
    if base_business is None:
        st.error("This business file has problems:\n\n" + "\n\n".join(f"- {e}" for e in errors))
        st.stop()

    st.caption("Set a range for each number. Make all three equal for a fixed value, or leave **Likely** empty for "
               "\"anywhere in the range\". Rates such as no-shows are fractions: 0.04 means 4%.")
    # The key includes a hash of the YAML, so editing the YAML resets the table.
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    frame = inputs_frame(base_business)
    edited = st.data_editor(
        frame,
        key=f"inputs::{choice}::{digest}",
        hide_index=True,
        disabled=["Input"],
        width="stretch",
        height=35 * (len(frame) + 1) + 3,
        column_config={
            "Input": st.column_config.TextColumn("Input", width=170),
            "Low": st.column_config.NumberColumn("Low", format="localized", width=80),
            "Likely": st.column_config.NumberColumn("Likely", format="localized", width=80),
            "High": st.column_config.NumberColumn("High", format="localized", width=80),
        },
    )
    updates, table_errors = read_frame(edited)
    business = None
    if not table_errors:
        try:
            business = base_business.with_inputs(updates)
        except ValidationError as e:
            table_errors = describe_errors(e)
    if table_errors:
        st.error("Fix these numbers to see results:\n\n" + "\n\n".join(f"- {e}" for e in table_errors))
        st.stop()
    assert business is not None
    st.caption(f"{business.horizon} {business.period}s · figures in {business.currency} · "
               "example template figures, not real data")

run_text = business.model_dump_json()  # JSON is valid YAML, so the cached runners can read it
result, summary, sens = run_base(run_text, n, seed)
cur, per = summary.currency, summary.period
tp = summary.total_profit

with right:
    st.subheader(business.name)
    runout = (f" Cash runs out in <b>{summary.prob_cash_runs_out:.0%}</b> of scenarios, usually by {per} {summary.cash_runs_out_period:.0f}."
              if summary.cash_runs_out_period else " Cash never runs out in any scenario.")
    st.markdown(
        f'<p class="lead">In a typical {per if summary.horizon == 1 else "run of " + str(summary.horizon) + " " + per + "s"}, '
        f"this business makes <b>{cur} {money(tp.p50)}</b>. One time in ten it does worse than <b>{money(tp.p10)}</b>, "
        f"and <b>{summary.prob_loss:.0%}</b> of scenarios lose money.{runout}</p>",
        unsafe_allow_html=True,
    )
    units = " · ".join(f"{k}: ~{v:,.0f} per {per}" for k, v in summary.breakeven_units.items())
    st.markdown(
        '<div class="kpis">'
        + kpi("Median profit", f"{money(tp.p50)}", f"P10 {money(tp.p10)} · P90 {money(tp.p90)}")
        + kpi("Chance of a loss", f"{summary.prob_loss:.1%}", "scenarios with negative total profit", risk_pill(summary.prob_loss))
        + kpi("Break-even", f"{summary.breakeven_ratio.p50:.0%}", f"of sales volume · {units}")
        + kpi("Cash runs out", f"{summary.prob_cash_runs_out:.1%}",
              f"usually by {per} {summary.cash_runs_out_period:.0f}" if summary.cash_runs_out_period
              else f"median ending cash {money(summary.ending_cash.p50)}",
              risk_pill(summary.prob_cash_runs_out))
        + "</div>",
        unsafe_allow_html=True,
    )

    profit_tab, cash_tab, drivers_tab, compare_tab = st.tabs(
        [":material/bar_chart: Profit", ":material/account_balance: Cash", ":material/tune: What drives profit", ":material/compare_arrows: Compare plans"]
    )

    with profit_tab:
        st.plotly_chart(profit_histogram(result, s=summary), width="stretch")
        st.caption("Each bar counts how many simulated futures ended with that total profit. "
                   "Bars left of the red line are futures that lose money.")

    with cash_tab:
        st.plotly_chart(cash_fan(result), width="stretch")
        st.caption("The shaded band holds the middle 80% of scenarios. If it dips below the red line, "
                   "the business can run out of cash even when it is profitable over the year.")

    with drivers_tab:
        st.plotly_chart(tornado_chart(sens), width="stretch")
        top = sens.table.iloc[0] if not sens.table.empty else None
        if top is not None:
            st.caption(
                f"**{input_label(top['input'])}** moves profit the most, by {money(top['swing'])}. "
                "Pin that number down first. Each bar moves one input from low to high while the others stay at "
                f"their likely value (base case {money(sens.base_profit)})."
            )

    with compare_tab:
        if not business.plans:
            st.info("This template has no plans. Add a `plans:` section in the YAML editor to compare alternatives.")
        else:
            plan_name = st.selectbox("Compare the current plan with", [p.name for p in business.plans])
            plan = get_plan(business, plan_name)
            cmp = run_compare(run_text, n, seed, plan_name)
            d = cmp.profit_difference
            w = cmp.prob_b_better
            judgement = ("a consistent improvement" if w >= 0.7 else
                         "usually worse than the current plan" if w <= 0.3 else
                         "a coin toss: it depends on which future you get")
            st.markdown(
                f'<div class="verdict">{plan.description + "<br>" if plan.description else ""}'
                f"<b>{plan_name}</b> makes more profit in <b>{w:.0%}</b> of scenarios: {judgement}. "
                f"The typical change is <b>{'+' if d.p50 > 0 else ''}{money(d.p50)}</b> "
                f"(P10 {money(d.p10)}, P90 {money(d.p90)}).</div>",
                unsafe_allow_html=True,
            )
            table = cmp.table()
            is_rate = table["measure"].str.startswith("Chance")
            fmt = lambda v, rate: f"{v:.1%}" if rate else money(v)  # noqa: E731
            shown = pd.DataFrame({
                "Measure": table["measure"],
                "Current plan": [fmt(v, r) for v, r in zip(table["plan_a"], is_rate)],
                plan_name: [fmt(v, r) for v, r in zip(table["plan_b"], is_rate)],
                "Change": [f"{v * 100:+.1f} pts" if r else ("+" if v > 0 else "") + money(v)
                           for v, r in zip(table["change"], is_rate)],
            })
            st.dataframe(shown, hide_index=True, width="stretch")
            st.plotly_chart(profit_histogram(cmp.result_a, cmp.result_b), width="stretch")
            st.plotly_chart(cash_fan(cmp.result_a, cmp.result_b), width="stretch")
            st.caption("Both plans face exactly the same random futures, so differences come from the plan, not from luck.")

# ---- download -------------------------------------------------------------------

with st.sidebar:
    st.header("Results")
    per_scenario = pd.DataFrame({
        "scenario": np.arange(1, result.n + 1),
        "total_profit": result.total_profit.round(2),
        "ending_cash": result.cash[:, -1].round(2),
        "cash_runs_out_period": cash_runout_period(result),
    })
    st.download_button(
        "Download scenarios (CSV)", per_scenario.to_csv(index=False).encode("utf-8"),
        file_name=f"{business.name.lower().replace(' ', '_')}_scenarios.csv", mime="text/csv", width="stretch",
    )
    st.caption("One row per simulated future, for your own analysis in Excel or pandas.")
