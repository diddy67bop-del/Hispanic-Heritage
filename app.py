"""
Hispanic Heritage Historical Data Explorer
Self-contained Streamlit app for World Bank historical data and function analysis.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor,
)
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Data Lab", page_icon="🌎", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Expected years a newborn would live if current age-specific mortality patterns persisted."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people. This is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants over the period; it is not a count of people emigrating."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in consumer prices."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods/services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP."),
}
CONTEXTS = {
    "Business": ("Revenue", "US dollars", "Time", "years", "A business indicator can describe growth or decline; a fitted curve is descriptive, not a business guarantee."),
    "Engineering": ("Position", "meters", "Time", "seconds", "Position changes over time; its first derivative is velocity and second derivative is acceleration."),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "A concentration model is illustrative and must not be used for medical decisions."),
    "Sports science": ("Distance", "meters", "Time", "seconds", "A motion model can describe changing distance; derivatives represent rates of change."),
    "Biology": ("Population", "organisms", "Time", "days", "A population model is a simplified representation of changing population."),
    "Bioengineering": ("Output", "units", "Time", "hours", "A system-output model is simplified and depends on assumptions."),
    "Economic forecasting": ("Indicator", "index units", "Time", "years", "Economic projections are uncertain and do not establish historical causes."),
    "Weather": ("Temperature", "°C", "Time", "hours", "A temperature curve is a simplified model and is not a weather forecast."),
    "Custom": ("f(x)", "units", "x", "units", "Customize the meaning and units of the variables."),
}
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
x = sp.symbols("x", real=True)

def wb_url(country, indicator, start, end):
    return f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(country: str, indicator: str, start: int, end: int):
    url = wb_url(country, indicator, start, end)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError("The World Bank API returned no observations for this selection.")
    rows = []
    for item in payload[1]:
        val = item.get("value")
        yr = item.get("date")
        if val is not None and yr is not None:
            try:
                rows.append({"Year": int(yr), "Value": float(val)})
            except (ValueError, TypeError):
                continue
    df = pd.DataFrame(rows, columns=["Year", "Value"]).dropna()
    if df.empty:
        raise ValueError("The API response contained no non-missing values in the requested period.")
    return df.sort_values("Year").reset_index(drop=True), url

def validate_data(df, indicator_label):
    errors = []
    if not {"Year", "Value"}.issubset(df.columns):
        return ["The editable table must have Year and Value columns."]
    if df.empty:
        errors.append("There are no rows to fit.")
        return errors
    years = pd.to_numeric(df["Year"], errors="coerce")
    vals = pd.to_numeric(df["Value"], errors="coerce")
    if years.isna().any() or not np.all(np.equal(years.dropna(), np.floor(years.dropna()))):
        errors.append("Year values must be integers.")
    if years.duplicated().any():
        errors.append("Duplicate years are not allowed.")
    if vals.isna().any() or not np.isfinite(vals.dropna()).all():
        errors.append("Values must be finite numbers.")
    if indicator_label == "Life expectancy at birth" and ((vals < 0) | (vals > 125)).any():
        errors.append("Life expectancy values should be between 0 and 125 years.")
    if indicator_label == "Unemployment, national estimate" and ((vals < 0) | (vals > 100)).any():
        errors.append("Unemployment percentages should be between 0 and 100.")
    if indicator_label == "Crude birth rate" and (vals < 0).any():
        errors.append("Crude birth rates cannot be negative.")
    return errors

def safe_parse(expr_text):
    allowed = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
        "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
        "pi": sp.pi, "E": sp.E, "x": x,
    }
    if len(expr_text) > 180 or not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,^ \t]+", expr_text):
        raise ValueError("Use standard notation with x, numbers, +, -, *, /, ^, parentheses, sin, cos, tan, exp, log, ln, sqrt, or pi.")
    expr = parse_expr(expr_text, local_dict=allowed, transformations=TRANSFORMS, evaluate=True)
    if not expr.free_symbols.issubset({x}):
        raise ValueError("Only the variable x is allowed.")
    return sp.simplify(expr)

def numeric_fn(expr):
    return sp.lambdify(x, expr, modules=["numpy"])

def fmt_num(v):
    try:
        if not np.isfinite(float(v)): return "undefined"
        return f"{float(v):,.6g}"
    except Exception:
        return str(v)

def fig_to_png(fig):
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    out.seek(0)
    return out

def make_pdf(title, summary, chart_png, data, results):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in summary.split("\n"):
        story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
        story.append(Spacer(1, 4))
    if chart_png:
        story.extend([Spacer(1, 8), Image(chart_png, width=6.8*inch, height=3.6*inch)])
    story.extend([Spacer(1, 8), Paragraph("Data preview", styles["Heading2"])])
    preview = data.head(18).copy()
    tab = [list(preview.columns)] + [[str(v) for v in row] for row in preview.itertuples(index=False, name=None)]
    table = Table(tab, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(table)
    if not results.empty:
        story.extend([Spacer(1, 8), Paragraph("Calculated years", styles["Heading2"])])
        r = results.head(30)
        rt = [list(r.columns)] + [[str(v) for v in row] for row in r.itertuples(index=False, name=None)]
        table2 = Table(rt, repeatRows=1)
        table2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(table2)
    doc.build(story)
    out.seek(0)
    return out.getvalue()

st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB • historical data, fitted functions, and calculus exploration")
st.info("Brazil is included as a Latin American comparison. Country and indicator data come from the World Bank API; models describe patterns and do not prove causes.")

with st.sidebar:
    st.header("Historical data")
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    indicator_label = st.selectbox("Indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Sample marker increment (years)", 1, 10, 3)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_labels = st.multiselect("Compare with countries", [c for c in COUNTRIES if c != country_label], default=[])
    printer_mode = st.checkbox("Printer-friendly view", False)

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
code, indicator_code = COUNTRIES[country_label], INDICATORS[indicator_label][0]
units, definition = INDICATORS[indicator_label][1], INDICATORS[indicator_label][2]
st.write(f"**Indicator definition:** {definition}  \n**Units:** {units}  \n**Requested years:** {start_year}–{end_year} inclusive.")

try:
    source_df, source_url = fetch_indicator(code, indicator_code, start_year, end_year)
    st.caption(f"Source: World Bank API • Retrieved {datetime.now().strftime('%Y-%m-%d %H:%M')} • {len(source_df)} non-missing observations from {source_df.Year.min()} to {source_df.Year.max()}. [Open API endpoint]({source_url})")
except Exception as exc:
    st.error(f"Could not retrieve source data: {exc}")
    st.markdown("Check your internet connection or the World Bank API, then reload. No synthetic replacement values are used.")
    st.stop()

if "source_key" not in st.session_state or st.session_state.source_key != (code, indicator_code):
    st.session_state.source_key = (code, indicator_code)
    st.session_state.source_data = source_df.copy()
    st.session_state.edit_data = source_df.copy()
    st.session_state.calculated_year = None
    st.session_state.batch_results = pd.DataFrame()
if st.button("Reset edited data to source"):
    st.session_state.edit_data = st.session_state.source_data.copy()
    st.session_state.calculated_year = None
    st.session_state.batch_results = pd.DataFrame()
    st.rerun()

st.subheader("1. Source data and editable working table")
st.write("Edit the year/value table below. Edits are treated as a user-modified scenario and do not change the original API response.")
edited = st.data_editor(st.session_state.edit_data, num_rows="dynamic", use_container_width=True, key="data_editor")
st.session_state.edit_data = edited.copy()
is_modified = not edited.reset_index(drop=True).equals(st.session_state.source_data.reset_index(drop=True))
if is_modified:
    st.warning("User-modified scenario: exported results will reflect your edited working data.")
validation_errors = validate_data(edited, indicator_label)
if validation_errors:
    for err in validation_errors: st.error(err)

st.subheader("2. Fit a polynomial model")
model = None
metrics = {}
working = pd.DataFrame()
if not validation_errors:
    working = edited.copy()
    working["Year"] = pd.to_numeric(working["Year"]).astype(int)
    working["Value"] = pd.to_numeric(working["Value"]).astype(float)
    working = working.sort_values("Year").reset_index(drop=True)
    if working.Year.nunique() < degree + 1:
        st.error(f"Degree {degree} requires at least {degree+1} distinct years; only {working.Year.nunique()} are available.")
    else:
        base_year = int(working.Year.min())
        tvals = working.Year.to_numpy(dtype=float) - base_year
        coeffs = np.polynomial.polynomial.polyfit(tvals, working.Value.to_numpy(), degree)
        model = np.polynomial.Polynomial(coeffs)
        fitted = model(tvals)
        residuals = working.Value.to_numpy() - fitted
        metrics = {"R-squared (in-sample)": r2_score(working.Value, fitted), "MAE (in-sample)": mean_absolute_error(working.Value, fitted), "Observation count": len(working), "Base year": base_year, "Mean observed value": float(working.Value.mean())}
        eq_terms = [f"{coeffs[i]:.10g}·t^{i}" for i in range(len(coeffs))]
        equation = "y = " + " + ".join(eq_terms) + f", where t = year − {base_year}"
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("R² (in-sample)", f"{metrics['R-squared (in-sample)']:.4f}")
        c2.metric("MAE (in-sample)", f"{metrics['MAE (in-sample)']:.4g} {units}")
        c3.metric("Observations", str(len(working)))
        c4.metric("Historical mean", f"{working.Value.mean():.4g}")
        st.code(equation)
        st.caption(f"Historical range: {working.Year.min()}–{working.Year.max()} • Degree {degree}. Residuals are actual minus fitted values. R² and MAE are in-sample, not guarantees of future accuracy.")
        residual_table = working[["Year","Value"]].copy()
        residual_table["Fitted value"] = fitted
        residual_table["Residual (actual − fitted)"] = residuals
        with st.expander("Show residuals and model details"):
            st.dataframe(residual_table, use_container_width=True)
            st.write("**Polynomial extrapolation warning:** high-degree polynomials can swing sharply outside the observed years. Residuals and MAE describe fit error, not all uncertainty.")
        if len(working) >= max(degree + 3, 10):
            hold_n = max(2, int(round(len(working)*0.2)))
            train = working.iloc[:-hold_n]
            test = working.iloc[-hold_n:]
            if len(train) >= degree+1:
                hc = np.polynomial.polynomial.polyfit(train.Year.to_numpy()-base_year, train.Value.to_numpy(), degree)
                hp = np.polynomial.Polynomial(hc)(test.Year.to_numpy()-base_year)
                st.write(f"Chronological holdout MAE (last {len(test)} observations, model fit on earlier years): **{mean_absolute_error(test.Value, hp):.4g} {units}**. This is separate from in-sample metrics.")

# Comparison data, plotted using same indicator
comparison_data = {}
for label in comparison_labels:
    try:
        cdf, curl = fetch_indicator(COUNTRIES[label], indicator_code, start_year, end_year)
        comparison_data[label] = (cdf, curl)
    except Exception as exc:
        st.warning(f"Comparison data unavailable for {label}: {exc}")

# Plot historical data, smooth curve, extrapolation, and estimates
st.subheader("3. Graph and year calculator")
single_year = st.number_input("Year for single-year estimate (1950–2100)", min_value=1950, max_value=2100, value=min(max(int(working.Year.max()) if not working.empty else current_year-5,1950),2100), step=1)
if st.button("Calculate value"):
    st.session_state.calculated_year = int(single_year)
if st.session_state.get("calculated_year") is not None and model is not None:
    yr = st.session_state.calculated_year
    val = float(model(yr - int(working.Year.min())))
    within = int(working.Year.min()) <= yr <= int(working.Year.max())
    observed = working.loc[working.Year == yr, "Value"]
    st.write(f"**Year:** {yr} • **Estimate:** {val:,.6g} {units} • **Fitted range:** {working.Year.min()}–{working.Year.max()} • **Classification:** {'within fitted historical range' if within else 'outside fitted range (extrapolation)'}")
    if not observed.empty:
        st.caption(f"An actual observation exists for this year: {observed.iloc[0]:,.6g} {units}. The model estimate is fitted, not the same thing as the actual observation.")
    if not within:
        st.warning("Extrapolation can be unreliable. The prediction is not clipped, and no validated 95% prediction interval is implied.")

with st.expander("Batch year estimates"):
    batch_mode = st.radio("Input type", ["Year range", "Custom comma-separated years"], horizontal=True)
    if batch_mode == "Year range":
        a,b = st.columns(2)
        with a: batch_start = st.number_input("Start year", 1950, 2100, int(working.Year.min()) if not working.empty else 2000, key="batch_start")
        with b: batch_end = st.number_input("End year", 1950, 2100, min(2100, (int(working.Year.max()) if not working.empty else 2020)+10), key="batch_end")
        batch_step = st.slider("Year step", 1, 10, 1, key="batch_step")
        batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_step))) if batch_end >= batch_start else []
        if batch_end < batch_start: st.error("End year must be greater than or equal to start year.")
    else:
        custom = st.text_input("Years separated by commas", "2000, 2010, 2020")
        batch_years = []
        for part in custom.split(","):
            part = part.strip()
            try:
                if not re.fullmatch(r"\d{4}", part): raise ValueError()
                yy = int(part)
                if not 1950 <= yy <= 2100: raise ValueError()
                batch_years.append(yy)
            except ValueError:
                if part: st.warning(f"Ignored invalid year input: {part!r}. Enter an integer year from 1950 to 2100.")
        batch_years = sorted(set(batch_years))
    if st.button("Calculate batch"):
        if model is None: st.error("A valid model is required before calculating.")
        elif not batch_years: st.error("Enter at least one valid year.")
        else:
            by = np.array(batch_years, dtype=int)
            vals = model(by - int(working.Year.min()))
            st.session_state.batch_results = pd.DataFrame({"Year": by, "Estimate": vals, "Units": units, "Classification": np.where((by>=working.Year.min())&(by<=working.Year.max()),"within fitted range","outside fitted range")})
    batch_results = st.session_state.get("batch_results", pd.DataFrame())
    if not batch_results.empty: st.dataframe(batch_results, use_container_width=True)

if model is not None:
    base_year = int(working.Year.min())
    last_year = int(working.Year.max())
    display_end = min(2100, last_year + int(horizon))
    curve_years = np.linspace(float(working.Year.min()), float(display_end), max(250, (display_end-int(working.Year.min())+1)*4))
    curve_vals = model(curve_years-base_year)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(working.Year, working.Value, label="Observed / edited data", s=25)
    ax.plot(curve_years, curve_vals, label="Polynomial fit", linewidth=2)
    ax.axvline(last_year, linestyle="--", linewidth=1, label="Historical boundary")
    if display_end > last_year:
        ax.axvspan(last_year, display_end, alpha=0.08, label="Extrapolated display segment")
    sampled = working.iloc[::sample_step]
    ax.scatter(sampled.Year, sampled.Value, marker="x", s=50, label=f"Sample markers every {sample_step} observations")
    if st.session_state.get("calculated_year") is not None:
        yy = st.session_state.calculated_year
        ax.scatter([yy], [model(yy-base_year)], marker="*", s=150, label="Single-year estimate")
    batch_results = st.session_state.get("batch_results", pd.DataFrame())
    if not batch_results.empty:
        ax.scatter(batch_results.Year, batch_results.Estimate, marker="D", s=35, label="Batch estimates")
    for label,(cdf,curl) in comparison_data.items():
        ax.plot(cdf.Year, cdf.Value, marker=".", linewidth=1, label=f"{label} observed")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_label} ({units})")
    ax.set_title(f"{country_label}: {indicator_label}")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    chart_png = fig_to_png(fig)
else:
    chart_png = None

# Function context / course-level learning lab
st.divider()
st.header("4. Function studio and course pathway")
pathway = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
context = st.selectbox("Context", list(CONTEXTS))
dep_default, dep_units_default, indep_default, indep_units_default, context_note = CONTEXTS[context]
col1,col2,col3,col4 = st.columns(4)
with col1: dep_name = st.text_input("Dependent variable", dep_default)
with col2: dep_units = st.text_input("Dependent units", dep_units_default)
with col3: indep_name = st.text_input("Independent variable", indep_default)
with col4: indep_units = st.text_input("Independent units", indep_units_default)
context_note = st.text_area("Context interpretation", context_note)
function_source = st.radio("Function source", ["World Bank curve of best fit", "Suggested function", "Custom function"], horizontal=True)
suggestions = {
    "Business": "x^2 - 4*x + 8", "Engineering": "x^2 - 2*x", "Medical": "exp(-0.2*x)",
    "Sports science": "2*x + 0.5*sin(x)", "Biology": "100*exp(0.1*x)",
    "Bioengineering": "x^3 - 3*x", "Economic forecasting": "0.02*x^2 + x",
    "Weather": "10 + 5*sin(x)", "Custom": "x^2 - 4*x + 8",
}
custom_expr = st.text_input("Function in x (use ^ for powers)", suggestions[context])
analysis_left, analysis_right = st.columns(2)
with analysis_left:
    interval_start = st.number_input("Analysis interval start", value=0.0, key="analysis_start")
with analysis_right:
    interval_end = st.number_input("Analysis interval end", value=10.0, key="analysis_end")
point = st.slider("Current input x", min_value=-10.0, max_value=20.0, value=2.0, step=0.1)
show_f = st.checkbox("Show function f(x)", True)
show_d1 = st.checkbox("Show first derivative f′(x)", True)
show_d2 = st.checkbox("Show second derivative f″(x)", True)
play = st.checkbox("Automatic playback (updates on rerun)", False)
speed = st.select_slider("Playback step size", options=[0.1,0.25,0.5,1.0,2.0], value=0.5)
if play:
    st.caption(f"Playback mode selected: move the input slider by {speed} per update. Streamlit reruns are user-driven; this app does not run a background animation timer.")
    st.session_state.playback_value = float(point + speed)
    if st.button("Use next playback value"):
        st.session_state.playback_value = float(point + speed)
        st.info(f"Next suggested input: {st.session_state.playback_value:g}. Move the slider to continue.")
if st.button("Reset simulation"):
    st.session_state.playback_value = 0.0
    st.info("Simulation reset reference value to x = 0. Move the slider to reset the graph point.")

expr = None
expr_error = None
try:
    if function_source == "World Bank curve of best fit":
        if model is None: raise ValueError("The historical model is unavailable. Fix the source data or select another function source.")
        # Function studio x is a year in the same units as the historical year.
        expr = sp.Float(1) * sum(sp.Float(float(model.coef[i])) * (x - int(working.Year.min()))**i for i in range(len(model.coef)))
        indep_name, indep_units = "Year", "years"
        context_note = "This is the polynomial fitted to the selected World Bank data. It describes a historical pattern and should not be treated as a reliable long-range forecast."
    elif function_source == "Suggested function":
        expr = safe_parse(suggestions[context])
    else:
        expr = safe_parse(custom_expr)
except Exception as exc:
    expr_error = str(exc)
if expr_error:
    st.error(f"Function error: {expr_error}")
if expr is not None:
    f = numeric_fn(expr)
    d1_expr = sp.diff(expr, x)
    d2_expr = sp.diff(d1_expr, x)
    d1, d2 = numeric_fn(d1_expr), numeric_fn(d2_expr)
    st.markdown(f"**Function:** `{sp.sstr(expr)}`  \n**First derivative:** `{sp.sstr(d1_expr)}`  \n**Second derivative:** `{sp.sstr(d2_expr)}`")
    try:
        fv, d1v, d2v = float(f(point)), float(d1(point)), float(d2(point))
        st.write(f"At x = **{point:g}**: {dep_name} = **{fmt_num(fv)} {dep_units}**; rate of change = **{fmt_num(d1v)} {dep_units}/{indep_units}**; change in rate = **{fmt_num(d2v)} {dep_units}/{indep_units}²**.")
        st.write(context_note)
    except Exception:
        st.warning("The function is undefined or non-real at the selected input.")
    if interval_end == interval_start:
        st.warning("Choose different interval endpoints to calculate average rate of change.")
    else:
        try:
            fa, fb = float(f(interval_start)), float(f(interval_end))
            avg = (fb-fa)/(interval_end-interval_start)
            st.write(f"**Average rate of change** from {interval_start:g} to {interval_end:g}: {fmt_num(avg)} {dep_units}/{indep_units}.")
        except Exception:
            avg = np.nan
            st.warning("Average rate of change is undefined on this interval.")
    if pathway in ("Precalculus", "Calculus"):
        st.write(f"**Degree / shape note:** the displayed expression has the structure `{sp.sstr(expr)}`. Coefficients and degree influence scale, turning points, and end behavior.")
    if pathway == "Honors Algebra 2":
        st.write("**Algebra 2 focus:** inspect domain, range over the displayed interval, end behavior, graph trends, and average rates of change.")
    if pathway == "Calculus":
        st.subheader("Calculus analysis")
        st.write(f"**f′(x):** `{sp.sstr(d1_expr)}`  \n**f″(x):** `{sp.sstr(d2_expr)}`")
        try:
            critical = sp.solve(sp.Eq(d1_expr, 0), x)
            crit_real = []
            for cp in critical:
                if cp.is_real is True or (cp.is_number and abs(float(sp.im(cp))) < 1e-9):
                    crit_real.append(float(sp.re(cp)))
            st.write("**Critical-number candidates (real roots of f′):** " + (", ".join(f"{v:.6g}" for v in crit_real) if crit_real else "none found symbolically"))
            extrema = []
            for cp in crit_real:
                try:
                    second = float(d2(cp))
                    if second > 1e-8: extrema.append(f"local minimum near x={cp:.6g} (f″>0)")
                    elif second < -1e-8: extrema.append(f"local maximum near x={cp:.6g} (f″<0)")
                    else: extrema.append(f"x={cp:.6g}: second-derivative test inconclusive")
                except Exception: extrema.append(f"x={cp:.6g}: test inconclusive")
            st.write("**Derivative-test classification:** " + ("; ".join(extrema) if extrema else "No classified local extrema found."))
        except Exception:
            st.write("Critical points could not be solved symbolically; inspect the graph and derivative numerically.")
        if interval_end > interval_start:
            try:
                candidates = [float(interval_start), float(interval_end)] + [v for v in crit_real if interval_start <= v <= interval_end]
                vals = [(v, float(f(v))) for v in candidates]
                vals = [(v,yv) for v,yv in vals if np.isfinite(yv)]
                if vals:
                    lo, hi = min(vals,key=lambda q:q[1]), max(vals,key=lambda q:q[1])
                    st.write(f"**Absolute extrema on closed interval:** minimum ≈ ({lo[0]:.6g}, {lo[1]:.6g}); maximum ≈ ({hi[0]:.6g}, {hi[1]:.6g}).")
            except Exception:
                st.write("Could not evaluate closed-interval extrema.")
        try:
            inf_candidates = sp.solve(sp.Eq(d2_expr,0),x)
            inflections=[]
            for ip in inf_candidates:
                if ip.is_real is True or (ip.is_number and abs(float(sp.im(ip)))<1e-9):
                    ipf=float(sp.re(ip))
                    delta=max(1e-4, abs(ipf)*1e-4)
                    left=float(d2(ipf-delta)); right=float(d2(ipf+delta))
                    if np.isfinite(left) and np.isfinite(right) and left*right < 0:
                        inflections.append(f"x≈{ipf:.6g}, f(x)≈{float(f(ipf)):.6g}")
            st.write("**Inflection candidates verified by concavity change:** " + ("; ".join(inflections) if inflections else "none verified"))
        except Exception:
            st.write("Inflection analysis was inconclusive.")
    # Domain-aware graph: non-finite and complex values become gaps, avoiding false connections over discontinuities.
    lo, hi = sorted((float(interval_start), float(interval_end)))
    if lo == hi: lo, hi = point-5, point+5
    gx = np.linspace(lo, hi, 700)
    fig2, ax2 = plt.subplots(figsize=(10,4.5))
    try:
        gy = np.asarray(f(gx), dtype=np.complex128)
        gy = np.where(np.abs(gy.imag)<1e-7, gy.real, np.nan)
        gy = np.where(np.isfinite(gy), gy, np.nan)
        if show_f: ax2.plot(gx, gy, label="f(x)")
    except Exception: pass
    try:
        if show_d1:
            yy = np.asarray(d1(gx), dtype=np.complex128)
            yy = np.where(np.abs(yy.imag)<1e-7, yy.real, np.nan)
            ax2.plot(gx, np.where(np.isfinite(yy), yy, np.nan), label="f′(x)")
        if show_d2:
            yy = np.asarray(d2(gx), dtype=np.complex128)
            yy = np.where(np.abs(yy.imag)<1e-7, yy.real, np.nan)
            ax2.plot(gx, np.where(np.isfinite(yy), yy, np.nan), label="f″(x)")
        ypoint = float(f(point))
        if show_f and np.isfinite(ypoint):
            ax2.scatter([point],[ypoint],s=70,marker="o",label="Current point")
            slope = float(d1(point))
            tangent_x = np.linspace(point-2,point+2,80)
            ax2.plot(tangent_x, ypoint+slope*(tangent_x-point), linestyle="--", label="Tangent line")
        if interval_end != interval_start:
            ax2.plot([interval_start, interval_end], [float(f(interval_start)), float(f(interval_end))], linestyle=":", label="Secant line")
    except Exception:
        st.warning("Some graph values are undefined in this interval; gaps are left where possible.")
    ax2.axhline(0,linewidth=.7)
    ax2.axvline(0,linewidth=.7)
    ax2.set_xlabel(f"{indep_name} ({indep_units})")
    ax2.set_ylabel(f"{dep_name} ({dep_units})")
    ax2.set_title(f"{pathway}: {context}")
    ax2.grid(True,alpha=.25); ax2.legend(fontsize=8,ncol=2); fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    function_png = fig_to_png(fig2)
else:
    function_png = None

# Exports reflect current selected data, fit, and calculated-year results.
st.divider()
st.header("5. Export current work")
summary_lines = [
    f"Country: {country_label}",
    f"Indicator: {indicator_label} ({indicator_code})",
    f"Units: {units}",
    f"Source: {source_url}",
    f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"Data status: {'user-modified scenario' if is_modified else 'source data'}",
]
if model is not None:
    summary_lines += [f"Equation: {equation}", f"Degree: {degree}", f"Base year: {base_year}", f"R-squared (in-sample): {metrics['R-squared (in-sample)']:.8g}", f"MAE (in-sample): {metrics['MAE (in-sample)']:.8g} {units}", f"Historical range: {working.Year.min()}–{working.Year.max()}", f"Observation count: {len(working)}"]
summary = "\n".join(summary_lines)
csv_data = working.to_csv(index=False).encode("utf-8") if not working.empty else edited.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", data=csv_data, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
    pd.DataFrame({"Item": summary_lines}).to_excel(writer, sheet_name="Summary", index=False)
    (working if not working.empty else edited).to_excel(writer, sheet_name="Working Data", index=False)
    st.session_state.get("batch_results", pd.DataFrame()).to_excel(writer, sheet_name="Calculated Years", index=False)
st.download_button("Download Excel workbook", data=excel_buf.getvalue(), file_name="hispanic_heritage_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
pdf_bytes = make_pdf("Hispanic Heritage Historical Data Explorer", summary, chart_png, working if not working.empty else edited, st.session_state.get("batch_results", pd.DataFrame()))
st.download_button("Download PDF report", data=pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(summary)
    st.write(f"Current data status: {'user-modified scenario' if is_modified else 'source data'}.")
    if model is not None:
        st.write(f"Model equation: {equation}")
        st.write(f"In-sample R² = {metrics['R-squared (in-sample)']:.4f}; in-sample MAE = {metrics['MAE (in-sample)']:.4g} {units}.")

st.caption("Educational tool only. Polynomial fits can be unstable outside the observed range. No causal explanations or validated prediction intervals are provided.")
