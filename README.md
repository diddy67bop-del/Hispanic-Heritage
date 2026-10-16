# Hispanic Heritage Historical Data Explorer

A self-contained Streamlit app for exploring World Bank historical indicators for nine Latin American countries, fitting polynomial models, editing a working data table, and exploring functions through Honors Algebra 2, Precalculus, and Calculus lenses.

## Run locally

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Open Streamlit Community Cloud and create an app from that repository.
4. Set the main file path to `app.py`, then deploy.
5. If the app fails, open the app's logs and look for dependency installation, network/API, or Python traceback errors.

## Deploy to Render

Create a **Python Web Service** linked to the GitHub repository.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Verification checklist

- Confirm source country, indicator code, units, retrieval date, and coverage.
- Edit a year/value, verify the modified-scenario warning, then use **Reset edited data to source**.
- Try duplicate years, non-integer years, non-numeric values, and too few distinct years for the selected polynomial degree.
- Check the polynomial equation, in-sample R², MAE, residual table, and chronological holdout score when available.
- Calculate a year inside and outside the fitted range; compare the same year in batch results.
- Test malformed custom years and function expressions, including division by zero or logarithms outside their domain.
- Toggle each graph curve and compare contexts/course pathways.
- Download CSV, Excel, and PDF and confirm that they contain the current working data and batch results.
- Deploy and check logs if the app cannot install packages or reach the World Bank API.

## Mathematical notes and limitations

The polynomial uses centered time `t = year - base year` to improve numerical conditioning. R² and MAE are in-sample metrics. A chronological holdout score is shown separately when enough data are available. High-degree polynomial extrapolations can be implausible; estimates are not clipped and no validated 95% prediction intervals are claimed. Derivative-based classifications are numerical/symbolic educational aids and may be inconclusive for complicated functions. The app does not infer historical causes.
