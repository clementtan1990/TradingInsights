import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from fredapi import Fred


# ============================================================
# CONFIG
# ============================================================

START_DATE = "1980-01-01"

OUTPUT_DIR = Path("docs")
ASSET_DIR = OUTPUT_DIR / "assets"
DATA_DIR = OUTPUT_DIR / "data"

FCI_UPPER = 6
FCI_LOWER = 2

API_KEY = os.getenv("FRED_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing FRED_API_KEY. Set it as an environment variable or GitHub Actions secret."
    )

fred = Fred(api_key=API_KEY)


# ============================================================
# FRED SERIES MAPPING
# ============================================================

MAPPING = {
    # Policy-linked variables
    "10y_trsy": "DGS10",
    "2y_trsy": "DGS2",
    "fed_deficit": "MTSDS133FMS",
    "real_m2": "M2REAL",

    # Macro pressure variables
    "oil": "WTISPLC",
    "cpi": "CPIAUCSL",
    "real_wage": "AHETPI",

    # Other charting / backtest variables
    "recessions": "USREC",
    "sp500": "SP500",
    "nasdaq100": "NASDAQ100",
}


# ============================================================
# DATA FUNCTIONS
# ============================================================

def get_fred_series(name: str, code: str, start_date: str) -> pd.Series:
    s = fred.get_series(
        code,
        observation_start=start_date,
        frequency="m",
        aggregation_method="eop",
    )
    s.name = name
    return s


def load_raw_data() -> pd.DataFrame:
    series = []

    for name, code in MAPPING.items():
        print(f"Downloading {name}: {code}")
        series.append(get_fred_series(name, code, START_DATE))

    raw = pd.concat(series, axis=1).sort_index()
    raw = raw.ffill()

    return raw


def build_paulsen_fci(raw: pd.DataFrame):
    """
    Replicates Paulsen FCI:
    8 components, each counted as restrictive or not.

    Restrictive if:
    1. 10y yield higher than 12m ago
    2. 2y yield higher than 12m ago
    3. 10y-2y curve flatter than 12m ago
    4. TTM federal deficit/surplus indicates smaller deficit / less fiscal juice
    5. annual real M2 growth rate lower than 12m ago
    6. oil price higher than 12m ago
    7. annual CPI inflation higher than 12m ago
    8. real wage rate lower than 12m ago
    """

    df = raw.copy()

    # Derived series
    df["yc_10y_2y"] = df["10y_trsy"] - df["2y_trsy"]

    # Monthly Treasury Statement deficit/surplus TTM.
    # MTSDS133FMS is deficit/surplus, where deficits are generally negative.
    df["fed_deficit_ttm"] = df["fed_deficit"].rolling(12, min_periods=12).sum()

    # Real M2 is already a FRED real money stock series.
    # Paulsen uses annual growth in real M2, then checks if growth rate declined over 12m.
    df["real_m2_yoy"] = df["real_m2"].pct_change(12)

    # CPI inflation rate, then change versus 12m ago.
    df["cpi_yoy"] = df["cpi"].pct_change(12)

    # Real wage rate.
    # AHETPI is already an index of real average hourly earnings.
    # If you use nominal wages instead, divide by CPI.
    df["real_wage_rate"] = df["real_wage"]

    flags = pd.DataFrame(index=df.index)

    flags["10y_yield_higher"] = (df["10y_trsy"].diff(12) > 0).astype(float)
    flags["2y_yield_higher"] = (df["2y_trsy"].diff(12) > 0).astype(float)
    flags["curve_flatter"] = (df["yc_10y_2y"].diff(12) < 0).astype(float)

    # Since deficits are negative in MTSDS133FMS:
    # deficit shrinking means series rises, e.g. -2000 -> -1000.
    flags["fiscal_juice_smaller"] = (df["fed_deficit_ttm"].diff(12) > 0).astype(float)

    flags["real_m2_growth_lower"] = (df["real_m2_yoy"].diff(12) < 0).astype(float)
    flags["oil_higher"] = (df["oil"].diff(12) > 0).astype(float)
    flags["cpi_inflation_higher"] = (df["cpi_yoy"].diff(12) > 0).astype(float)
    flags["real_wage_lower"] = (df["real_wage_rate"].diff(12) < 0).astype(float)

    # Preserve invalid early rows as NaN
    valid = pd.DataFrame(index=df.index)
    valid["10y_yield_higher"] = df["10y_trsy"].diff(12).notna()
    valid["2y_yield_higher"] = df["2y_trsy"].diff(12).notna()
    valid["curve_flatter"] = df["yc_10y_2y"].diff(12).notna()
    valid["fiscal_juice_smaller"] = df["fed_deficit_ttm"].diff(12).notna()
    valid["real_m2_growth_lower"] = df["real_m2_yoy"].diff(12).notna()
    valid["oil_higher"] = df["oil"].diff(12).notna()
    valid["cpi_inflation_higher"] = df["cpi_yoy"].diff(12).notna()
    valid["real_wage_lower"] = df["real_wage_rate"].diff(12).notna()

    flags = flags.where(valid)

    count = flags.sum(axis=1, min_count=8)
    indicator = count.rolling(3, min_periods=3).mean()

    return df, flags, count, indicator


# ============================================================
# STRATEGY FUNCTIONS
# ============================================================

def build_strategy_panel(df: pd.DataFrame, indicator: pd.Series) -> pd.DataFrame:
    panel = pd.concat(
        {
            "fci": indicator,
            "nasdaq100": df["nasdaq100"],
            "sp500": df["sp500"],
            "recessions": df["recessions"],
        },
        axis=1,
    ).sort_index().ffill()

    panel["ndx_ret"] = panel["nasdaq100"].pct_change()
    panel["spx_ret"] = panel["sp500"].pct_change()

    panel["ndx_10mma"] = panel["nasdaq100"].rolling(10, min_periods=10).mean()
    panel["trend_ok"] = panel["nasdaq100"] > panel["ndx_10mma"]
    panel["fci_ok"] = panel["fci"] < FCI_UPPER

    # Signal calculated at month-end, traded next month.
    panel["raw_signal"] = (panel["fci_ok"] & panel["trend_ok"]).astype(float)
    panel["position"] = panel["raw_signal"].shift(1)

    # Cash return assumed 0%.
    panel["strategy_ret"] = panel["position"] * panel["ndx_ret"]

    panel["strategy_equity"] = (1 + panel["strategy_ret"].fillna(0)).cumprod()
    panel["ndx_equity"] = (1 + panel["ndx_ret"].fillna(0)).cumprod()
    panel["spx_equity"] = (1 + panel["spx_ret"].fillna(0)).cumprod()

    panel["strategy_dd"] = panel["strategy_equity"] / panel["strategy_equity"].cummax() - 1
    panel["ndx_dd"] = panel["ndx_equity"] / panel["ndx_equity"].cummax() - 1
    panel["spx_dd"] = panel["spx_equity"] / panel["spx_equity"].cummax() - 1

    return panel


def calc_metrics(ret: pd.Series, equity: pd.Series) -> dict:
    data = pd.concat({"ret": ret, "equity": equity}, axis=1).dropna()

    if data.empty:
        return {}

    n_months = len(data)
    years = n_months / 12

    total_return = data["equity"].iloc[-1] - 1
    cagr = data["equity"].iloc[-1] ** (1 / years) - 1
    vol = data["ret"].std() * np.sqrt(12)
    sharpe = (data["ret"].mean() * 12) / vol if vol > 0 else np.nan
    dd = data["equity"] / data["equity"].cummax() - 1
    max_dd = dd.min()

    return {
        "months": n_months,
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ============================================================
# CHARTING HELPERS
# ============================================================

def shade_periods(ax, flag: pd.Series, color="grey", alpha=0.2):
    flag = flag.fillna(False).astype(bool)

    in_period = False
    start = None

    for date, value in flag.items():
        if value and not in_period:
            start = date
            in_period = True
        elif not value and in_period:
            ax.axvspan(start, date, color=color, alpha=alpha, linewidth=0)
            in_period = False

    if in_period:
        ax.axvspan(start, flag.index[-1], color=color, alpha=alpha, linewidth=0)


def plot_fci_vs_nasdaq(panel: pd.DataFrame):
    data = panel.dropna(subset=["fci", "nasdaq100", "ndx_10mma"]).copy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.25]},
    )

    ax1, ax2 = axes

    recession_flag = data["recessions"] == 1
    restrictive_flag = data["fci"] >= FCI_UPPER

    shade_periods(ax1, recession_flag, color="grey", alpha=0.22)
    shade_periods(ax2, recession_flag, color="grey", alpha=0.22)
    shade_periods(ax2, restrictive_flag, color="tab:red", alpha=0.08)

    fci_line, = ax1.plot(data.index, data["fci"], linewidth=1.8, label="FCI 3m average")
    upper_line = ax1.axhline(
        FCI_UPPER,
        linestyle="--",
        linewidth=1.2,
        color="red",
        label=f"Restrictive threshold: {FCI_UPPER}",
    )
    lower_line = ax1.axhline(
        FCI_LOWER,
        linestyle="--",
        linewidth=1.2,
        color="green",
        label=f"Easy threshold: {FCI_LOWER}",
    )

    recession_patch = mpatches.Patch(color="grey", alpha=0.22, label="NBER recession")

    ax1.set_title("Financial Conditions Index vs Nasdaq 100")
    ax1.set_ylabel("FCI count")
    ax1.set_ylim(-0.25, 8.25)
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=[fci_line, upper_line, lower_line, recession_patch], loc="upper left")

    ndx_line, = ax2.plot(data.index, data["nasdaq100"], linewidth=1.8, label="Nasdaq 100")
    ma_line, = ax2.plot(data.index, data["ndx_10mma"], linewidth=1.4, label="10MMA")

    restrictive_patch = mpatches.Patch(color="tab:red", alpha=0.08, label=f"FCI ≥ {FCI_UPPER}")

    ax2.set_yscale("log")
    ax2.set_ylabel("Nasdaq 100, log scale")
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=[ndx_line, ma_line, recession_patch, restrictive_patch], loc="upper left")

    ax2.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_minor_locator(mdates.YearLocator(base=1))

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "fci_vs_nasdaq.png", dpi=160)
    plt.close(fig)


def plot_strategy_equity(panel: pd.DataFrame):
    data = panel.dropna(subset=["strategy_equity", "ndx_equity", "spx_equity"]).copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(data.index, data["strategy_equity"], label="FCI + 10MMA Nasdaq/Cash")
    ax.plot(data.index, data["ndx_equity"], label="Buy & Hold Nasdaq 100")
    ax.plot(data.index, data["spx_equity"], label="Buy & Hold S&P 500")

    ax.set_yscale("log")
    ax.set_title("Strategy Equity Curve")
    ax.set_ylabel("Growth of $1, log scale")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "strategy_equity_curve.png", dpi=160)
    plt.close(fig)


def plot_drawdown(panel: pd.DataFrame):
    data = panel.dropna(subset=["strategy_dd", "ndx_dd", "spx_dd"]).copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(data.index, data["strategy_dd"], label="FCI + 10MMA Nasdaq/Cash")
    ax.plot(data.index, data["ndx_dd"], label="Nasdaq 100")
    ax.plot(data.index, data["spx_dd"], label="S&P 500")

    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "drawdown.png", dpi=160)
    plt.close(fig)


def plot_forward_returns_by_quartile(panel: pd.DataFrame):
    data = panel.dropna(subset=["fci", "nasdaq100", "sp500"]).copy()

    q1, q2, q3 = data["fci"].quantile([0.25, 0.50, 0.75])

    def assign_quartile(x):
        if x <= q1:
            return "Q1 easiest"
        if x <= q2:
            return "Q2"
        if x <= q3:
            return "Q3"
        return "Q4 tightest"

    data["fci_quartile"] = data["fci"].map(assign_quartile)

    horizons = [1, 3, 6, 12]
    for h in horizons:
        data[f"ndx_{h}m_fwd"] = data["nasdaq100"].shift(-h) / data["nasdaq100"] - 1

    quartile_order = ["Q1 easiest", "Q2", "Q3", "Q4 tightest"]

    rows = []
    for h in horizons:
        col = f"ndx_{h}m_fwd"
        for q in quartile_order:
            sub = data.loc[data["fci_quartile"] == q, col].dropna()
            rows.append(
                {
                    "horizon": f"{h}m",
                    "quartile": q,
                    "obs": len(sub),
                    "avg_return": sub.mean(),
                    "median_return": sub.median(),
                    "hit_rate": (sub > 0).mean(),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "forward_returns_by_fci_quartile.csv", index=False)

    pivot = summary.pivot(index="quartile", columns="horizon", values="avg_return")
    pivot = pivot.loc[quartile_order, ["1m", "3m", "6m", "12m"]]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(quartile_order))
    width = 0.18

    for i, h in enumerate(["1m", "3m", "6m", "12m"]):
        ax.bar(x + (i - 1.5) * width, pivot[h], width, label=h)

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(quartile_order)
    ax.set_title("Nasdaq 100 Average Forward Returns by FCI Quartile")
    ax.set_ylabel("Average forward return")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Horizon")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "ndx_forward_returns_by_fci_quartile.png", dpi=160)
    plt.close(fig)

    return summary


# ============================================================
# HTML DASHBOARD
# ============================================================

def format_pct(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:.1%}"


def build_dashboard_html(panel: pd.DataFrame, metrics_table: pd.DataFrame):
    latest = panel.dropna(subset=["fci", "nasdaq100", "ndx_10mma"]).iloc[-1]
    prev = panel.dropna(subset=["fci"]).iloc[-2]

    latest_date = latest.name.strftime("%Y-%m-%d")

    fci = latest["fci"]
    prev_fci = prev["fci"]
    ndx = latest["nasdaq100"]
    ndx_10mma = latest["ndx_10mma"]
    fci_ok = bool(latest["fci_ok"])
    trend_ok = bool(latest["trend_ok"])

    model_signal = "Nasdaq / Risk-on" if fci_ok and trend_ok else "Cash / Defensive"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Paulsen FCI Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            background: #f7f7f7;
            color: #222;
        }}
        h1, h2 {{
            margin-bottom: 8px;
        }}
        .card {{
            background: white;
            padding: 18px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.12);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
        }}
        .metric {{
            background: #fafafa;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: bold;
            margin-top: 4px;
        }}
        img {{
            width: 100%;
            max-width: 1200px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }}
        th:first-child, td:first-child {{
            text-align: left;
        }}
        th {{
            background: #eee;
        }}
        .signal {{
            color: {"#147a00" if model_signal.startswith("Nasdaq") else "#b00020"};
        }}
    </style>
</head>
<body>

<h1>Financial Conditions Index Dashboard</h1>
<p>Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>

<div class="card">
    <h2>Current Regime</h2>
    <div class="grid">
        <div class="metric">
            <div class="metric-label">Latest data month</div>
            <div class="metric-value">{latest_date}</div>
        </div>
        <div class="metric">
            <div class="metric-label">FCI 3m avg</div>
            <div class="metric-value">{fci:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Previous FCI</div>
            <div class="metric-value">{prev_fci:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Nasdaq vs 10MMA</div>
            <div class="metric-value">{"Above" if trend_ok else "Below"}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Model signal</div>
            <div class="metric-value signal">{model_signal}</div>
        </div>
    </div>
</div>

<div class="card">
    <h2>FCI vs Nasdaq 100</h2>
    <img src="assets/fci_vs_nasdaq.png">
</div>

<div class="card">
    <h2>Strategy Equity Curve</h2>
    <img src="assets/strategy_equity_curve.png">
</div>

<div class="card">
    <h2>Drawdown</h2>
    <img src="assets/drawdown.png">
</div>

<div class="card">
    <h2>Nasdaq Forward Returns by FCI Quartile</h2>
    <img src="assets/ndx_forward_returns_by_fci_quartile.png">
</div>

<div class="card">
    <h2>Backtest Metrics</h2>
    {metrics_table.to_html(index=True, border=0)}
</div>

</body>
</html>
"""

    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    transformed, flags, count, indicator = build_paulsen_fci(raw)
    panel = build_strategy_panel(transformed, indicator)

    raw.to_csv(DATA_DIR / "raw_fred_data.csv")
    transformed.to_csv(DATA_DIR / "transformed_data.csv")
    flags.to_csv(DATA_DIR / "paulsen_component_flags.csv")
    count.rename("monthly_count").to_csv(DATA_DIR / "paulsen_monthly_count.csv")
    indicator.rename("fci_3m_avg").to_csv(DATA_DIR / "paulsen_fci_indicator.csv")
    panel.to_csv(DATA_DIR / "strategy_panel.csv")

    plot_fci_vs_nasdaq(panel)
    plot_strategy_equity(panel)
    plot_drawdown(panel)
    plot_forward_returns_by_quartile(panel)

    metrics = {
        "FCI + 10MMA Nasdaq/Cash": calc_metrics(
            panel["strategy_ret"],
            panel["strategy_equity"],
        ),
        "Buy & Hold Nasdaq 100": calc_metrics(
            panel["ndx_ret"],
            panel["ndx_equity"],
        ),
        "Buy & Hold S&P 500": calc_metrics(
            panel["spx_ret"],
            panel["spx_equity"],
        ),
    }

    metrics_table = pd.DataFrame(metrics).T

    # Make the HTML table readable
    metrics_display = metrics_table.copy()
    for col in ["total_return", "cagr", "vol", "max_drawdown"]:
        metrics_display[col] = metrics_display[col].map(format_pct)

    metrics_display["sharpe"] = metrics_display["sharpe"].map(
        lambda x: "n/a" if pd.isna(x) else f"{x:.2f}"
    )

    metrics_display["months"] = metrics_display["months"].map(
        lambda x: "n/a" if pd.isna(x) else f"{int(x)}"
    )

    metrics_display.columns = [
        "Months",
        "Total Return",
        "CAGR",
        "Vol",
        "Sharpe",
        "Max Drawdown",
    ]

    metrics_table.to_csv(DATA_DIR / "backtest_metrics.csv")

    build_dashboard_html(panel, metrics_display)

    latest = panel.dropna(subset=["fci", "nasdaq100", "ndx_10mma"]).iloc[-1]

    print("\nDashboard generated successfully.")
    print(f"Latest date: {latest.name.date()}")
    print(f"Latest FCI: {latest['fci']:.2f}")
    print(f"Nasdaq: {latest['nasdaq100']:.2f}")
    print(f"Nasdaq 10MMA: {latest['ndx_10mma']:.2f}")
    print(f"FCI OK: {bool(latest['fci_ok'])}")
    print(f"Trend OK: {bool(latest['trend_ok'])}")
    print(f"Output: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
