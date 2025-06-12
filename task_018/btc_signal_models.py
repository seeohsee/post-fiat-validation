#!/usr/bin/env python3
"""
btc_signal_models.py
--------------------
Two-model pipeline using only the provided on-chain signal features.

Outputs
-------
btc_predictions_<UTC-timestamp>.csv
btc_models_report_<UTC-timestamp>.pdf
"""
from __future__ import annotations
import os, sys, math
from datetime import datetime, timezone

from fpdf import FPDF, XPos, YPos
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

# ────────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREP DATA
# ────────────────────────────────────────────────────────────────────────────────
def load_frame(path: str) -> tuple[pd.DataFrame, str]:
    """Read CSV, detect price column, sort chronologically."""
    df = pd.read_csv(path)

    req = ["Signal Intensity Percent", "Signal Intensity Percent Change"]
    if any(c not in df.columns for c in req):
        missing = [c for c in req if c not in df.columns]
        raise ValueError(f"Required column(s) missing: {', '.join(missing)}")

    # Find the first column whose name contains "price" but not "intensity".
    price_cols = [c for c in df.columns if "price" in c.lower() and "intensity" not in c.lower()]
    if not price_cols:
        raise ValueError("Couldn’t identify BTC price column (expects 'price' in its name).")
    price_col = price_cols[0]

    # Date column (if any) to keep order reproducible.
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.sort_values(date_cols[0])
    else:
        df = df.sort_index()

    return df.reset_index(drop=True), price_col


def add_targets_and_lags(df: pd.DataFrame, price_col: str, lags: list[int]) -> pd.DataFrame:
    """Create next-day log-return + direction targets; build lagged features."""
    # Next-day targets
    df["log_return_t+1"] = np.log(df[price_col].shift(-1) / df[price_col])
    df["direction_t+1"] = (df["log_return_t+1"] > 0).astype(int)

    # Lagged copies of the two signals (0 = today already present)
    for lag in lags:
        if lag == 0:
            continue
        df[f"SIP_lag{lag}"]  = df["Signal Intensity Percent"].shift(lag)
        df[f"SIPC_lag{lag}"] = df["Signal Intensity Percent Change"].shift(lag)

    return df.dropna().reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / PREDICT
# ────────────────────────────────────────────────────────────────────────────────
def split_train_test(df: pd.DataFrame, split_ratio: float = 0.7):
    split_idx = int(len(df) * split_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:], split_idx


def train_models(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Pipeline, Pipeline]:
    """Return (regressor, classifier) pipelines fitted on the same feature set."""
    X = df[feature_cols].values
    y_reg = df["log_return_t+1"].values
    y_clf = df["direction_t+1"].values

    regressor = Pipeline(
        [("scaler", StandardScaler()), ("gbr", GradientBoostingRegressor(random_state=42))]
    )
    classifier = Pipeline(
        [("scaler", StandardScaler()), ("gbc", GradientBoostingClassifier(random_state=42))]
    )

    regressor.fit(X, y_reg)
    classifier.fit(X, y_clf)
    return regressor, classifier


def evaluate(y_true, y_pred, *, task: str):
    if task == "reg":
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse}
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {"Accuracy": acc, "F1": f1}


# ────────────────────────────────────────────────────────────────────────────────
# 3. WRITE CSV + PDF
# ────────────────────────────────────────────────────────────────────────────────
def save_csv(
    df: pd.DataFrame,
    split_idx: int,
    pred_reg: np.ndarray,
    pred_clf: np.ndarray,
    out_path: str,
) -> None:
    df_out = df.copy()

    # Fill predictions only for the test rows
    df_out["pred_log_return"] = np.nan
    df_out["pred_direction"] = np.nan

    df_out.loc[df_out.index[split_idx:], "pred_log_return"] = pred_reg
    df_out.loc[df_out.index[split_idx:], "pred_direction"] = pred_clf

    # Residual and hit indicator
    df_out["residual"] = df_out["pred_log_return"] - df_out["log_return_t+1"]
    df_out["correct_direction"] = (df_out["pred_direction"] == df_out["direction_t+1"]).astype(int)

    df_out.to_csv(out_path, index=False)


def plot_series(test_df, pred_reg, price_col, png_dir, ts):
    """Generate two PNGs: log-return comparison & cumulative rebased price."""
    paths = {}
    # 1. Log-return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(test_df.index, test_df["log_return_t+1"], label="Actual", lw=1)
    ax.plot(test_df.index, pred_reg,               label="Predicted", lw=1)
    ax.set_title("Next-day Log-Return")
    ax.set_xlabel("Test-period index")
    ax.set_ylabel("Log-return")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(png_dir, f"log_returns_{ts}.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["log"] = p1

    # 2. Cumulative price path
    rebased_actual = test_df[price_col].iloc[0] * np.exp(test_df["log_return_t+1"].cumsum())
    rebased_pred   = test_df[price_col].iloc[0] * np.exp(pd.Series(pred_reg, index=test_df.index).cumsum())

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rebased_actual, label="Actual", lw=1)
    ax.plot(rebased_pred,   label="Predicted", lw=1)
    ax.set_title("Rebased Cumulative Price")
    ax.set_xlabel("Test-period index")
    ax.set_ylabel("Price (rebased)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p2 = os.path.join(png_dir, f"cum_price_{ts}.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["cum"] = p2
    return paths


def build_pdf(metrics_reg: dict, metrics_clf: dict, plot_paths, out_path: str, n_train: int, n_test: int) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "Bitcoin Signal-Driven Models Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_font("Helvetica", "", 12)
    lines = [
        f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Samples - train: {n_train} | test: {n_test}",
        "Models: GradientBoostingRegressor & GradientBoostingClassifier",
        "",
        "Regression (next-day log-return):",
        f"  MAE  : {metrics_reg['MAE']:.6f}",
        f"  RMSE : {metrics_reg['RMSE']:.6f}",
        "",
        "Classification (direction up/down):",
        f"  Accuracy: {metrics_clf['Accuracy']*100:5.2f}%",
        f"  F1-score: {metrics_clf['F1']:.4f}",
    ]
    for line in lines:
        pdf.cell(0, 8, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Insert plots
    for key in ("log", "cum"):
        pdf.image(plot_paths[key], w=170)
        pdf.ln(4)

    pdf.output(out_path)


# ────────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python btc_signal_models.py /path/to/input.csv")

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        sys.exit(f"File not found: {csv_path}")

    df_raw, price_col = load_frame(csv_path)
    lags = [0, 1, 2, 3, 5]  # today plus up to 5-day history
    df = add_targets_and_lags(df_raw, price_col, lags)

    feature_cols = [c for c in df.columns if c.startswith("SIP")]
    train_df, test_df, split_idx = split_train_test(df)

    reg_model, clf_model = train_models(train_df, feature_cols)

    # Out-of-sample predictions
    X_test = test_df[feature_cols].values
    pred_reg = reg_model.predict(X_test)
    pred_clf = clf_model.predict(X_test)

    metrics_reg = evaluate(test_df["log_return_t+1"], pred_reg, task="reg")
    metrics_clf = evaluate(test_df["direction_t+1"], pred_clf, task="clf")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    csv_out = os.path.join(base_dir, f"btc_predictions_{ts}.csv")
    pdf_out = os.path.join(base_dir, f"btc_models_report_{ts}.pdf")

    save_csv(df, split_idx, pred_reg, pred_clf, csv_out)
    pngs = plot_series(test_df, pred_reg, price_col, base_dir, ts)
    build_pdf(metrics_reg, metrics_clf, pngs, pdf_out, len(train_df), len(test_df))

    print("!!! Finished.")
    print("CSV :", csv_out)
    print("PDF :", pdf_out)


if __name__ == "__main__":
    main()

