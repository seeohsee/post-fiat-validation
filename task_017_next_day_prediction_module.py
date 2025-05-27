import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fpdf import FPDF
from datetime import datetime, timezone
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for PDF embedding

# Define tickers
tickers = ["AAPL", "LMND", "QBTS", "RGTI", "CB", "CINF", "AROW", "AFRM", "CSX", "RDFN", "TGT", "CLX", "DIS"]

# PDF setup
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
pdf_filename = f"task_017_outputs/task_017_Report_{timestamp}.pdf"
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, f"Stock Classification Model Report - {timestamp}", ln=True)

for ticker in tickers:
    try:
        # Fetch the data
        print(f"Downloading price history for {ticker}...")
        raw = yf.download(ticker, progress=False)
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        # Features
        df["RSI"]      = RSIIndicator(close=df["Close"]).rsi()
        df["SMA_10"]   = SMAIndicator(close=df["Close"], window=10).sma_indicator()
        df["SMA_30"]   = SMAIndicator(close=df["Close"], window=30).sma_indicator()
        df["MA_Cross"] = (df["SMA_10"] > df["SMA_30"]).astype(int)
        df["Target"]   = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df[["RSI", "MA_Cross", "Volume", "Target", "Close"]].dropna()

        # Split
        X          = df[["RSI", "MA_Cross", "Volume"]]
        y          = df["Target"]
        dates      = df.index
        prices     = df["Close"]
        split_idx  = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        date_test  = dates[split_idx:]
        price_test = prices.iloc[split_idx:]

        # Train the model
        print(f"\t--> Training...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\t\t--> Accuracy = {accuracy:.2%}")

        # Ensure all arrays are the same length
        assert len(X_test) == len(y_test) == len(y_pred) == len(date_test)

        # Create a prediction log
        print("\t--> Writing to log...")
        log_df = pd.DataFrame({
            "Date": date_test,
            "Prediction": np.where(y_pred == 1, "higher", "lower"),
            "Actual": np.where(y_test == 1, "higher", "lower"),
            "Correct": y_pred == y_test,
            "Close": price_test.values
        })
        log_df.to_csv(f"task_017_outputs/{ticker}_prediction_log.csv", index=False)

        # Strategy Simulation
        print("\t--> Strategy Simulation...")
        position = np.where(log_df["Prediction"] == "higher", 1, 0)
        next_day_return = log_df["Close"].pct_change().shift(-1)
        strategy_return = (position * next_day_return).fillna(0)
        cumulative_return = (1 + strategy_return).cumprod()

        # Plot the cumulative return
        plt.figure(figsize=(6, 3))
        plt.plot(log_df["Date"], cumulative_return, label="Strategy")
        plt.title(f"Cumulative Return - {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.tight_layout()
        return_plot_path = f"task_017_outputs/{ticker}_return.png"
        plt.savefig(return_plot_path)
        plt.close()

        # Find the most recent correct/incorrect predictions
        correct_dates   = log_df[log_df["Correct"]]["Date"]
        incorrect_dates = log_df[~log_df["Correct"]]["Date"]
        last_correct    = correct_dates.max() if not correct_dates.empty else "N/A"
        last_incorrect  = incorrect_dates.max() if not incorrect_dates.empty else "N/A"

        # Write summary to PDF
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"{ticker}: Accuracy = {accuracy:.2%}", ln=True)
        pdf.cell(0, 10, f"   Last Correct Prediction:   {last_correct}", ln=True)
        pdf.cell(0, 10, f"   Last Incorrect Prediction: {last_incorrect}", ln=True)

        # Embed plots in the PDF
        pdf.image(return_plot_path, w=pdf.w / 2)
        pdf.ln(45)
    except Exception as e:
        print(f"EXCEPTION for {ticker} ~> {e}")
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"{ticker}: Error - {str(e)}", ln=True)
        pdf.set_text_color(0, 0, 0)

# Save PDF
pdf.output(pdf_filename)
print(f"PDF report generated: {pdf_filename}")
