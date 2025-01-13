import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from fpdf import FPDF

def compute_rsi(series, period=14):
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gains = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    losses = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(series, period=20, num_std=2):
    """Compute Bollinger Bands (Upper and Lower)."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def compute_moving_average_crossovers(series, short_window=50, long_window=200):
    """Return the difference between short and long moving average."""
    ma_short = series.rolling(window=short_window).mean()
    ma_long = series.rolling(window=long_window).mean()
    return ma_short - ma_long

def label_market_condition(df, forecast_days=5):
    """
    Label each day as bullish or bearish.
    A simple rule: if Close price in `forecast_days` is higher than
    today's close, label is 1 (bullish) else 0 (bearish).
    """
    df['Future_Close'] = df['Close'].shift(-forecast_days)
    df['Label'] = (df['Future_Close'] > df['Close']).astype(int)
    return df

def generate_pdf_report(df_probs, feature_importances, features):
    """
    Generate a PDF report of classification probabilities and feature importances.
    """
    timestamp_utc = dt.datetime.now(dt.UTC).strftime('%Y%m%d%H%M%S')
    pdf_file_name = f"Market_Regime_Classification_Report_{timestamp_utc}.pdf"

    # Initialize FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Market Regime Classification Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated (UTC): {timestamp_utc}", ln=1, align='C')

    pdf.ln(5)  # Add a small vertical space

    # Classification probabilities (show last few rows)
    pdf.cell(200, 10, txt="Daily Classification Probabilities (Bullish):", ln=1)
    last_few = df_probs.tail(10)  # e.g. last 10 rows
    for idx, row in last_few.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        prob_str = f"{row['Bullish_Prob']:.3f}"
        pdf.cell(200, 8, txt=f"{date_str} -> {prob_str}", ln=1)

    pdf.ln(5)  # Add a small vertical space

    # Feature importances
    pdf.cell(200, 10, txt="Feature Importances (RandomForest):", ln=1)
    for feat, imp in zip(features, feature_importances):
        pdf.cell(200, 8, txt=f"{feat}: {imp:.4f}", ln=1)

    # Optionally, save a feature importance plot to embed
    plt.figure(figsize=(6, 4))
    plt.bar(features, feature_importances)
    plt.title("Feature Importances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_file = f"feature_importances_{timestamp_utc}.png"
    plt.savefig(plot_file)
    plt.close()

    # Embed the plot into PDF
    pdf.add_page()
    pdf.cell(200, 10, txt="Feature Importance Plot:", ln=1, align='L')
    pdf.image(plot_file, x=10, y=30, w=180)

    # Output PDF
    pdf.output(pdf_file_name)
    print(f"PDF report saved as {pdf_file_name}")

def main():
    # --------------------
    # 1. Download S&P 500 data
    # --------------------
    start_date = "1990-01-01"
    end_date = "2025-01-10"
    ticker = "^GSPC"

    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.droplevel(1) # Remove the MultiIndex
    df.dropna(inplace=True)
    # Ensure daily frequency only
    df = df.asfreq('B').ffill()

    # --------------------
    # 2. Compute indicators
    # --------------------
    df['RSI'] = compute_rsi(df['Close'])
    df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['Close'])
    df['BB_Width'] = df['Upper_BB'] - df['Lower_BB']  # Bollinger Band width
    df['MA_Crossover'] = compute_moving_average_crossovers(df['Close'])

    # --------------------
    # 3. Label data
    # --------------------
    df = label_market_condition(df, forecast_days=5)

    # Because we used shift(-5), drop last 5 rows that have no future data
    df.dropna(inplace=True)

    # --------------------
    # 4. Create features & target
    # --------------------
    features = ['RSI', 'Upper_BB', 'Lower_BB', 'BB_Width', 'MA_Crossover']
    X = df[features].values
    y = df['Label'].values

    # --------------------
    # 5. Train / Test split
    # --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # --------------------
    # 6. RandomForest training
    # --------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # --------------------
    # 7. Predictions & Probabilities
    # --------------------
    y_pred = rf.predict(X_test)
    y_pred_probs = rf.predict_proba(X_test)  # shape (N, 2), columns: [Prob of Bearish, Prob of Bullish]

    # Evaluate quickly
    print("Accuracy on test:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    # --------------------
    # 8. Prepare daily probabilities for entire dataset (not just test)
    # --------------------
    # Usually, you'd retrain on the entire dataset or use a walk-forward scheme.  
    # For simplicity here, let's just get probabilities for all days in df:
    all_probs = rf.predict_proba(X)  # corresponds to df.index
    # Store in a DataFrame
    df_probs = pd.DataFrame(
        index=df.index,
        data={
            "Bearish_Prob": all_probs[:, 0],
            "Bullish_Prob": all_probs[:, 1],
        }
    )

    # --------------------
    # 9. Feature importance
    # --------------------
    importances = rf.feature_importances_

    # --------------------
    # 10. Generate PDF report
    # --------------------
    generate_pdf_report(df_probs, importances, features)

if __name__ == "__main__":
    main()
