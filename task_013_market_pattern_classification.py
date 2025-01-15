import yfinance as yf
import pandas as pd
import numpy as np
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
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

def label_market_condition_for_reversal(df, lookahead_days=5):
    """
    Label data for trend reversals.
    A simple rule: if the price in the next `lookahead_days` is *lower* than today's price
    after an uptrend, it might be a reversal (label 1). Otherwise 0.
    
    You can tweak this logic. For demonstration, let’s define:
       - We look at slope of last few days vs. next few days, or
         just if the forward price is lower than current after an up move.
    """
    # We'll do a naive approach:
    # If price is currently higher than it was 5 days ago (an uptrend),
    # and the price 5 days later is lower than the current price, label = 1 (trend reversal).
    
    df['Prev_Close_5'] = df['Close'].shift(5)
    df['Future_Close_5'] = df['Close'].shift(-lookahead_days)
    
    # Condition: was there an uptrend from T-5 to T?
    df['was_up'] = (df['Close'] > df['Prev_Close_5']).astype(int)
    # Condition: is future price now lower than current? -> potential reversal
    df['future_lower'] = (df['Future_Close_5'] < df['Close']).astype(int)
    
    # Combine to define "reversal"
    df['Label'] = ((df['was_up'] == 1) & (df['future_lower'] == 1)).astype(int)
    
    # Drop rows that have become NaN (start or end)
    df.dropna(inplace=True)
    
    return df

def generate_pdf_report(
    df_probs_rf, 
    feature_importances, 
    features,
    accuracy_rf,
    accuracy_svm_linear,
    accuracy_svm_rbf,
    cm_rf,
    cm_svm_linear,
    cm_svm_rbf,
    cv_scores_rf,
    cv_scores_svm_linear,
    cv_scores_svm_rbf
):
    """
    Generate a PDF report:
      - Daily classification probabilities (from RandomForest).
      - Feature importances (RandomForest).
      - Accuracy comparisons.
      - Confusion matrices (as figures).
      - Cross-validation scores.
    """
    timestamp_utc = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    pdf_file_name = f"Market_Pattern_Classification_Report_{timestamp_utc}.pdf"

    # Prepare confusion matrix plots
    # We'll create them using seaborn and save them as PNG, then embed in PDF
    def plot_confusion_matrix(cm, model_name, filename):
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Save confusion matrix plots
    plot_confusion_matrix(cm_rf, "Random Forest", f"cm_rf_{timestamp_utc}.png")
    plot_confusion_matrix(cm_svm_linear, "SVM (Linear)", f"cm_svm_lin_{timestamp_utc}.png")
    plot_confusion_matrix(cm_svm_rbf, "SVM (RBF)", f"cm_svm_rbf_{timestamp_utc}.png")

    # Feature importance plot
    plt.figure(figsize=(6, 4))
    plt.bar(features, feature_importances)
    plt.title("Feature Importances (RandomForest)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    feature_imp_file = f"feature_importances_{timestamp_utc}.png"
    plt.savefig(feature_imp_file)
    plt.close()

    # Initialize FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Market Pattern Classification Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated (UTC): {timestamp_utc}", ln=1, align='C')
    pdf.ln(5)

    # Section: Accuracy comparisons
    pdf.cell(200, 10, txt="Accuracy Comparison:", ln=1)
    pdf.cell(200, 8, txt=f"RandomForest Accuracy: {accuracy_rf:.4f}", ln=1)
    pdf.cell(200, 8, txt=f"SVM (Linear) Accuracy: {accuracy_svm_linear:.4f}", ln=1)
    pdf.cell(200, 8, txt=f"SVM (RBF) Accuracy: {accuracy_svm_rbf:.4f}", ln=1)
    pdf.ln(5)

    # Cross-validation scores
    pdf.cell(200, 10, txt="Cross-Validation Scores (mean +/- std):", ln=1)
    pdf.cell(200, 8, txt=f"RandomForest: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})", ln=1)
    pdf.cell(200, 8, txt=f"SVM (Linear): {cv_scores_svm_linear.mean():.4f} (+/- {cv_scores_svm_linear.std():.4f})", ln=1)
    pdf.cell(200, 8, txt=f"SVM (RBF): {cv_scores_svm_rbf.mean():.4f} (+/- {cv_scores_svm_rbf.std():.4f})", ln=1)
    pdf.ln(5)

    # Classification probabilities (show last few rows for RandomForest)
    pdf.cell(200, 10, txt="Daily Classification Probabilities (RandomForest):", ln=1)
    last_few = df_probs_rf.tail(10)  # e.g. last 10 rows
    for idx, row in last_few.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        prob_str = f"{row['Prob(1)']:.3f}"
        pdf.cell(200, 8, txt=f"{date_str} -> {prob_str}", ln=1)
    pdf.ln(5)

    # Feature importances
    pdf.cell(200, 10, txt="Feature Importances (RandomForest):", ln=1)
    for feat, imp in zip(features, feature_importances):
        pdf.cell(200, 8, txt=f"{feat}: {imp:.4f}", ln=1)
    pdf.ln(5)

    # Add feature importance plot
    pdf.cell(200, 8, txt="Feature Importance Plot:", ln=1)
    pdf.image(feature_imp_file, x=10, y=None, w=180)
    pdf.add_page()

    # Embed the confusion matrix images
    pdf.cell(200, 10, txt="Confusion Matrices:", ln=1)
    pdf.ln(5)
    pdf.cell(200, 8, txt="Random Forest:", ln=1)
    pdf.image(f"cm_rf_{timestamp_utc}.png", x=10, y=None, w=100)
    pdf.ln(60)

    pdf.cell(200, 8, txt="SVM (Linear):", ln=1)
    pdf.image(f"cm_svm_lin_{timestamp_utc}.png", x=10, y=None, w=100)
    pdf.ln(60)

    pdf.cell(200, 8, txt="SVM (RBF):", ln=1)
    pdf.image(f"cm_svm_rbf_{timestamp_utc}.png", x=10, y=None, w=100)

    # Output PDF
    pdf.output(pdf_file_name)
    print(f"PDF report saved as {pdf_file_name}")

def main():
    # --------------------
    # 1. Download S&P 500 data
    # --------------------
    start_date = "2018-01-01"
    end_date = "2025-01-01"
    ticker = "^GSPC"

    df_full = yf.download(ticker, start=start_date, end=end_date)
    # Only keep the 'Close' for demonstration (and rename if needed)
    df = df_full[['Close']].copy()

    # Convert to business-day frequency and forward-fill
    df = df.asfreq('B').ffill()

    # --------------------
    # 2. Compute indicators
    # --------------------
    df['RSI'] = compute_rsi(df['Close'])
    df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['Close'])
    df['BB_Width'] = df['Upper_BB'] - df['Lower_BB']
    df['MA_Crossover'] = compute_moving_average_crossovers(df['Close'])

    # --------------------
    # 3. Label for trend reversals
    # --------------------
    df = label_market_condition_for_reversal(df, lookahead_days=5)

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
    # 6. Models Setup
    # --------------------
    # 6a. RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 6b. SVM (Linear)
    svm_linear = SVC(kernel='linear', probability=True, random_state=42)
    
    # 6c. SVM (RBF)
    svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)

    # --------------------
    # 7. Cross-validation (optional but recommended)
    # --------------------
    # For demonstration, we do a quick CV on the training set (though a better approach might do walk-forward).
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5)
    cv_scores_svm_linear = cross_val_score(svm_linear, X_train, y_train, cv=5)
    cv_scores_svm_rbf = cross_val_score(svm_rbf, X_train, y_train, cv=5)

    # --------------------
    # 8. Train all models
    # --------------------
    rf.fit(X_train, y_train)
    svm_linear.fit(X_train, y_train)
    svm_rbf.fit(X_train, y_train)

    # --------------------
    # 9. Predictions & Evaluations
    # --------------------
    # RandomForest
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("=== RandomForest ===")
    print("Accuracy:", accuracy_rf)
    print("Confusion Matrix:\n", cm_rf)
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))

    # SVM (Linear)
    y_pred_svm_linear = svm_linear.predict(X_test)
    accuracy_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
    cm_svm_linear = confusion_matrix(y_test, y_pred_svm_linear)
    print("\n=== SVM (Linear) ===")
    print("Accuracy:", accuracy_svm_linear)
    print("Confusion Matrix:\n", cm_svm_linear)
    print("Classification Report:\n", classification_report(y_test, y_pred_svm_linear))

    # SVM (RBF)
    y_pred_svm_rbf = svm_rbf.predict(X_test)
    accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
    cm_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)
    print("\n=== SVM (RBF) ===")
    print("Accuracy:", accuracy_svm_rbf)
    print("Confusion Matrix:\n", cm_svm_rbf)
    print("Classification Report:\n", classification_report(y_test, y_pred_svm_rbf))

    # --------------------
    # 10. Prepare daily probabilities for RandomForest
    #     (SVM can produce prob if probability=True, but let's keep it simple 
    #      and just store RF probs for the PDF.)
    # --------------------
    all_probs_rf = rf.predict_proba(X)  # shape (N, 2)
    df_probs_rf = pd.DataFrame(
        index=df.index,
        data={
            "Prob(0)": all_probs_rf[:, 0],
            "Prob(1)": all_probs_rf[:, 1],
        }
    )

    # --------------------
    # 11. Feature importance (RF only)
    # --------------------
    importances = rf.feature_importances_

    # --------------------
    # 12. Generate PDF report
    # --------------------
    generate_pdf_report(
        df_probs_rf=df_probs_rf,
        feature_importances=importances,
        features=features,
        accuracy_rf=accuracy_rf,
        accuracy_svm_linear=accuracy_svm_linear,
        accuracy_svm_rbf=accuracy_svm_rbf,
        cm_rf=cm_rf,
        cm_svm_linear=cm_svm_linear,
        cm_svm_rbf=cm_svm_rbf,
        cv_scores_rf=cv_scores_rf,
        cv_scores_svm_linear=cv_scores_svm_linear,
        cv_scores_svm_rbf=cv_scores_svm_rbf
    )

if __name__ == "__main__":
    main()
