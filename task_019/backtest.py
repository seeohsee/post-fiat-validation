import numpy as np
import pandas as pd
import yfinance as yf
from fpdf import FPDF
from datetime import datetime, timedelta, timezone

def compute_max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def compute_sharpe_ratio(series, risk_free_rate=0.0):
    excess_returns = series - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std(ddof=0) if excess_returns.std() > 0 else np.nan

def fetch_close_prices(ticker, period="3y"):
    data = yf.download(ticker, period=period, interval='1d', group_by='ticker', auto_adjust=True, threads=True)
    df = data.xs('Close', level='Price', axis=1)
    df = df.dropna(axis=1, how='any')
    df.columns = ['price']
    return df

def calculate_forward_metrics(ticker, entry_dates, holding_periods=[5, 20, 60]):

    # Fetch the historical price data
    df = fetch_close_prices(ticker)
    df['returns'] = df['price'].pct_change()

    results = []

    for date in entry_dates:
        entry_date = pd.to_datetime(date)
        idx = df.index.searchsorted(entry_date, side='left')
        if idx >= len(df):
            continue  # skip if entry date is beyond available data
        entry_date = df.index[idx]
        for period in holding_periods:
            exit_date = entry_date + timedelta(days=period * 2)
            forward_data = df.loc[entry_date:exit_date].iloc[:period]
            if len(forward_data) < period:
                continue

            price_start = forward_data.iloc[0]['price']
            price_end = forward_data.iloc[-1]['price']
            total_return = (price_end / price_start) - 1

            sharpe = compute_sharpe_ratio(forward_data['returns'])
            max_dd = compute_max_drawdown(forward_data['returns'])

            results.append({
                'entry_date': entry_date.date(),
                'holding_days': period,
                'total_return': round(total_return, 4),
                'sharpe_ratio': round(sharpe, 4),
                'max_drawdown': round(max_dd, 4)
            })

    return pd.DataFrame(results)

def generate_pdf_report(ticker, df_results):
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"forward_metrics_{ticker}_{utc_now}_UTC.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"{ticker} Forward Performance Report", ln=True)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    col_widths = [30, 30, 35, 35, 35]
    headers = ['Entry Date', 'Holding (Days)', 'Total Return', 'Sharpe Ratio', 'Max Drawdown']
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, border=1, align='C')
    pdf.ln()

    pdf.set_font("Helvetica", "", 12)
    for _, row in df_results.iterrows():
        pdf.cell(col_widths[0], 10, str(row['entry_date']), border=1)
        pdf.cell(col_widths[1], 10, str(row['holding_days']), border=1, align='C')
        pdf.cell(col_widths[2], 10, f"{row['total_return']:.4f}", border=1, align='R')
        pdf.cell(col_widths[3], 10, f"{row['sharpe_ratio']:.4f}", border=1, align='R')
        pdf.cell(col_widths[4], 10, f"{row['max_drawdown']:.4f}", border=1, align='R')
        pdf.ln()

    pdf.output(filename)
    print(f"PDF saved as: {filename}")

# === Example Usage ===
if __name__ == "__main__":
    sample_dates = ['2023-01-03', '2023-06-01', '2023-09-15']
    ticker = 'SPY'
    holding_periods = [30]

    df_metrics = calculate_forward_metrics(ticker, sample_dates, holding_periods)
    print(df_metrics)
    generate_pdf_report(ticker, df_metrics)
