import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os

def load_price_data(csv_path):
    """
    Load price data from a CSV file.
    Expects columns: [Date, Adjusted Close, Ticker, Sector].
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.sort_values(by=['Ticker', 'Date'], inplace=True)
    return df

def compute_moving_average(df, window=5):
    """
    Compute a simple moving average for the 'Adjusted Close' by Ticker.
    Adds a new column 'MovingAverage' to the DataFrame.
    """
    df['MovingAverage'] = df.groupby('Ticker')['Adjusted Close'].transform(lambda x: x.rolling(window).mean())
    return df

def compute_displacement(df):
    """
    Displacement as difference between Adjusted Close and the moving average.
    """
    df['Displacement'] = df['Adjusted Close'] - df['MovingAverage']
    return df

def compute_velocity(df):
    """
    Velocity (momentum) can be approximated by the discrete difference
    of the 'Displacement'. Another approach could be using indicators like RSI or MACD.
    """
    df['Velocity'] = df.groupby('Ticker')['Displacement'].diff()
    return df

def compute_natural_frequency(df, config):
    """
    Compute natural frequency per sector: omega_0 = sqrt(k / m).
    We’ll just assume m=1 for each sector to simplify.
    Uses a configured spring constant 'k' from config, or a default if missing.
    Returns a dictionary: sector -> (omega_0, damping_factor).
    """
    sectors = df['Sector'].unique()
    sector_params = {}
    for sector in sectors:
        k = config.get('spring_constants', {}).get(sector, 1.0)  # default k=1.0
        damping = config.get('damping_factors', {}).get(sector, 0.1)  # default damping
        omega_0 = np.sqrt(k)  # if m=1
        sector_params[sector] = (omega_0, damping)
    return sector_params

def identify_resonance(df, sector_params, correlation_threshold=0.8):
    """
    Identify potential 'resonance' conditions between assets in the same sector
    if their displacement/velocity time-series are highly correlated.
    For simplicity, we use Pearson correlation on 'Displacement' or 'Velocity'.
    
    Returns a list of tuples (ticker1, ticker2, correlation).
    """
    resonance_pairs = []
    
    # Group data by Ticker for correlation
    tickers = df['Ticker'].unique()
    # Create a dictionary that holds displacement series for each ticker
    displacement_map = {}
    
    for t in tickers:
        sub_df = df[df['Ticker'] == t].dropna(subset=['Displacement'])
        displacement_map[t] = sub_df['Displacement'].values  # or sub_df['Velocity'].values
    
    # Compare every pair of tickers in the same sector
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            s1 = df.loc[df['Ticker'] == t1, 'Sector'].iloc[0]
            s2 = df.loc[df['Ticker'] == t2, 'Sector'].iloc[0]
            
            # Only compare resonance if same sector
            if s1 == s2 and t1 in displacement_map and t2 in displacement_map:
                disp1 = displacement_map[t1]
                disp2 = displacement_map[t2]
                
                # Align lengths (simple approach; real code would handle date alignment properly)
                min_len = min(len(disp1), len(disp2))
                corr = np.corrcoef(disp1[:min_len], disp2[:min_len])[0,1]
                
                if abs(corr) >= correlation_threshold:
                    resonance_pairs.append((t1, t2, corr))
    
    return resonance_pairs

def plot_displacement_and_velocity(df, output_pdf):
    """
    Create a plot of Displacement and Velocity for demonstration.
    Save to output_pdf file.
    """
    # For demonstration, just plot the last Ticker's time series
    last_ticker = df['Ticker'].iloc[-1]
    sub_df = df[df['Ticker'] == last_ticker].copy()
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax[0].plot(sub_df['Date'], sub_df['Displacement'], label='Displacement')
    ax[0].set_title(f'{last_ticker} Displacement')
    ax[0].legend()
    
    ax[1].plot(sub_df['Date'], sub_df['Velocity'], label='Velocity', color='orange')
    ax[1].set_title(f'{last_ticker} Velocity')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close(fig)

def main():
    """
    Main function orchestrating the pipeline:
      1. Load price data
      2. Compute moving average, displacement, velocity
      3. Compute natural frequencies
      4. Identify resonance
      5. Plot and save PDF with UTC timestamp
    """
    # --- Configuration ---
    config = {
        'spring_constants': {
            'Technology': 2.0,
            'Healthcare': 1.5,
            # You can add more sector-specific values here
        },
        'damping_factors': {
            'Technology': 0.05,
            'Healthcare': 0.1,
            # You can add more sector-specific values here
        },
        'moving_average_window': 3,    # For demonstration, short window
        'correlation_threshold': 0.8,
    }
    
    csv_path = 'pft_price_history.csv'
    
    # 1. Load data
    df = load_price_data(csv_path)
    
    # 2. Compute moving average, displacement (x), velocity (dx/dt)
    df = compute_moving_average(df, window=config['moving_average_window'])
    df = compute_displacement(df)
    df = compute_velocity(df)
    
    # 3. Compute natural frequencies for each sector
    sector_params = compute_natural_frequency(df, config)
    print("Sector Parameters (omega_0, damping):", sector_params)
    
    # 4. Identify resonance
    resonance_results = identify_resonance(df, sector_params, correlation_threshold=config['correlation_threshold'])
    print("Resonance pairs (ticker1, ticker2, correlation):", resonance_results)
    
    # 5. Plot and export PDF with UTC timestamp
    now_utc = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    pdf_filename = f"harmosci_{now_utc}.pdf"
    
    plot_displacement_and_velocity(df, pdf_filename)
    print(f"Output PDF saved to: {pdf_filename}")


if __name__ == '__main__':
    main()
