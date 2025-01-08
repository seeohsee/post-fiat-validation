import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

###############################################################################
# 1. CONFIGURATION
###############################################################################
CONFIG_PATH = "config_market_gravity.txt"
HISTORICAL_CSV = "pft_price_history.csv"

def load_config(config_path):
    """
    Loads configuration parameters from a simple text file with lines like:
        key=value
    Ignores empty lines and those starting with '#'.
    """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expect a line like: gravitational_constant=6.67408e-11
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Attempt to convert numeric values; if fail, store as string
                try:
                    # If it's numeric, we parse it as float
                    value = float(value)
                except ValueError:
                    pass
                config[key] = value
    return config

###############################################################################
# 2. DATA LOADING
###############################################################################
def load_price_data(csv_path):
    """
    Loads historical price data from a CSV.
    The CSV is expected to have columns: Date, Adjusted Close, Ticker, Sector.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    return df

def fetch_market_cap_for_ticker(ticker):
    """
    Fetches an approximate (current) market cap for `ticker` from Yahoo Finance.
    We'll do so by:
      - Getting the current shares outstanding (if available).
      - We'll return the integer number of shares or None if not found.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info  # dictionary of info
        shares_outstanding = info.get("sharesOutstanding", None)
        # If shares_outstanding is None, we can't compute approximate market cap
        return shares_outstanding
    except Exception as e:
        print(f"Warning: Could not fetch market cap for {ticker}: {e}")
        return None

def assign_approx_market_cap(df):
    """
    For each row in df, compute approximate daily market cap:
        market_cap = Adjusted Close * shares_outstanding
    
    Because yfinance typically does NOT provide historical shares outstanding,
    this uses the current shares outstanding from fetch_market_cap_for_ticker(),
    which is only a naive approximation for demonstration.
    """
    tickers = df["Ticker"].unique()
    ticker_to_shares = {}

    for t in tickers:
        ticker_to_shares[t] = fetch_market_cap_for_ticker(t)

    market_caps = []
    for idx, row in df.iterrows():
        t = row["Ticker"]
        close_price = row["Adjusted Close"]
        shares_outstanding = ticker_to_shares[t]
        if shares_outstanding is not None:
            mc = close_price * shares_outstanding
        else:
            mc = np.nan
        market_caps.append(mc)

    df["MarketCap"] = market_caps
    return df

###############################################################################
# 3. GRAVITATIONAL CALCULATIONS
###############################################################################
def calculate_sector_masses(df):
    """
    Aggregates per-day market cap by sector.
    Returns a DataFrame with columns: Date, Sector, Mass (the sum of MarketCaps in that sector).
    """
    grouped = df.groupby(["Date", "Sector"], as_index=False).agg({"MarketCap": "sum"})
    grouped.rename(columns={"MarketCap": "Mass"}, inplace=True)
    return grouped

def calculate_gravitational_force(m1, m2, distance, G=6.67408e-11):
    """
    Newton's law of gravitation:
        F = G * (m1 * m2) / (r^2)

    In a financial context, 'distance' might be some domain-specific metric.
    """
    if distance == 0:
        return 0
    return G * (m1 * m2) / (distance**2)

def calculate_sector_forces(sector_data, config):
    """
    Calculates the 'gravitational' forces between sectors on each Date.
    We must define 'distance' between two sectors. Example approach:
    - distance = absolute difference in sector indices (very naive).
    - You might instead use correlation distance, price velocity, etc.

    sector_data: DataFrame of [Date, Sector, Mass]
    """
    # Extract config values (we expect them to be floats)
    G = config.get("gravitational_constant", 6.67408e-11)
    min_mass_threshold = config.get("min_mass_threshold", 1e7)

    sector_data.sort_values(by="Date", inplace=True)
    unique_dates = sector_data["Date"].unique()
    unique_sectors = sector_data["Sector"].unique()

    results = []

    for d in unique_dates:
        daily_data = sector_data[sector_data["Date"] == d].copy()
        # Filter by min mass threshold
        daily_data = daily_data[daily_data["Mass"] >= min_mass_threshold]

        # Compare each pair of sectors
        for i, row_i in daily_data.iterrows():
            for j, row_j in daily_data.iterrows():
                if j <= i:
                    continue  # avoid double counting
                m1 = row_i["Mass"]
                m2 = row_j["Mass"]
                # Example: sector distance is just difference in their index
                idx_i = list(unique_sectors).index(row_i["Sector"])
                idx_j = list(unique_sectors).index(row_j["Sector"])
                distance = abs(idx_i - idx_j)

                force = calculate_gravitational_force(m1, m2, distance, G=G)
                results.append({
                    "Date": d,
                    "Sector1": row_i["Sector"],
                    "Sector2": row_j["Sector"],
                    "Force": force
                })
    return pd.DataFrame(results)

###############################################################################
# 4. IDENTIFYING ORBITAL PATTERNS (Placeholder)
###############################################################################
def identify_orbital_patterns(df_forces):
    """
    Placeholder for 'orbital pattern' detection.
    Here, we just compute a daily average force as a simple metric.
    """
    orbital_df = df_forces.groupby("Date")["Force"].mean().reset_index()
    orbital_df.rename(columns={"Force": "AverageForce"}, inplace=True)
    return orbital_df

###############################################################################
# 5. VISUALIZATION AND PDF OUTPUT
###############################################################################
def create_visualization(orbital_df):
    """
    Creates a line plot of average daily force (example),
    then saves to a timestamped PDF using UTC time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(orbital_df["Date"], orbital_df["AverageForce"], marker="o")
    plt.title("Average Sector Gravitational Force Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Force")
    plt.grid(True)

    utc_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_filename = f"gravity_analysis_{utc_timestamp}.pdf"
    plt.savefig(output_filename, format="pdf")
    plt.close()
    print(f"Saved analysis to {output_filename}")

###############################################################################
# 6. MAIN SCRIPT
###############################################################################
def main():
    # Load config from config.txt
    config = load_config(CONFIG_PATH)

    # Load historical price data
    df_prices = load_price_data(HISTORICAL_CSV)

    # Assign approximate market cap for each row
    df_prices = assign_approx_market_cap(df_prices)

    # Optionally compute daily velocity or momentum here if desired...

    # Aggregate by sector to get sector masses
    df_sector_masses = calculate_sector_masses(df_prices)

    # Calculate gravitational forces between sectors (for each Date)
    df_sector_forces = calculate_sector_forces(df_sector_masses, config)

    # Identify orbital patterns (placeholder)
    df_orbital = identify_orbital_patterns(df_sector_forces)

    # Create visualization and output PDF
    create_visualization(df_orbital)

if __name__ == "__main__":
    main()
