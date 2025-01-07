import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone

###############################################################################
# Configuration parameters for "quantum tunneling" analysis
###############################################################################
CONFIG = {
    "energy_threshold": 0.01,    # Minimal daily momentum needed to consider "penetration"
    "barrier_strength": 0.05,    # Adjusts the effective barrier height
    "lookback_days": 5           # Number of days for calculating near-term resistance
}

def calculate_momentum(prices: pd.Series) -> pd.Series:
    """
    Calculate daily momentum, e.g., a simple difference in adjusted close
    or percent change. Here we use % change as a simple stand-in for 'kinetic energy'.
    
    Because we will use 'transform', we can define a plain function, but we could
    also inline the logic in the transform call. This is optional.
    """
    return prices.pct_change().fillna(0.0)

def calculate_resistance_level(prices: pd.Series, lookback: int) -> pd.Series:
    """
    Calculate a simple 'resistance level' as the rolling max over a lookback window.
    This represents the potential barrier in the quantum analogy.
    """
    return prices.rolling(window=lookback, min_periods=1).max()

def calculate_tunneling_probability(momentum: float, barrier: float, config: dict) -> float:
    """
    Given momentum (E) and barrier (V), return a 'tunneling probability' T(E).
    This is a simplified form. In quantum mechanics, the calculation is more involved.
    
    Example approach:
        T(E) = exp( - (barrier - E) ) if barrier>energy, else ~1
    with some scaling by config["energy_threshold"] and config["barrier_strength"].
    """
    energy_threshold = config["energy_threshold"]  # minimal kinetic energy
    barrier_strength = config["barrier_strength"]
    
    # If momentum < threshold, treat momentum as zero
    # (i.e. no "penetration" if below minimal energy needed).
    effective_energy = max(0, momentum - energy_threshold)
    
    # Effective barrier
    effective_barrier = barrier_strength * barrier
    
    if effective_energy <= 0:
        # Not enough "energy" to attempt barrier penetration
        return 0.0
    
    # A toy formula for tunneling:
    #   T = exp( - (barrier - energy) ) if barrier > energy, else 1
    # We incorporate the config params for demonstration.
    if effective_barrier > effective_energy:
        return np.exp(-(effective_barrier - effective_energy))
    else:
        return 1.0

def process_data(df: pd.DataFrame, config: dict) -> dict:
    """
    Process the data by sector: calculate momentum, resistance, and tunneling
    probability. Return a dictionary of {sector: DataFrame} containing
    daily average tunneling probabilities.
    """
    # Ensure data is sorted by Date (important for rolling windows, etc.)
    df.sort_values(by=["Date"], inplace=True)

    # Group by sector
    sector_groups = df.groupby("Sector")

    # Dictionary to store final data for plotting: { sector: DataFrame of daily T }
    sector_plot_data = {}

    for sector, group in sector_groups:
        # Make a copy so we don't modify the original DataFrame
        group = group.copy()
        
        # Calculate momentum per ticker using transform
        group["Momentum"] = (
            group.groupby("Ticker")["Adjusted Close"]
                 .transform(lambda x: x.pct_change().fillna(0.0))
        )
        
        # Calculate rolling resistance per ticker using transform
        group["Resistance"] = (
            group.groupby("Ticker")["Adjusted Close"]
                 .transform(lambda x: x.rolling(config["lookback_days"], min_periods=1).max())
        )
        
        # Calculate tunneling probability row-by-row
        tunnel_probs = []
        for idx, row in group.iterrows():
            momentum = row["Momentum"]
            barrier = row["Resistance"]
            t_prob = calculate_tunneling_probability(momentum, barrier, config)
            tunnel_probs.append(t_prob)
        
        group["TunnelingProbability"] = tunnel_probs
        
        # Aggregate to a daily average by date if we want a single line per sector
        daily_by_sector = group.groupby("Date")["TunnelingProbability"].mean().reset_index()
        
        # Store for plotting
        sector_plot_data[sector] = daily_by_sector

    return sector_plot_data

def plot_results(sector_plot_data: dict) -> plt.Figure:
    """
    Plot the daily tunneling probability by sector. Return a Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Each sector gets a line on the plot
    for sector, data in sector_plot_data.items():
        ax.plot(data["Date"], data["TunnelingProbability"], label=sector)
    
    ax.set_title("Quantum Tunneling Probability by Sector")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tunneling Probability")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig  # Return the figure for further processing (e.g., saving)

def save_plot_to_pdf(fig: plt.Figure, output_filename: str) -> None:
    """
    Save the given Matplotlib Figure to a PDF file.
    """
    fig.savefig(output_filename)
    plt.close(fig)

def main():
    # -------------------------------------------------------------------------
    # 1. Read CSV Data
    # -------------------------------------------------------------------------
    csv_file = "pft_price_history.csv"  # Replace with the actual path if needed
    df = pd.read_csv(csv_file, parse_dates=["Date"])
    
    # -------------------------------------------------------------------------
    # 2. Process data to get daily average tunneling probabilities per sector
    # -------------------------------------------------------------------------
    sector_plot_data = process_data(df, CONFIG)
    
    # -------------------------------------------------------------------------
    # 3. Plot results (return a Figure)
    # -------------------------------------------------------------------------
    fig = plot_results(sector_plot_data)
    
    # -------------------------------------------------------------------------
    # 4. Save to PDF with UTC timestamp in filename
    # -------------------------------------------------------------------------
    utc_now = datetime.now(timezone.utc)
    timestamp_str = utc_now.strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"tunneling_analysis_{timestamp_str}.pdf"
    
    save_plot_to_pdf(fig, output_filename)
    print(f"Saved Tunneling Analysis to: {output_filename}")

if __name__ == "__main__":
    main()
