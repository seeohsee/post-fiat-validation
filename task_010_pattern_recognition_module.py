import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# -------------------------------------------------------------------
# 1. Import the existing modules
# -------------------------------------------------------------------
import market_gravity
import harmonic_oscillator
import quantum_tunneling

# -------------------------------------------------------------------
# 2. Additional Helper Functions
# -------------------------------------------------------------------
def calculate_high_gravity_flag(orbital_df, current_date, threshold='75%'):
    """
    Example function:
    - We interpret 'high gravitational pull' as the day’s average Force 
      (from the Market Gravity orbital_df) being in the top X percentile 
      across all days.
    - `orbital_df` has columns [Date, AverageForce].
    - For demonstration, we default to the 75th percentile.

    Returns True/False for high gravitational pull on `current_date`.
    """
    # Compute the numeric threshold:
    percentile_value = orbital_df["AverageForce"].quantile(float(threshold.strip('%')) / 100)
    
    row_for_date = orbital_df[orbital_df["Date"] == current_date]
    if row_for_date.empty:
        return False
    day_force = row_for_date["AverageForce"].iloc[0]
    return day_force >= percentile_value


def calculate_oscillator_compression_flag(df, ticker, current_date, window=5, std_threshold=0.5):
    """
    Example definition:
    - 'Oscillator compression' if the std dev of Displacement over last X days 
      is below a certain threshold.
    - Could also consider velocity or other oscillator signals.

    df: the DataFrame from harmonic oscillator with columns:
        [Date, Ticker, Displacement, Velocity, etc.]
    ticker: the ticker symbol
    current_date: the date in question
    window: number of days to look back
    std_threshold: threshold for "compression"

    Returns True/False.
    """
    # Filter for the given ticker and up to current_date
    sub = df[(df["Ticker"] == ticker) & (df["Date"] <= current_date)].copy()
    sub.sort_values("Date", inplace=True)

    # Last X days
    sub = sub.tail(window)
    if sub.empty:
        return False

    disp_std = sub["Displacement"].std(skipna=True)
    if pd.isna(disp_std):
        return False

    return disp_std < std_threshold


def calculate_tunneling_flag(tunneling_df_sector, ticker_sector, current_date, prob_threshold=0.8):
    """
    - 'Tunneling probability > prob_threshold' for the ticker's sector on the given date.
    - quantum_tunneling returns a dict: {sector: DataFrame(Date, TunnelingProbability)}
    - We'll look up the TunnelingProbability for this sector and date, 
      then check if it’s above `prob_threshold`.
    """
    if ticker_sector not in tunneling_df_sector:
        return False
    
    sector_data = tunneling_df_sector[ticker_sector]
    row_for_date = sector_data[sector_data["Date"] == current_date]
    if row_for_date.empty:
        return False
    
    prob = row_for_date["TunnelingProbability"].iloc[0]
    return (prob >= prob_threshold)


def generate_entry_exit_signal(df, ticker, current_date, velocity_threshold=0):
    """
    Example entry/exit logic:
    - If velocity > velocity_threshold, we mark "Entry"
    - If velocity < velocity_threshold, we mark "Exit"
    This is very naive. You could integrate more sophisticated signals.
    """
    sub = df[(df["Ticker"] == ticker) & (df["Date"] == current_date)]
    if sub.empty:
        return None

    vel = sub["Velocity"].iloc[0]
    if pd.isna(vel):
        return None
    
    return "Entry" if vel > velocity_threshold else "Exit"


# -------------------------------------------------------------------
# 3. Main Orchestration
# -------------------------------------------------------------------
def main():
    # -------------------------------------------------------------------------
    # A) MARKET GRAVITY ANALYSIS
    # -------------------------------------------------------------------------
    # 1. Load config & price data from market_gravity module
    #    (We assume you have a config file named 'config_market_gravity.txt'
    #     in the same directory, and pft_price_history.csv as described.)
    config = market_gravity.load_config(market_gravity.CONFIG_PATH)
    df_prices = market_gravity.load_price_data(market_gravity.HISTORICAL_CSV)

    # 2. Assign approximate market cap per day
    df_prices = market_gravity.assign_approx_market_cap(df_prices)

    # 3. Compute sector masses and forces
    df_sector_masses = market_gravity.calculate_sector_masses(df_prices)
    df_sector_forces = market_gravity.calculate_sector_forces(df_sector_masses, config)

    # 4. Identify 'orbital' patterns (placeholder), typically a daily average force
    df_orbital = market_gravity.identify_orbital_patterns(df_sector_forces)
    # df_orbital columns: [Date, AverageForce]

    # -------------------------------------------------------------------------
    # B) HARMONIC OSCILLATOR ANALYSIS
    # -------------------------------------------------------------------------
    # For convenience, we’ll re-load the price data in the oscillator module’s format
    df_osc = harmonic_oscillator.load_price_data("pft_price_history.csv")
    # Compute moving average, displacement, velocity
    df_osc = harmonic_oscillator.compute_moving_average(df_osc, window=3)   # example short window
    df_osc = harmonic_oscillator.compute_displacement(df_osc)
    df_osc = harmonic_oscillator.compute_velocity(df_osc)
    # We could also use identify_resonance if desired:
    # sector_params = harmonic_oscillator.compute_natural_frequency(df_osc, config={...})
    # resonance_results = harmonic_oscillator.identify_resonance(df_osc, sector_params, correlation_threshold=0.8)

    # -------------------------------------------------------------------------
    # C) QUANTUM TUNNELING ANALYSIS
    # -------------------------------------------------------------------------
    # Load raw price data again (or reuse df_osc or df_prices) 
    df_tunnel_raw = pd.read_csv("pft_price_history.csv", parse_dates=["Date"])
    # We do the quantum tunneling analysis
    # quantum_tunneling.process_data returns a dict: { sector: DataFrame(Date, TunnelingProbability) }
    sector_plot_data = quantum_tunneling.process_data(df_tunnel_raw, quantum_tunneling.CONFIG)
    # Example structure: sector_plot_data["Technology"] => DataFrame with columns [Date, TunnelingProbability]

    # -------------------------------------------------------------------------
    # D) COMBINE RESULTS INTO A DAILY STOCK FILTER
    # -------------------------------------------------------------------------
    # We'll go day by day, ticker by ticker, check:
    #   1) High gravitational pull? (use df_orbital’s average force)
    #   2) Oscillator compression?   (use df_osc’s displacement std)
    #   3) Tunneling probability > 80%?  (use sector_plot_data)
    #   4) Generate entry/exit signals?

    all_dates = sorted(df_prices["Date"].unique())
    tickers = df_prices["Ticker"].unique()

    signal_rows = []

    for d in all_dates:
        # Check if day in the "orbital" DataFrame
        high_gravity = calculate_high_gravity_flag(df_orbital, d, threshold='75%')  

        for t in tickers:
            # Identify that ticker’s sector
            sub_price = df_prices[(df_prices["Date"] == d) & (df_prices["Ticker"] == t)]
            if sub_price.empty:
                continue
            sector = sub_price["Sector"].iloc[0]

            # (1) High gravitational pull: we already have bool high_gravity
            # (2) Oscillator compression:
            osc_compress = calculate_oscillator_compression_flag(df_osc, t, d, window=5, std_threshold=0.2)
            
            # (3) Tunneling probability:
            tunnel_flag = calculate_tunneling_flag(sector_plot_data, sector, d, prob_threshold=0.8)

            # (4) Entry/Exit signals:
            entry_exit = generate_entry_exit_signal(df_osc, t, d, velocity_threshold=0.0)

            # If all conditions are satisfied, add to our daily signal list
            if high_gravity and osc_compress and tunnel_flag:
                signal_rows.append({
                    "Date": d,
                    "Ticker": t,
                    "Sector": sector,
                    "HighGravity": high_gravity,
                    "OscCompression": osc_compress,
                    "TunnelProb>80%": tunnel_flag,
                    "Signal": entry_exit
                })

    df_signals = pd.DataFrame(signal_rows)
    # This now has a daily list of stocks meeting your triple-condition,
    # plus an entry/exit decision for each.

    # -------------------------------------------------------------------------
    # E) Generate a Summary PDF
    # -------------------------------------------------------------------------
    # In this example, we’ll create a simple multi-panel figure:
    #   - Panel 1: Market Gravity average force over time
    #   - Panel 2: Example Tunneling probability lines by sector
    #   - Then we’ll add a text table of signals onto the figure or on a new page.
    # -------------------------------------------------------------------------

    fig = plt.figure(figsize=(10, 8))

    # Panel 1: Market Gravity average force
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(df_orbital["Date"], df_orbital["AverageForce"], marker="o")
    ax1.set_title("Average Sector Gravitational Force Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Average Force")
    ax1.grid(True)

    # Panel 2: Tunneling probabilities (just plot a couple of sectors for demo)
    ax2 = fig.add_subplot(2, 1, 2)
    colors = ["blue", "red", "green", "orange", "purple"]
    for i, (sector_key, sector_data) in enumerate(sector_plot_data.items()):
        ax2.plot(
            sector_data["Date"], 
            sector_data["TunnelingProbability"], 
            label=sector_key, 
            color=colors[i % len(colors)]
        )
    ax2.set_title("Quantum Tunneling Probability by Sector")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Tunneling Probability")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save this main figure to a PDF page
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    pdf_filename = f"HighProbSetups_{utc_now}.pdf"

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)  # Save the figure with the two plots

        # Create a new page for the signals table
        fig_table = plt.figure(figsize=(10, 6))
        ax_table = fig_table.add_subplot(111)
        ax_table.set_title("Daily Signals: High Grav + Osc Compression + Tunneling>80%")

        # If you have many signals, consider just showing the last few or summarize.
        # For demonstration, we’ll show them all in a text table.
        table_text = []
        for idx, row in df_signals.iterrows():
            table_text.append(
                f"{row['Date'].strftime('%Y-%m-%d')}, {row['Ticker']}, {row['Sector']}, "
                f"Signal={row['Signal']}"
            )
        # Join them into one big string
        table_str = "\n".join(table_text)

        ax_table.text(0.01, 0.95, table_str, fontsize=10, va="top", ha="left", wrap=True)
        ax_table.axis("off")  # Hide the actual axes
        plt.tight_layout()
        pdf.savefig(fig_table)
    
    plt.close('all')  # Close all figures
    print(f"[INFO] Analysis complete. PDF saved to: {pdf_filename}")

    # Finally, if you want to also print or write the signals as CSV:
    if not df_signals.empty:
        csv_out = f"HighProbSignals_{utc_now}.csv"
        df_signals.to_csv(csv_out, index=False)
        print(f"[INFO] Signals CSV saved to: {csv_out}")
    else:
        print("[INFO] No signals found for the given criteria.")


if __name__ == "__main__":
    main()
