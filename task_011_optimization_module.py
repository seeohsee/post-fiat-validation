import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone
from itertools import product

# Suppose you have your three modules available in the same project:
import market_gravity
import harmonic_oscillator
import quantum_tunneling

from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
# UTILITY FUNCTIONS FOR TIMESTAMPING AND PRINTING
###############################################################################
def utc_timestamp_str():
    """
    Return a UTC timestamp string, e.g., "20250109_123456Z".
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

def log_print(msg):
    """
    Print a message prefixed with a UTC timestamp for progress logging.
    """
    print(f"[{utc_timestamp_str()}] {msg}")


###############################################################################
# 1. DEFINE MARKET REGIMES
###############################################################################
def define_market_regimes():
    """
    Returns a list of tuples defining time periods (start_date, end_date, label)
    for different 'market regimes'.
    For example:
     - 2008 Crash (2008-01-01 to 2009-03-31)
     - 2020 COVID (2020-02-01 to 2020-06-30)
    etc.

    Adjust these to match your actual data ranges and regime definitions.
    """
    regimes = [
        (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-03-31"), "2008_Crash"),
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30"), "2020_COVID"),
        # Add more regimes as desired...
    ]
    return regimes


###############################################################################
# 2. PARAMETER SEARCH SPACE
###############################################################################
def generate_parameter_combinations():
    """
    Example parameter search for each indicator. 
    In real usage, these might be read from a config file, or systematically expanded.

    - For market gravity, maybe we vary 'min_mass_threshold'.
    - For harmonic oscillator, we vary the moving average window, or damping factors.
    - For quantum tunneling, we vary the barrier_strength, energy_threshold, etc.

    This function returns a list of parameter dictionaries.
    """
    gravity_min_mass_threshold_values = [1e7, 5e7, 1e8]         # example
    oscillator_ma_window_values       = [3, 5, 10]              # example
    tunneling_barrier_strength_values = [0.02, 0.05, 0.1]       # example

    # Cartesian product of the parameter sets
    param_combos = []
    for g_thres, ma_win, bar_str in product(
        gravity_min_mass_threshold_values,
        oscillator_ma_window_values,
        tunneling_barrier_strength_values
    ):
        combo = {
            "gravity": {
                "min_mass_threshold": g_thres
            },
            "oscillator": {
                "moving_average_window": ma_win
            },
            "tunneling": {
                "barrier_strength": bar_str
            }
        }
        param_combos.append(combo)

    return param_combos


###############################################################################
# 3. MODEL EVALUATION FUNCTION
###############################################################################
def evaluate_model_on_period(df, params):
    """
    Given a subset of your historical price data `df` for a particular
    regime or walk-forward window, plus a dictionary of model parameters,
    this function should:
      1. Run the Market Gravity analysis with the specified param overrides
         (like min_mass_threshold).
      2. Run the Harmonic Oscillator computations (like MA window).
      3. Run the Quantum Tunneling with updated config (like barrier_strength).
      4. Evaluate performance metrics (e.g., final PnL, max drawdown, etc.)
    
    Returns a dictionary with:
        {
           "performance": <some numeric value or dict>,
           "drawdown": <some numeric>,
           "other_metrics": {...}
        }
    
    This is a placeholder. You’ll want to integrate your actual backtest logic:
      - Generate signals from the 3 modules
      - “Trade” them in a backtest framework
      - Compute the performance stats
    """
    # 1. Market Gravity => override min_mass_threshold
    local_config = market_gravity.load_config(market_gravity.CONFIG_PATH)
    local_config["min_mass_threshold"] = params["gravity"]["min_mass_threshold"]

    # Because we might not want to fetch from Yahoo every time, you could either:
    #    a) pass in pre-fetched data
    #    b) reduce calls to fetch_market_cap_for_ticker (cached)
    # For simplicity, we assume `df` is the subset we want for gravity:
    # So we do an approximate step:
    #    - Assign approximate market caps
    df_grav = df.copy()
    df_grav = market_gravity.assign_approx_market_cap(df_grav)
    df_sector_masses = market_gravity.calculate_sector_masses(df_grav)
    df_sector_forces = market_gravity.calculate_sector_forces(df_sector_masses, local_config)
    df_orbital = market_gravity.identify_orbital_patterns(df_sector_forces)

    # 2. Harmonic Oscillator => override moving_average_window
    df_osc = df.copy()
    df_osc.sort_values(["Ticker", "Date"], inplace=True)
    df_osc = harmonic_oscillator.compute_moving_average(df_osc, window=params["oscillator"]["moving_average_window"])
    df_osc = harmonic_oscillator.compute_displacement(df_osc)
    df_osc = harmonic_oscillator.compute_velocity(df_osc)

    # 3. Quantum Tunneling => override barrier_strength
    local_tunnel_config = dict(quantum_tunneling.CONFIG)
    local_tunnel_config["barrier_strength"] = params["tunneling"]["barrier_strength"]
    sector_plot_data = quantum_tunneling.process_data(df.copy(), local_tunnel_config)
    # sector_plot_data => { sector: DataFrame with [Date, TunnelingProbability] }
    
    # 4. Evaluate performance:
    # For demonstration, we’ll do a dummy performance measure.
    # In practice, you'd generate signals from df_orbital, df_osc, sector_plot_data,
    # run trades, compute equity curve, PnL, drawdowns, etc.
    performance_mock = np.random.uniform(0.0, 1.0)  # placeholder
    drawdown_mock = np.random.uniform(0.0, 0.5)     # placeholder

    # Return metrics:
    return {
        "performance": performance_mock,
        "drawdown": drawdown_mock,
        "orbital_df": df_orbital,    # might be used later for debugging or Monte Carlo
        "oscillator_df": df_osc,
        "tunneling_data": sector_plot_data
    }


###############################################################################
# 4. MONTE CARLO SIMULATION
###############################################################################
def run_monte_carlo_simulation(performance, drawdown):
    """
    Example Monte Carlo simulation approach:
      - We take the performance/drawdown values and randomly sample
        small variations, or use a synthetic approach to create distributions.
      - Return distribution stats (expected drawdown, profit, etc.).

    In practice, you'd run your trading strategy across random subsets or
    bootstrapped samples of data or trades.
    """
    # For demonstration, we generate random draws around the given performance/drawdown.
    n_sims = 1000
    # sample performance ~ Normal(mean=performance, sd=0.1)
    simulated_perfs = np.random.normal(loc=performance, scale=0.1, size=n_sims)
    # sample drawdown ~ Normal(mean=drawdown, sd=0.05)
    simulated_draws = np.random.normal(loc=drawdown, scale=0.05, size=n_sims)

    # Clip or bound them if needed
    simulated_perfs = np.clip(simulated_perfs, 0, 2.0)
    simulated_draws = np.clip(simulated_draws, 0, 1.0)

    mc_results = {
        "simulated_performance": simulated_perfs,
        "simulated_drawdown": simulated_draws,
        "expected_performance": np.mean(simulated_perfs),
        "expected_drawdown": np.mean(simulated_draws),
        "performance_std": np.std(simulated_perfs),
        "drawdown_std": np.std(simulated_draws),
    }
    return mc_results


###############################################################################
# 5. WALK-FORWARD & REGIME ANALYSIS
###############################################################################
def walk_forward_analysis(df, regimes, param_combos):
    """
    1) For each regime/time window,
    2) Evaluate each parameter set,
    3) Pick the best param set (by some performance metric),
    4) Return regime-specific best param set and associated metrics.

    Alternatively, you can do a rolling walk-forward, but here we do a
    simpler 'by-regime' approach.
    """
    regime_results = []
    for start_date, end_date, label in regimes:
        log_print(f"Analyzing regime {label} ({start_date.date()} to {end_date.date()})")

        # Subset data to this regime
        df_regime = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        if df_regime.empty:
            log_print(f"No data for regime {label}, skipping.")
            continue

        best_perf = -999
        best_combo = None
        best_metrics = None

        # Evaluate each parameter combo
        for i, combo in enumerate(param_combos):
            log_print(f" Regime={label} - Evaluating param set {i+1}/{len(param_combos)} ...")
            result = evaluate_model_on_period(df_regime, combo)
            performance = result["performance"]
            drawdown = result["drawdown"]

            if performance > best_perf:
                best_perf = performance
                best_combo = combo
                best_metrics = result

        # Once best param set is identified for this regime, run Monte Carlo
        # on the best metrics
        if best_metrics:
            mc_results = run_monte_carlo_simulation(
                best_metrics["performance"],
                best_metrics["drawdown"]
            )
            regime_results.append({
                "regime_label": label,
                "start_date": start_date,
                "end_date": end_date,
                "best_combo": best_combo,
                "best_performance": best_metrics["performance"],
                "best_drawdown": best_metrics["drawdown"],
                "monte_carlo": mc_results,
            })

    return regime_results


###############################################################################
# 6. REPORTING
###############################################################################
def generate_report(regime_results):
    """
    Create a PDF or textual report summarizing the best param sets per regime,
    their performance, and Monte Carlo distributions.
    """
    utc_now_str = utc_timestamp_str()
    pdf_filename = f"walkforward_report_{utc_now_str}.pdf"
    log_print(f"Generating PDF report {pdf_filename}")

    with PdfPages(pdf_filename) as pdf:
        # Page 1: Summary Table
        fig_summary = plt.figure(figsize=(8, 6))
        ax_summary = fig_summary.add_subplot(111)
        ax_summary.set_title("Regime-Specific Parameter Optimization Results")

        table_text = []
        for item in regime_results:
            label = item["regime_label"]
            perf = item["best_performance"]
            dd = item["best_drawdown"]
            combo = item["best_combo"]
            mc = item["monte_carlo"]

            table_text.append(
                f"Regime: {label}\n"
                f"  Dates: {item['start_date'].strftime('%Y-%m-%d')} to {item['end_date'].strftime('%Y-%m-%d')}\n"
                f"  Best Performance: {perf:.3f}\n"
                f"  Best Drawdown: {dd:.3f}\n"
                f"  Params: Gravity thres={combo['gravity']['min_mass_threshold']}, "
                f"Osc MA window={combo['oscillator']['moving_average_window']}, "
                f"Tunnel barrier={combo['tunneling']['barrier_strength']}\n"
                f"  Monte Carlo - expected perf={mc['expected_performance']:.3f}, "
                f"expected dd={mc['expected_drawdown']:.3f}\n"
                "----------------------------------------\n"
            )
        summary_str = "\n".join(table_text)

        ax_summary.text(0.01, 0.95, summary_str, fontsize=9, va="top", ha="left", wrap=True)
        ax_summary.axis("off")
        plt.tight_layout()
        pdf.savefig(fig_summary)

        # Page 2+: Optional: Monte Carlo distributions plotted
        for item in regime_results:
            fig_mc = plt.figure(figsize=(8, 6))
            ax1 = fig_mc.add_subplot(211)
            ax2 = fig_mc.add_subplot(212)
            
            mc_perf = item["monte_carlo"]["simulated_performance"]
            mc_dd = item["monte_carlo"]["simulated_drawdown"]

            ax1.hist(mc_perf, bins=30, alpha=0.7, color='blue')
            ax1.set_title(f"Monte Carlo Perf Distribution - {item['regime_label']}")
            ax1.set_xlabel("Performance")
            ax1.set_ylabel("Frequency")

            ax2.hist(mc_dd, bins=30, alpha=0.7, color='red')
            ax2.set_title(f"Monte Carlo Drawdown Distribution - {item['regime_label']}")
            ax2.set_xlabel("Drawdown")
            ax2.set_ylabel("Frequency")

            plt.tight_layout()
            pdf.savefig(fig_mc)

    plt.close('all')
    log_print(f"Report saved to {pdf_filename}")

    return pdf_filename


###############################################################################
# 7. MAIN SCRIPT
###############################################################################
def main():
    log_print("Starting Physics Model Optimization (Walk-Forward)")

    # A) Load full historical data
    # We assume pft_price_history.csv is your main dataset, with columns
    # [Date, Adjusted Close, Ticker, Sector], etc.
    csv_path = "pft_price_history.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values(by=["Date", "Ticker"], inplace=True)

    # B) Define market regimes
    regimes = define_market_regimes()

    # C) Generate parameter combos
    param_combos = generate_parameter_combinations()
    log_print(f"Total param combos to evaluate: {len(param_combos)}")

    # D) Run walk-forward or regime-based optimization
    regime_results = walk_forward_analysis(df, regimes, param_combos)

    # E) Generate a final PDF/summary of results
    if regime_results:
        report_file = generate_report(regime_results)
        log_print(f"Walk-forward analysis complete. See report: {report_file}")
    else:
        log_print("No regime results to report (no data or no matches?).")

    log_print("Physics Model Optimization finished.")


if __name__ == "__main__":
    main()
