import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timezone

def detect_market_regimes(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 20,
    corr_window: int = 20
) -> pd.DataFrame:
    """
    Detect market regimes using rolling volatility ratios and sector correlation changes.
    Returns a DataFrame containing:
        VolRatio (short vs. long vol ratio),
        AvgSectorCorr (average pairwise correlation),
        regime_probability (0 to 1).
    """

    # ---------------------------------------------------------------------
    # 1) Pre-processing
    # ---------------------------------------------------------------------
    # Convert UpdatedAt to datetime if it’s not already
    df["UpdatedAt"] = pd.to_datetime(df["UpdatedAt"])

    # Sort so that we can do proper rolling calculations
    df = df.sort_values(["UpdatedAt", "Ticker"])

    # ---------------------------------------------------------------------
    # 2) Create daily returns proxy
    # ---------------------------------------------------------------------
    # Replace fillna(method="ffill") with ffill():
    # We'll define a "PseudoPrice" from ZScore1 for demonstration only
    df["PseudoPrice"] = df["ZScore1"].ffill() + 50

    # Group by Ticker, then compute daily returns (pseudo-returns)
    df["Returns"] = df.groupby("Ticker")["PseudoPrice"].pct_change()

    # If needed, drop rows with NaN returns
    df.dropna(subset=["Returns"], inplace=True)

    # ---------------------------------------------------------------------
    # 3) Compute rolling volatility by Sector
    # ---------------------------------------------------------------------
    # We'll group by date & sector, then compute daily mean returns across tickers in that sector
    daily_sector_returns = (
        df.groupby(["UpdatedAt", "Sector"])["Returns"]
          .mean()
          .reset_index(name="SectorReturn")
    )

    # Pivot so each Sector becomes a column: wide format for correlation computations
    pivoted = daily_sector_returns.pivot(index="UpdatedAt", columns="Sector", values="SectorReturn")
    pivoted.sort_index(inplace=True)

    # Rolling volatility for each Sector (short-window and long-window)
    rolling_vol_short = pivoted.rolling(window=short_window).std()
    rolling_vol_long  = pivoted.rolling(window=long_window).std()

    # ---------------------------------------------------------------------
    # 4) Compute ratio of short-term vol to long-term vol
    # ---------------------------------------------------------------------
    vol_ratio = rolling_vol_short / rolling_vol_long

    # We'll define a single "market vol ratio" as the average across all sectors
    market_vol_ratio = vol_ratio.mean(axis=1)

    # ---------------------------------------------------------------------
    # 5) Compute rolling correlation across Sectors (if >1 column)
    # ---------------------------------------------------------------------
    if pivoted.shape[1] > 1:
        # We have multiple sectors, so a correlation matrix makes sense
        rolling_corr = pivoted.rolling(window=corr_window).corr()

        # Now rolling_corr typically has a MultiIndex: (date, sector) for rows, sector for columns
        # We'll group by the date level and take the mean across all pairs to get a single number per date
        def avg_correlation_across_sectors(corr_matrix: pd.DataFrame) -> float:
            # Remove diagonal (self-correlation = 1.0)
            m = corr_matrix.values
            # Flatten upper triangle only
            mask = np.triu_indices_from(m, k=1)
            return np.nanmean(m[mask])

        avg_corr_by_date = []
        if isinstance(rolling_corr.index, pd.MultiIndex):
            # Group by the first level (UpdatedAt) and compute average correlation
            for dt, subdf in rolling_corr.groupby(level=0):
                # subdf is a correlation matrix among Sectors
                # "sector" is both the row and column index
                # Reset the index so that we have a 2D matrix
                subdf_2d = subdf.droplevel(level=0)
                # We must ensure it's square if we have multiple columns
                if not subdf_2d.empty:
                    avg_corr_by_date.append(
                        (dt, avg_correlation_across_sectors(subdf_2d))
                    )
                else:
                    avg_corr_by_date.append((dt, np.nan))
        else:
            # If it's not a MultiIndex, it means we couldn't compute correlation properly
            avg_corr_by_date = [(dt, np.nan) for dt in pivoted.index]

        avg_corr_df = pd.DataFrame(avg_corr_by_date, columns=["UpdatedAt", "AvgSectorCorr"])
        avg_corr_df.set_index("UpdatedAt", inplace=True)

    else:
        # If there's only 1 or 0 sectors, we can't compute correlation
        # Just fill with NaN
        avg_corr_df = pd.DataFrame(index=pivoted.index, data={"AvgSectorCorr": np.nan})

    # ---------------------------------------------------------------------
    # 6) Combine signals to produce a "regime probability"
    # ---------------------------------------------------------------------
    signals = pd.concat(
        [market_vol_ratio.rename("VolRatio"), avg_corr_df["AvgSectorCorr"]], axis=1
    )
    signals.dropna(how="all", inplace=True)  # Drop rows where both are NaN

    # If we have no data or partial data, skip further steps
    if len(signals) == 0:
        # Create empty columns just to maintain structure
        signals["VolRatio"] = []
        signals["AvgSectorCorr"] = []
        signals["regime_probability"] = []
        return signals

    # Normalize each signal to [0, 1] for a naive approach
    # (Check if they are constant or NaN)
    def normalize_series(s: pd.Series) -> pd.Series:
        min_val, max_val = s.min(), s.max()
        if min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
            return pd.Series(np.nan, index=s.index)
        return (s - min_val) / (max_val - min_val)

    signals["VolRatioNorm"] = normalize_series(signals["VolRatio"])
    signals["CorrNorm"] = normalize_series(signals["AvgSectorCorr"])

    # For demonstration, define "regime_probability" as the average of VolRatioNorm & CorrNorm
    signals["regime_probability"] = 0.5 * signals["VolRatioNorm"].fillna(0) \
                                  + 0.5 * signals["CorrNorm"].fillna(0)

    return signals[["VolRatio", "AvgSectorCorr", "regime_probability"]]

def create_regime_pdf(
    signals: pd.DataFrame,
    threshold: float = 0.6
):
    """
    Create a PDF visualization of the regime detection signals.
    The PDF file name will include a UTC timestamp using datetime.now(timezone.utc).
    Highlights days when regime_probability > threshold.
    """

    # Check if signals is empty
    if signals.empty:
        print("No signals to plot. Skipping PDF creation.")
        return

    # Generate a UTC timestamp for filename (avoid datetime.utcnow() deprecation)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"market_regime_detection_{timestamp_str}.pdf"

    with PdfPages(pdf_filename) as pdf:
        # 1) Plot VolRatio
        plt.figure(figsize=(12, 6))
        plt.plot(signals.index, signals["VolRatio"], label="Rolling Vol Ratio")
        plt.title("Short-Term / Long-Term Volatility Ratio")
        plt.legend(loc="upper left")
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 2) Plot AvgSectorCorr
        plt.figure(figsize=(12, 6))
        plt.plot(signals.index, signals["AvgSectorCorr"], label="Avg Sector Correlation")
        plt.title("Average Sector Correlation")
        plt.legend(loc="upper left")
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 3) Plot Regime Probability
        plt.figure(figsize=(12, 6))
        plt.plot(signals.index, signals["regime_probability"], label="Regime Probability", color="blue")
        plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
        plt.title("Daily Regime Probability")
        plt.legend(loc="upper left")
        plt.grid(True)

        # Mark areas > threshold
        above_threshold = signals["regime_probability"] > threshold
        # Fill
        plt.fill_between(
            signals.index,
            0,
            1,
            where=above_threshold,
            color='red',
            alpha=0.1,
            transform=plt.gca().get_xaxis_transform()
        )
        pdf.savefig()
        plt.close()

    print(f"PDF with regime warnings saved to: {pdf_filename}")

if __name__ == "__main__":
    df = pd.read_csv("data.csv") # Load the data from disk

    signals = detect_market_regimes(df)
    print(signals.tail())

    # Create PDF if we have some data
    create_regime_pdf(signals, threshold=0.6)
