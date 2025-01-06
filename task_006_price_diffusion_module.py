import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class DiffusionAnalysisConfig:
    """
    Configuration parameters for the diffusion analysis.
    
    Attributes:
        time_scale: int
            Number of trading days to consider as one time-step (e.g., 1 for daily, 5 for weekly).
        diffusion_threshold: float
            Threshold for filtering negligible returns (e.g., 1e-6).
        output_path: str
            Base output path for the PDF report. We'll append a timestamp to it automatically.
        start_date: str
            (Optional) Start date in 'YYYY-MM-DD' format to filter data.
        end_date: str
            (Optional) End date in 'YYYY-MM-DD' format to filter data.
    """
    def __init__(
        self, 
        time_scale=1, 
        diffusion_threshold=1e-6,
        output_path="output/sector_diffusion_report.pdf",
        start_date=None,
        end_date=None
    ):
        self.time_scale = time_scale
        self.diffusion_threshold = diffusion_threshold
        self.output_path = output_path
        self.start_date = start_date
        self.end_date = end_date

class PriceDiffusionAnalyzer:
    """
    Class to perform price diffusion analysis on stock data using 
    Brownian motion principles.
    """
    def __init__(self, config: DiffusionAnalysisConfig):
        self.config = config

    def load_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load and preprocess the price data from a CSV file.
        Ensures the 'Date' column is datetime and sorts by date.
        """
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        
        # Optional date filtering
        if self.config.start_date:
            df = df[df['Date'] >= self.config.start_date]
        if self.config.end_date:
            df = df[df['Date'] <= self.config.end_date]

        df.reset_index(drop=True, inplace=True)
        return df

    def compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns for each Ticker. 
        Groups by Ticker to avoid cross-sectional mixing.
        """
        df = df.groupby('Ticker', group_keys=False).apply(
            lambda x: x.assign(
                log_return=np.log(x['Adjusted Close'] / x['Adjusted Close'].shift(1))
            )
        )
        # Drop the first NaN log_return in each group
        df.dropna(subset=['log_return'], inplace=True)
        return df

    def apply_time_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate returns according to the specified time scale 
        (e.g., for weekly returns if time_scale=5).
        This is a simplistic approach: we just sum the log returns
        over each block of `time_scale` days, per Ticker.
        """
        if self.config.time_scale <= 1:
            # Daily data, no aggregation
            return df

        def aggregate_log_returns(group):
            group = group.sort_values(by='Date')
            idx = np.arange(len(group))
            group['block'] = idx // self.config.time_scale
            grouped = group.groupby('block', as_index=False).agg({
                'Date': 'last',
                'Adjusted Close': 'last',
                'log_return': 'sum',
                'Sector': 'first',
                'Ticker': 'first'
            })
            return grouped

        df_agg = df.groupby('Ticker', group_keys=False).apply(aggregate_log_returns)
        df_agg.drop(columns='block', inplace=True, errors='ignore')
        return df_agg

    def filter_small_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out negligible log returns based on the specified diffusion threshold.
        """
        threshold = self.config.diffusion_threshold
        df = df[np.abs(df['log_return']) >= threshold].copy()
        return df

    def compute_diffusion_coefficients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sector-wise diffusion coefficients. 
        A simple approach is to take the standard deviation of the log returns 
        within each sector as the diffusion coefficient.
        """
        diffusion_df = df.groupby('Sector')['log_return'].std().reset_index()
        diffusion_df.rename(columns={'log_return': 'diffusion_coefficient'}, inplace=True)
        return diffusion_df

    def compute_market_temperature(self, diffusion_df: pd.DataFrame) -> float:
        """
        Compute an overall “market temperature” from the sector diffusion coefficients.
        One possible measure is the average of the diffusion coefficients.
        """
        return diffusion_df['diffusion_coefficient'].mean()

    def generate_pdf_report(self, diffusion_df: pd.DataFrame, df: pd.DataFrame, output_path: str):
        """
        Generate a PDF report with:
            - Bar chart of diffusion coefficients per sector
            - Histograms of log returns per sector
        Saves the PDF to the specified output path.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with PdfPages(output_path) as pdf:
            # 1) Diffusion Coefficients Bar Chart
            plt.figure(figsize=(8, 5))
            plt.bar(diffusion_df['Sector'], diffusion_df['diffusion_coefficient'], color='blue', alpha=0.7)
            plt.xlabel('Sector')
            plt.ylabel('Diffusion Coefficient (Std of Log Returns)')
            plt.title('Sector-wise Diffusion Coefficients')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # 2) Histograms of Log Returns per Sector
            for sector in diffusion_df['Sector'].unique():
                sector_df = df[df['Sector'] == sector]
                if sector_df.empty:
                    continue
                plt.figure(figsize=(8, 5))
                plt.hist(sector_df['log_return'], bins=50, alpha=0.7, color='green')
                plt.title(f'Log Return Distribution - {sector}')
                plt.xlabel('Log Return')
                plt.ylabel('Frequency')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def run_analysis(self, csv_file: str):
        """
        Main entry point to run the entire diffusion analysis.
        """
        # 1. Load data
        df = self.load_data(csv_file)

        # 2. Compute log returns
        df = self.compute_log_returns(df)

        # 3. Apply time scale (aggregating returns)
        df = self.apply_time_scale(df)

        # 4. Filter out negligible returns
        df = self.filter_small_returns(df)

        # 5. Compute diffusion coefficients
        diffusion_df = self.compute_diffusion_coefficients(df)

        # 6. Compute market temperature
        market_temp = self.compute_market_temperature(diffusion_df)
        print(f"Market Temperature (Average Diffusion): {market_temp:.6f}")

        # 7. Generate timestamped PDF report
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        # Insert UTC timestamp before ".pdf" in the user-defined path
        base_path = self.config.output_path
        if base_path.lower().endswith(".pdf"):
            output_path_with_time = base_path[:-4] + f"_{timestamp}.pdf"
        else:
            # In case the user didn't specify ".pdf" in their config
            output_path_with_time = base_path + f"_{timestamp}.pdf"

        self.generate_pdf_report(diffusion_df, df, output_path_with_time)

        return df, diffusion_df, market_temp

def main():
    # Example usage:
    config = DiffusionAnalysisConfig(
        time_scale=1, 
        diffusion_threshold=1e-6,
        output_path="output/sector_diffusion_report.pdf", # output path
        start_date="2024-11-06", # e.g. "2010-01-01"
        end_date=None            # e.g. "2025-01-06"
    )
    analyzer = PriceDiffusionAnalyzer(config)
    
    csv_file = "pft_price_history.csv"  # Path to your CSV file
    df, diffusion_df, market_temp = analyzer.run_analysis(csv_file)

if __name__ == "__main__":
    main()
