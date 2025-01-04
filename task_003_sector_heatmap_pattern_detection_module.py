import pytz
import pandas as pd
import seaborn as sns
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class SectorHeatmap:
    """
    A class to generate heatmaps and detect patterns in sector-based metrics.

    Attributes:
        data_file (str): Path to the CSV file containing the data.
        config_file (str): Path to the configuration file specifying the metric to use.
        data (DataFrame): The loaded data.
        metric (str): The selected metric for analysis.
    """

    def __init__(self, data_file, config_file) -> None:
        """
        Initializes the SectorHeatmap class.

        Args:
            data_file (str): Path to the data file.
            config_file (str): Path to the configuration file.
        """
        self.data        = None
        self.metric      = None
        self.data_file   = data_file
        self.config_file = config_file

    def load_data(self) -> None:
        """
        Loads data from the CSV file and parses the 'UpdatedAt' column as datetime.
        """
        self.data = pd.read_csv(self.data_file)
        self.data['UpdatedAt'] = pd.to_datetime(self.data['UpdatedAt'])

    def load_config(self) -> None:
        """
        Reads the metric to be used from the configuration file.
        """
        with open(self.config_file, 'r') as f:
            self.metric = f.read().strip()

    def filter_data(self, start_date, end_date) -> None:
        """
        Filters the data for the given date range.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
        """
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.data = self.data[(self.data['UpdatedAt'] >= start_date) & (self.data['UpdatedAt'] <= end_date)]

    def create_heatmap(self) -> tuple:
        """
        Creates a heatmap for the selected metric and saves both static and interactive versions.

        Returns:
            tuple: Paths to the generated PDF and HTML files.
        """
        if self.metric not in self.data.columns:
            raise ValueError(f"Metric {self.metric} not found in data columns.")

        sector_means = self.data.groupby('Sector')[self.metric].mean().sort_values()

        # Static Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(sector_means.to_frame().T, annot=True, cmap="YlGnBu", cbar_kws={'label': self.metric})
        plt.title(f"Sector Heatmap ({self.metric})")
        plt.tight_layout()

        # Save Static Heatmap to PDF
        timestamp = datetime.now(pytz.utc).strftime('%Y%m%d_%H%M%S')
        pdf_path = f"sector_heatmap_{self.metric}_{timestamp}.pdf"
        with PdfPages(pdf_path) as pdf:
            pdf.savefig()
        print(f"Static heatmap saved as {pdf_path}")

        # Interactive Heatmap
        interactive_fig = px.imshow(sector_means.to_frame().T,
                                     color_continuous_scale="YlGnBu",
                                     labels={"color": self.metric},
                                     title=f"Sector Heatmap ({self.metric})")

        # Save Interactive Heatmap to disk
        html_path = f"sector_heatmap_{self.metric}_{timestamp}.html"
        interactive_fig.write_html(html_path)
        print(f"Interactive heatmap saved as {html_path}")

        return pdf_path, html_path

    def detect_pattern_deviation(self, lookback_period, threshold):
        """
        Detects deviations in sector correlations and generates visual and statistical outputs.
    
        Args:
            lookback_period (int): Number of periods for the rolling correlation.
            threshold (float): Threshold below which deviations are flagged.
    
        Returns:
            tuple: Paths to the deviation heatmap and summary PDF files.
        """
        if self.metric not in self.data.columns:
            raise ValueError(f"Metric {self.metric} not found in data columns.")
    
        # Compute rolling correlations
        pivot_data = self.data.pivot_table(index='UpdatedAt', columns='Sector', values=self.metric)
        print("Pivot data preview:", pivot_data.head())  # Debugging
    
        # Fill missing values (optional)
        pivot_data = pivot_data.fillna(method='ffill').fillna(method='bfill')
        print("Filled pivot data preview:", pivot_data.head())  # Debugging
    
        # Check for sufficient data
        if pivot_data.shape[0] < lookback_period:
            print(f"Warning: Insufficient data for lookback period ({lookback_period}). Adjusting to available rows.")
            lookback_period = pivot_data.shape[0]
    
        correlation_matrix = pivot_data.rolling(lookback_period).corr()
        print("Correlation matrix preview:", correlation_matrix.head())  # Debugging
    
        # Filter valid correlation matrix values
        correlation_matrix = correlation_matrix.dropna()
        if correlation_matrix.empty:
            raise ValueError("Correlation matrix is empty after dropping NaNs. Check data completeness or reduce lookback period.")
    
        # Identify deviations
        deviations = (correlation_matrix < threshold).groupby(level=0).sum()
        print("Deviations preview:", deviations.head())  # Debugging
    
        if deviations.empty:
            raise ValueError("No deviations detected; deviations DataFrame is empty.")
    
        # Visual Alert in Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(deviations.T, annot=True, cmap="Reds", cbar_kws={'label': 'Deviations'})
        plt.title(f"Sector Correlation Deviations ({self.metric})")
        plt.tight_layout()
    
        timestamp = datetime.now(pytz.utc).strftime('%Y%m%d_%H%M%S')
        deviation_pdf_path = f"sector_deviation_{self.metric}_{timestamp}.pdf"
        with PdfPages(deviation_pdf_path) as pdf:
            pdf.savefig()
        print(f"Deviation heatmap saved as {deviation_pdf_path}")
    
        # Statistical Summary
        deviation_summary = deviations.describe()
        summary_pdf_path = f"deviation_summary_{self.metric}_{timestamp}.pdf"
        with PdfPages(summary_pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('tight')
            ax.axis('off')
            table = plt.table(cellText=deviation_summary.values,
                              colLabels=deviation_summary.columns,
                              rowLabels=deviation_summary.index,
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.auto_set_column_width(col=list(range(len(deviation_summary.columns))))
            pdf.savefig()
        print(f"Deviation summary saved as {summary_pdf_path}")
    
        return deviation_pdf_path, summary_pdf_path

if __name__ == "__main__":
    data_file   = "data.csv"   # Path to the CSV data
    config_file = "config.txt" # Path to the configuration file
    start_date  = "2025-01-01" # Date to start the analysis
    end_date    = "2025-01-02" # Date to end the analysis

    # Generate the heatmaps
    heatmap = SectorHeatmap(data_file, config_file)
    heatmap.load_data()
    heatmap.load_config()
    heatmap.filter_data(start_date, end_date)
    heatmap.create_heatmap()

    # Detect patterns
    lookback_period = 5   # Example lookback period
    threshold       = 0.5 # Example deviation threshold
    heatmap.detect_pattern_deviation(lookback_period, threshold)
