import pandas as pd
import seaborn as sns
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class SectorHeatmap:
    def __init__(self, data_file, config_file):
        self.data        = None
        self.metric      = None
        self.data_file   = data_file
        self.config_file = config_file

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        self.data['UpdatedAt'] = pd.to_datetime(self.data['UpdatedAt'])

    def load_config(self):
        with open(self.config_file, 'r') as f:
            self.metric = f.read().strip()

    def filter_data(self, start_date, end_date):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.data = self.data[(self.data['UpdatedAt'] >= start_date) & (self.data['UpdatedAt'] <= end_date)]

    def create_heatmap(self):
        if self.metric not in self.data.columns:
            raise ValueError(f"Metric {self.metric} not found in data columns.")

        sector_means = self.data.groupby('Sector')[self.metric].mean().sort_values()

        # Static Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(sector_means.to_frame().T, annot=True, cmap="YlGnBu", cbar_kws={'label': self.metric})
        plt.title(f"Sector Heatmap ({self.metric})")
        plt.tight_layout()

        # Save Static Heatmap to PDF
        pdf_path = f"sector_heatmap_{self.metric}.pdf"
        with PdfPages(pdf_path) as pdf:
            pdf.savefig()
        print(f"Static heatmap saved as {pdf_path}")

        # Interactive Heatmap
        interactive_fig = px.imshow(
            sector_means.to_frame().T,
            color_continuous_scale="YlGnBu",
            labels={"color": self.metric},
            title=f"Sector Heatmap ({self.metric})"
        )

        # Save Interactive Heatmap to disk
        html_path = f"sector_heatmap_{self.metric}.html"
        interactive_fig.write_html(html_path)
        print(f"Interactive heatmap saved as {html_path}")

        return pdf_path, html_path

if __name__ == "__main__":

    data_file   = "data.csv"    # Path to the CSV data
    config_file = "config.txt"  # Path to the configuration file
    start_date  = "2025-01-01"  # Date to start the analysis
    end_date    = "2025-01-02"  # Date to end the analysis

    # Generate the heatmaps
    heatmap = SectorHeatmap(data_file, config_file)
    heatmap.load_data()
    heatmap.load_config()
    heatmap.filter_data(start_date, end_date)
    heatmap.create_heatmap()
