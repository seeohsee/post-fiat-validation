import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from datetime import datetime

class SectorRiskScorer:
    def __init__(self, data, risk_weights=None, thresholds=None):
        self.data = data
        self.risk_weights = risk_weights or {'volatility': 0.4, 'drawdown': 0.3, 'correlation': 0.3}
        self.thresholds = thresholds or {'volatility': 20, 'drawdown': 15, 'correlation': 0.8}
    
    def calculate_volatility(self, sector_data):
        """
        Calculate the average standard deviation for the sector data.
        """
        return sector_data.std().mean()

    def calculate_drawdown(self, sector_data):
        """
        Calculate drawdown as the average relative drop from the max value.
        """
        high = sector_data.max()
        low = sector_data.min()
        return ((high - low) / high).mean()

    def calculate_correlation_risk(self, sector_data):
        """
        Calculate the average correlation within the sector's data.
        """
        # Compute the correlation matrix
        corr_matrix = sector_data.corr(method='pearson')
        
        # Extract the upper triangle of the correlation matrix, excluding the diagonal
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Calculate the mean of absolute correlations
        avg_correlation = upper_tri.abs().mean().mean()
        
        return avg_correlation

    def calculate_risk_scores(self):
        """
        Calculate risk scores for each sector.
        """
        sectors = self.data['Sector'].unique()
        risk_scores = []

        for sector in sectors:
            sector_data = self.data[self.data['Sector'] == sector]
            metrics_data = sector_data.iloc[:, 2:10]  # Adjust columns as needed
            
            # Calculate risk metrics
            volatility = self.calculate_volatility(metrics_data)
            drawdown = self.calculate_drawdown(metrics_data)
            correlation = self.calculate_correlation_risk(metrics_data)

            # Weighted risk score
            total_score = (
                self.risk_weights['volatility'] * (volatility / self.thresholds['volatility']) +
                self.risk_weights['drawdown'] * (drawdown / self.thresholds['drawdown']) +
                self.risk_weights['correlation'] * (correlation / self.thresholds['correlation'])
            ) * 100

            risk_scores.append({'Sector': sector, 'Volatility': volatility, 'Drawdown': drawdown,
                                'Correlation': correlation, 'RiskScore': total_score})
        
        return pd.DataFrame(risk_scores)
    
    def generate_pdf(self, scores_df):
        """
        Generate a PDF report of the risk scores.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"sector_risk_report_{timestamp}.pdf"
        
        c = canvas.Canvas(pdf_filename)
        c.drawString(100, 800, "Sector Risk Report")
        c.drawString(100, 780, f"Generated at: {timestamp} UTC")

        for i, row in scores_df.iterrows():
            sector_text = f"Sector: {row['Sector']} | Risk Score: {row['RiskScore']:.2f}"
            c.drawString(100, 750 - (i * 20), sector_text)
        
        # Save the PDF
        c.save()
        print(f"PDF saved as {pdf_filename}")

# Load data
data = pd.read_csv("data.csv")

# Initialize scorer
scorer = SectorRiskScorer(data)

# Calculate scores
scores_df = scorer.calculate_risk_scores()
print(scores_df)

# Generate PDF
scorer.generate_pdf(scores_df)

