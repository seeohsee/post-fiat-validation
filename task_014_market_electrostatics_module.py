import yfinance as yf
from fpdf import FPDF
import numpy as np
from datetime import datetime

class MarketElectrostatics:
    def __init__(self, tickers, start_date, end_date, charge_decay_rate, min_volume_threshold):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.charge_decay_rate = charge_decay_rate
        self.min_volume_threshold = min_volume_threshold
        self.data = {}
        self.forces = {}

    def fetch_data(self):
        for ticker in self.tickers:
            stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)
            stock_data.columns = stock_data.columns.droplevel(1) # Remove the MultiIndex
            self.data[ticker] = stock_data

    def calculate_charges(self):
        charges = {}
        for ticker, df in self.data.items():
            df['Momentum'] = df['Close'].pct_change()
            df['Charge'] = df['Momentum'] * df['Volume']
            df['Charge'] = df['Charge'].apply(lambda x: x if abs(x) > self.min_volume_threshold else 0)
            charges[ticker] = df['Charge'].fillna(0)
        return charges

    def calculate_coulomb_forces(self, charges):
        tickers = list(charges.keys())
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if i < j:
                    force = (charges[t1] * charges[t2]) / self.charge_decay_rate
                    self.forces[(t1, t2)] = force.fillna(0)

    def generate_report(self, filename):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Market Electrostatics Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=10)
        for (t1, t2), force in self.forces.items():
            pdf.cell(200, 10, txt=f"Coulomb Force between {t1} and {t2}: {force.mean():.2f}", ln=True)

        pdf.output(filename)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "ABBV", "COIN", "PYPL", "NVDA", "TSLA"]
    start_date = "2022-01-01"
    end_date = "2025-01-17"
    charge_decay_rate = 1000
    min_volume_threshold = 1e6

    # Instantiate the module
    market_module = MarketElectrostatics(tickers, start_date, end_date, charge_decay_rate, min_volume_threshold)

    # Fetch historical data
    market_module.fetch_data()

    # Calculate charges
    charges = market_module.calculate_charges()

    # Calculate Coulomb forces
    market_module.calculate_coulomb_forces(charges)

    # Generate a report with a UTC timestamp in the filename
    utc_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f"market_electrostatics_report_{utc_timestamp}.pdf"
    market_module.generate_report(filename)

    print(f"Report generated: {filename}")
