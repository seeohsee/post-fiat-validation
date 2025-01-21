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
        self.charge_dynamics = {}

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

    def calculate_charge_dynamics(self, charges):
        dynamics = {}
        timeframes = [1, 5, 20]  # 1D, 5D, 20D

        for ticker, charge_series in charges.items():
            dynamics[ticker] = {}
            for timeframe in timeframes:
                charge_diff = charge_series.diff(timeframe)
                charge_accel = charge_diff.diff(timeframe)
                polarity_reversals = ((charge_series * charge_series.shift(timeframe)) < 0).sum()

                dynamics[ticker][f'{timeframe}D'] = {
                    'Charge Acceleration': charge_accel.fillna(0),
                    'Polarity Reversals': polarity_reversals
                }

        self.charge_dynamics = dynamics

    def rank_stocks(self):
        stability_scores = {}
        acceleration_magnitudes = {}

        for ticker, dynamics in self.charge_dynamics.items():
            stability_score = 0
            total_acceleration = 0
            for timeframe, metrics in dynamics.items():
                total_acceleration += metrics['Charge Acceleration'].abs().sum()
                stability_score += metrics['Polarity Reversals']

            stability_scores[ticker] = stability_score
            acceleration_magnitudes[ticker] = total_acceleration

        sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1])
        sorted_acceleration = sorted(acceleration_magnitudes.items(), key=lambda x: x[1], reverse=True)

        return sorted_stability, sorted_acceleration

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

        pdf.ln(10)
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(200, 10, txt="Charge Dynamics", ln=True)
        pdf.set_font("Arial", size=10)

        for ticker, dynamics in self.charge_dynamics.items():
            pdf.cell(200, 10, txt=f"{ticker} Dynamics:", ln=True)
            for timeframe, metrics in dynamics.items():
                pdf.cell(200, 10, txt=f"  {timeframe}: Polarity Reversals: {metrics['Polarity Reversals']}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(200, 10, txt="Stock Rankings", ln=True)

        stability_ranking, acceleration_ranking = self.rank_stocks()

        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Stability Rankings (Lowest to Highest):", ln=True)
        for ticker, score in stability_ranking:
            pdf.cell(200, 10, txt=f"  {ticker}: {score}", ln=True)

        pdf.ln(5)
        pdf.cell(200, 10, txt="Acceleration Magnitude Rankings (Highest to Lowest):", ln=True)
        for ticker, magnitude in acceleration_ranking:
            pdf.cell(200, 10, txt=f"  {ticker}: {magnitude:.2f}", ln=True)

        pdf.output(filename)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "ABBV", "COIN", "PYPL", "NVDA", "TSLA"]
    start_date = "2022-01-01"
    end_date = "2023-01-01"
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

    # Calculate charge dynamics
    market_module.calculate_charge_dynamics(charges)

    # Generate a report with a UTC timestamp in the filename
    utc_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f"market_electrostatics_report_{utc_timestamp}.pdf"
    market_module.generate_report(filename)

    print(f"Report generated: {filename}")
