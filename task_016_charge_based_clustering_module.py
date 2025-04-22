import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from fpdf import FPDF
from datetime import datetime, timezone
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram


# ---------------------------
# 1. Get S&P 500 tickers + sectors
# ---------------------------
def get_sp500_tickers_and_sectors():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = table['Symbol'].tolist()
    sectors = dict(zip(table['Symbol'], table['GICS Sector']))
    return tickers, sectors


# ---------------------------
# 2. Fetch adjusted close price data
# ---------------------------
def fetch_close_prices(tickers, period='6mo'):
    data = yf.download(tickers, period=period, interval='1d', group_by='ticker', auto_adjust=True, threads=True)
    close_prices = data.xs('Close', level='Price', axis=1)
    return close_prices.dropna(axis=1, how='any')


# ---------------------------
# 3. Calculate charge values
# ---------------------------
def calculate_multiperiod_charges(close_prices):
    timeframes = {'short': 5, 'medium': 21, 'long': 63}
    charge_df = pd.DataFrame(index=close_prices.columns)

    for name, window in timeframes.items():
        returns = close_prices.pct_change(window)
        z_scores = (returns - returns.mean()) / returns.std()
        charge_df[f'charge_{name}'] = z_scores.iloc[-1]

    return charge_df.dropna()


# ---------------------------
# 4. Perform clustering
# ---------------------------
def cluster_stocks(charge_df, n_clusters=6):
    X = charge_df[['charge_short', 'charge_medium', 'charge_long']].values
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)
    charge_df['cluster'] = labels
    return charge_df


# ---------------------------
# 5. Generate trade ideas from charge polarity
# ---------------------------
def generate_trade_ideas(cluster_df):
    ideas = []

    for cluster_id in sorted(cluster_df['cluster'].unique()):
        members = cluster_df[cluster_df['cluster'] == cluster_id]

        strong_bulls = members[(members[['charge_short', 'charge_medium', 'charge_long']] > 1).all(axis=1)]
        strong_bears = members[(members[['charge_short', 'charge_medium', 'charge_long']] < -1).all(axis=1)]

        for bull in strong_bulls.index:
            ideas.append(f"LONG {bull} (cluster {cluster_id}) — strong positive charge across timeframes")

        for bear in strong_bears.index:
            ideas.append(f"SHORT {bear} (cluster {cluster_id}) — strong negative charge across timeframes")

    return ideas


def plot_charge_heatmap(charge_df, output_path="heatmap.png"):
    df_plot = charge_df[['charge_short', 'charge_medium', 'charge_long']].copy()
    df_plot.index.name = "Ticker"
    fig = px.imshow(
        df_plot,
        labels=dict(color="Charge Z-Score"),
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-2, zmax=2
    )
    fig.update_layout(title="Charge Heatmap (Short / Medium / Long)", width=1000, height=800)
    fig.write_image(output_path)

def plot_dendrogram(charge_df, output_path="dendrogram.png"):
    data = charge_df[['charge_short', 'charge_medium', 'charge_long']].values
    labels = charge_df.index.tolist()
    linkage_matrix = linkage(data, method='ward')
    fig = ff.create_dendrogram(data, labels=labels, linkagefun=lambda x: linkage_matrix)
    fig.update_layout(title="Hierarchical Clustering Dendrogram", width=1200, height=800)
    fig.write_image(output_path)


# ---------------------------
# 6. Generate PDF report
# ---------------------------
def generate_pdf_report(cluster_df, sectors, trade_ideas, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def safe_str(s):
        return str(s).replace("—", "-").encode('latin-1', errors='replace').decode('latin-1')

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, "Daily Charge-Based Clustering Report", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # Heatmap + Dendrogram
    pdf.set_font("Arial", style="B", size=13)
    pdf.cell(200, 10, "Charge Heatmap & Dendrogram", ln=True)
    pdf.image("heatmap.png", w=180)
    pdf.ln(2)
    pdf.image("dendrogram.png", w=180)
    pdf.ln(5)

    # Clusters
    for cluster_id in sorted(cluster_df['cluster'].unique()):
        members = cluster_df[cluster_df['cluster'] == cluster_id]
        net_charge = members[['charge_short', 'charge_medium', 'charge_long']].mean().mean()

        pdf.set_font("Arial", style="B", size=13)
        pdf.cell(200, 10, f"Cluster {cluster_id} (Net Charge: {net_charge:.2f})", ln=True)
        pdf.set_font("Arial", size=11)

        for symbol, row in members.iterrows():
            line = (
                f"{symbol} | Sector: {sectors.get(symbol, 'N/A')} | "
                f"Short: {row['charge_short']:.2f}, "
                f"Medium: {row['charge_medium']:.2f}, "
                f"Long: {row['charge_long']:.2f}"
            )
            pdf.cell(200, 10, safe_str(line), ln=True)
        pdf.ln(2)

    if trade_ideas:
        pdf.set_font("Arial", style="B", size=13)
        pdf.cell(200, 10, "Trade/Hedge Ideas", ln=True)
        pdf.set_font("Arial", size=11)
        for idea in trade_ideas:
            pdf.cell(200, 10, safe_str(f"- {idea}"), ln=True)

    pdf.output(filename)


# ---------------------------
# 7. Pipeline Entrypoint
# ---------------------------
def main():
    tickers, sectors = get_sp500_tickers_and_sectors()
    close_prices = fetch_close_prices(tickers)

    charge_df = calculate_multiperiod_charges(close_prices)
    if charge_df.shape[0] < 2:
        print("⚠️ Not enough valid assets for clustering.")
        return

    clustered = cluster_stocks(charge_df)
    trade_ideas = generate_trade_ideas(clustered)

    plot_charge_heatmap(clustered)
    plot_dendrogram(clustered)

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
    filename = f"charge_clusters_{timestamp}.pdf"
    generate_pdf_report(clustered, sectors, trade_ideas, filename)
    print(f"✅ Report saved to {filename}")


if __name__ == "__main__":
    main()

