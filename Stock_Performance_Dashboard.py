# 📊 DATA-DRIVEN STOCK ANALYSIS DASHBOARD (Streamlit + PostgreSQL)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sqlalchemy import create_engine, text

# --------------------------------------------------
# 🎯 PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(page_title="📈 Stock Performance Dashboard", layout="wide")

st.title("📊 Data-Driven Stock Analysis Dashboard")
st.markdown("""This interactive dashboard provides insights into **Nifty 50 stock performance trends**.""")

# --------------------------------------------------
# ⚙️ DATABASE CONNECTION
# --------------------------------------------------
DB_URL = "postgresql+psycopg2://postgres:8098086631@localhost:5432/stock_analysis"
engine = create_engine(DB_URL)

# --------------------------------------------------
# 📂 FOLDER PATH (ALL CSV FILES)
# --------------------------------------------------
DATA_FOLDER = r"E:\DATA SCIENCE\Data-Driven Stock Analysis Project\All_Ticker_CSVs"

# --------------------------------------------------
# 🧱 FUNCTION: DATABASE SETUP (CREATE BASE TABLE)
# --------------------------------------------------
def setup_database():
    """Create a base table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tickers (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10),
                close FLOAT,
                date DATE,
                high FLOAT,
                low FLOAT,
                month VARCHAR(20),
                open FLOAT,
                volume BIGINT
            );
        """))

# --------------------------------------------------
# 📊 FUNCTION: CREATE TABLE FROM CSV FILES
# --------------------------------------------------
def create_table_from_csv():
    """Safely load CSVs into 'stock_data' — create if not exists, else append."""

    # 1️⃣ Check folder
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ Folder not found: {DATA_FOLDER}")
        return

    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if not csv_files:
        st.warning("⚠️ No CSV files found in the folder.")
        return

    all_data = []
    for file in csv_files:
        file_path = os.path.join(DATA_FOLDER, file)
        df = pd.read_csv(file_path)

        # Add ticker column if missing
        ticker_name = os.path.splitext(file)[0].upper()
        if 'ticker' not in [c.lower() for c in df.columns]:
            df['ticker'] = ticker_name
        else:
            # Clean and standardize existing ticker column
            df.rename(columns={c: 'ticker' for c in df.columns if c.lower() == 'ticker'}, inplace=True)
            df['ticker'] = df['ticker'].fillna(ticker_name).str.upper().str.strip()

        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.columns = final_df.columns.str.lower().str.strip()

    # 2️⃣ Check if table exists
    table_check_query = text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'stock_data'
        );
    """)
    with engine.connect() as conn:
        table_exists = conn.execute(table_check_query).scalar()

    # 3️⃣ Upload logic
    if not table_exists:
        final_df.to_sql('stock_data', engine, if_exists='replace', index=False)
        
    else:
        st.info("📈 Appending new data to existing 'stock_data' table...")

        # ✅ Fetch table columns directly from PostgreSQL
        with engine.connect() as conn:
            existing_cols = pd.read_sql(
                "SELECT column_name FROM information_schema.columns WHERE table_name='stock_data';", 
                conn
            )['column_name'].tolist()

        # ✅ Ensure we only upload matching columns
        final_df = final_df[[col for col in final_df.columns if col in existing_cols]]

        # ✅ Append safely without redefining schema
        final_df.to_sql('stock_data', engine, if_exists='append', index=False, method='multi')
        st.success("✅ New records appended successfully!")

# --------------------------------------------------
# 🔍 FUNCTION: CHECK IF TABLE EXISTS
# --------------------------------------------------
def table_exists(table_name: str) -> bool:
    """Check if a given table exists in PostgreSQL."""
    query = text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = :table_name
        );
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"table_name": table_name}).scalar()
    return result

# --------------------------------------------------
# 📥 FUNCTION: LOAD DATA (WITH CACHE)
# --------------------------------------------------
@st.cache_data
def load_data():
    """Load stock data from PostgreSQL, or create table if missing."""
    if not table_exists("stock_data"):
        create_table_from_csv()

    query = "SELECT * FROM stock_data;"
    df = pd.read_sql(query, engine)
    return df

# --------------------------------------------------
# 🧮 LOAD + FILTER DATA
# --------------------------------------------------
try:
    df = load_data()

    # ✅ Add daily return column if missing
    if 'daily_return' not in df.columns:
        df['daily_return'] = df.groupby('ticker')['close'].pct_change()

    tickers = sorted(df['ticker'].unique())
    selected_ticker = st.selectbox("Select a Ticker:", tickers)
    ticker_df = df[df['ticker'] == selected_ticker]

    st.write("### Stock Data Preview")
    st.dataframe(ticker_df.head())

    st.line_chart(ticker_df.set_index('date')['close'])

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# --------------------------------------------------
# 🏆 TOP & BOTTOM STOCKS (Yearly Return)
# --------------------------------------------------
st.subheader("🏆 Top 10 Gainers & 📉 Top 10 Losers of the Year")

# 1️⃣ Calculate Yearly Return
yearly_return = df.groupby('ticker')['close'].agg(['first', 'last'])
yearly_return['Yearly_Return_%'] = ((yearly_return['last'] - yearly_return['first']) / yearly_return['first']) * 100
yearly_return.reset_index(inplace=True)

# Function to format large numbers
def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    else:
        return f"{n:,.0f}"
    
# 2️⃣ Market Summary (Green, Red, Avg Return, Avg Price, Avg Volume)
# Calculate metrics
green_stocks = (yearly_return['Yearly_Return_%'] > 0).sum()
red_stocks = (yearly_return['Yearly_Return_%'] <= 0).sum()
avg_return = yearly_return['Yearly_Return_%'].mean()
avg_price = df['close'].mean()
avg_volume = df['volume'].mean()

st.markdown("""<style>.metric-card {background-color: #F8F9FA; border-radius: 10px; padding: 12px; text-align: center; font-size: 18px; font-weight: bold; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);}.metric-green {background-color: #DFF0D8;}.metric-red {background-color: #F2DEDE;}.metric-blue {background-color: #D9EDF7;}.metric-gray {background-color: #FCF8E3;}</style>""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(f"<div class='metric-card metric-green'>🟢 Total Green Stocks<br>{green_stocks}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card metric-red'>🔴 Total Red Stocks<br>{red_stocks}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card metric-blue'>💰 Average Stock Price<br>{format_number(avg_price)}</div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card metric-gray'>📦 Average Volume<br>{format_number(avg_volume)}</div>", unsafe_allow_html=True)
col5.markdown(f"<div class='metric-card'>📈 Avg Yearly Return<br>{avg_return:.2f}%</div>", unsafe_allow_html=True)


# 3️⃣ Identify Top Gainers & Losers
col1, col2 = st.columns(2)
top_gainers = yearly_return.sort_values('Yearly_Return_%', ascending=False).head(10)
top_losers = yearly_return.sort_values('Yearly_Return_%', ascending=True).head(10)

with col1:
    fig = px.bar(top_gainers, x='ticker', y='Yearly_Return_%', color='Yearly_Return_%',
                color_continuous_scale='Greens', title="Top 10 Gainers")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.bar(top_losers, x='ticker', y='Yearly_Return_%', color='Yearly_Return_%',
                color_continuous_scale='Reds', title="Top 10 Losers")
    st.plotly_chart(fig, use_container_width=True)

# Push to PostgreSQL
yearly_return.to_sql('yearly_returns', engine, if_exists='replace', index=False)

# --------------------------------------------------
# 📈 VOLATILITY ANALYSIS (Top 10 Most Volatile Stocks)
# --------------------------------------------------
st.subheader("📈 Volatility Analysis (Top 10 Most Volatile Stocks)")

# Calculate volatility per ticker
volatility = df.groupby('ticker')['daily_return'].std().sort_values(ascending=False).head(10).reset_index()

# ✅ Rename column for clarity
volatility.columns = ['ticker', 'daily_return']

# ✅ Check if DataFrame is not empty before plotting
fig = px.bar(volatility, x='ticker', y='daily_return', color='daily_return',
            color_continuous_scale='Blues', title="Top 10 Most Volatile Stocks")
st.plotly_chart(fig, use_container_width=True)

# Push to PostgreSQL
volatility.to_sql('volatility', engine, if_exists='replace', index=False)

# --------------------------------------------------
# 📊 CUMULATIVE RETURN OVER TIME
# --------------------------------------------------
st.subheader("📊 Cumulative Return Over Time (Top 5 Stocks)")

# ✅ Calculate cumulative return for each stock
df['cumulative_return'] = (1 + df['daily_return']).groupby(df['ticker']).cumprod() - 1

# ✅ Calculate yearly return (if not already computed)
# If 'yearly_return' DataFrame exists, it’s reused. Otherwise, compute it.
if 'yearly_return' not in locals():
    yearly_return = (
        df.groupby('ticker')
        .apply(lambda x: (1 + x['daily_return']).prod() - 1)
        .reset_index(name='Yearly_Return_%')
    )

# ✅ Identify top 5 performing stocks based on yearly return
top5_tickers = yearly_return.sort_values('Yearly_Return_%', ascending=False)['ticker'].head(5).tolist()

# ✅ Plot cumulative return over time for top 5 stocks
fig, ax = plt.subplots(figsize=(10, 5))

for ticker in top5_tickers:
    ticker_df = df[df['ticker'] == ticker].copy()

    # Ensure date is in datetime format
    ticker_df['date'] = pd.to_datetime(ticker_df['date'], errors='coerce')

    # Drop rows with missing date or cumulative return
    ticker_df = ticker_df.dropna(subset=['date', 'cumulative_return'])

    # Sort by date for proper line plotting
    ticker_df = ticker_df.sort_values('date')

    if not ticker_df.empty:
        ax.plot(ticker_df['date'], ticker_df['cumulative_return'], label=ticker)

ax.set_title("Cumulative Return Over Time (Top 5 Performing Stocks)", fontweight='bold')
ax.set_xlabel("Date", fontweight='bold')
ax.set_ylabel("Cumulative Return", fontweight='bold')
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ✅ Prepare cumulative returns data for PostgreSQL storage
cumulative_returns = df[['ticker', 'date', 'cumulative_return']].copy()
cumulative_returns['date'] = pd.to_datetime(cumulative_returns['date'], errors='coerce')
cumulative_returns = cumulative_returns.dropna(subset=['date', 'cumulative_return'])

# Push to PostgreSQL
cumulative_returns.to_sql('cumulative_returns', engine, if_exists='replace', index=False)

# --------------------------------------------------
# 🏭 Sector-wise Performance
# --------------------------------------------------
st.subheader("🏭 Sector-wise Performance")

# Manual mapping of tickers to sectors
sector_map = {
    'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT',
    'HDFCBANK': 'Financials', 'ICICIBANK': 'Financials', 'SBIN': 'Financials',
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'COALINDIA': 'Energy',
    'ITC': 'FMCG', 'HINDUNILVR': 'FMCG', 'BRITANNIA': 'FMCG',
    'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals',
    'ULTRACEMCO': 'Cement', 'ASIANPAINT': 'Consumer',
    'MARUTI': 'Automobile', 'EICHERMOT': 'Automobile'
}

# Add sector info
df['sector'] = df['ticker'].map(sector_map).fillna('Unknown')
df = df[df['sector'] != 'Unknown']

# Calculate yearly average return per sector
df['year'] = pd.to_datetime(df['date']).dt.year
returns = df.groupby(['year', 'ticker', 'sector'])['close'].agg(['first', 'last'])
returns['Yearly_Return_%'] = (returns['last'] - returns['first']) / returns['first'] * 100
sector_perf = returns.groupby(['year', 'sector'])['Yearly_Return_%'].mean().reset_index()

# Select year
year = st.selectbox("Select Year", sorted(sector_perf['year'].unique()))
data = sector_perf[sector_perf['year'] == year]

# 📊 Average Yearly Return by Sector (Plotly)
fig = px.bar(
    data,
    x='sector',
    y='Yearly_Return_%',
    color='Yearly_Return_%',
    color_continuous_scale='RdYlGn',
    title=f"Average Yearly Return by Sector ({year})",
)

# Customize layout
fig.update_layout(
    xaxis_title="Sector",
    yaxis_title="Avg Return (%)",
    title_font=dict(size=16, family='Arial', color='black'),
    xaxis_tickangle=45,
    template='plotly_white'
)

# Add % labels on top of bars
fig.update_traces(
    text=data['Yearly_Return_%'].round(2).astype(str) + '%',
    textposition='outside',
)

# Display chart in Streamlit
st.plotly_chart(fig, use_container_width=True)


# Push to PostgreSQL
sector_perf.to_sql('sector_performance', engine, if_exists='replace', index=False)

# --------------------------------------------------
# 🤝 CORRELATION HEATMAP
# --------------------------------------------------
st.subheader("🤝 Stock Price Correlation Heatmap")

# 1️⃣ Create pivot table for closing prices
pivot_close = df.pivot(index='date', columns='ticker', values='close')

# 2️⃣ Compute correlation matrix
corr_matrix = pivot_close.corr()

# 3️⃣ Reset index and rename it clearly
corr_matrix.index.name = 'ticker'

# 4️⃣ Convert to long format for Power BI
corr_long = corr_matrix.reset_index().melt(
    id_vars='ticker',
    var_name='ticker_2',
    value_name='correlation'
)

# 5️⃣ Clean data (remove NaN, round values)
corr_long = corr_long.dropna()
corr_long['correlation'] = corr_long['correlation'].round(3)

# 6️⃣ Display heatmap (Plotly
fig = px.imshow(
    corr_matrix,
    text_auto=True,  # show correlation values
    color_continuous_scale='RdBu_r',
    title="Correlation Heatmap of Stock Features")

fig.update_layout(
    width=800,
    height=600,
    title_font=dict(size=16, family='Arial', color='black'),
    template='plotly_white')

st.plotly_chart(fig, use_container_width=True)


# 7️⃣ Push to PostgreSQL (long format)
corr_long.to_sql('correlation_matrix', engine, if_exists='replace', index=False)


# --------------------------------------------------
# 🗓️ MONTHLY GAINERS & LOSERS (aligned with Power BI)
# --------------------------------------------------

monthly_return = pd.read_sql("SELECT * FROM public.monthly_returns", engine)

if 'month' in monthly_return.columns and not monthly_return['month'].isna().all():
    months = sorted(monthly_return['month'].dropna().unique())
    st.markdown("### 📅 Monthly Gainers & Losers")

    selected_month = st.selectbox("Select Month", months, index=len(months)-1)
    st.subheader(f"Monthly ▲ Gainers & ▼ Losers ({selected_month})")

    # Filter selected month
    month_data = monthly_return[monthly_return['month'] == selected_month]

    # Top 5 and Bottom 5
    top5_month = month_data.sort_values('Monthly_Return_%', ascending=False).head(5)
    bottom5_month = month_data.sort_values('Monthly_Return_%', ascending=True).head(5)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(top5_month, x='ticker', y='Monthly_Return_%', color='Monthly_Return_%',
                     color_continuous_scale='Greens', title=f"Top 5 Gainers - {selected_month}",)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(bottom5_month, x='ticker', y='Monthly_Return_%', color='Monthly_Return_%',
                     color_continuous_scale='Reds', title=f"Top 5 Losers - {selected_month}")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 🧾 FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("📊 **Data-Driven Stock Analysis Dashboard** | Built with ❤️ using Python, Streamlit, and PostgreSQL")

# Push to PostgreSQL all summary tables
def push_all_summary_tables(df):
    try:
        # 1. Yearly returns
        yearly_return = df.groupby('ticker')['close'].agg(['first', 'last'])
        yearly_return['Yearly_Return_%'] = ((yearly_return['last'] - yearly_return['first']) / yearly_return['first']) * 100
        yearly_return.reset_index(inplace=True)
        yearly_return.to_sql('yearly_returns', engine, if_exists='replace', index=False)

        # 2. Volatility
        volatility = df.groupby('ticker')['daily_return'].std().reset_index()
        volatility.rename(columns={'daily_return': 'Volatility'}, inplace=True)
        volatility.to_sql('volatility', engine, if_exists='replace', index=False)

        # 3. Cumulative returns
        df['cumulative_return'] = (1 + df['daily_return']).groupby(df['ticker']).cumprod() - 1
        cumulative_returns = df[['ticker', 'date', 'cumulative_return']].dropna()
        cumulative_returns.to_sql('cumulative_returns', engine, if_exists='replace', index=False)

        # 4. Sector performance
        df['sector'] = df['ticker'].map(sector_map).fillna('Unknown')
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
        returns = df.groupby(['year', 'ticker', 'sector'])['close'].agg(['first', 'last'])
        returns['Yearly_Return_%'] = (returns['last'] - returns['first']) / returns['first'] * 100
        sector_perf = returns.groupby(['year', 'sector'])['Yearly_Return_%'].mean().reset_index()
        sector_perf.to_sql('sector_performance', engine, if_exists='replace', index=False)

        # 5. Monthly returns
        if 'month' in df.columns and not df['month'].isna().all():
            monthly_return = df.groupby(['month', 'ticker'])['close'].agg(['first', 'last'])
            monthly_return['Monthly_Return_%'] = ((monthly_return['last'] - monthly_return['first']) / monthly_return['first']) * 100
            monthly_return.reset_index(inplace=True)
            monthly_return.to_sql('monthly_returns', engine, if_exists='replace', index=False)

        # 6. Correlation matrix
        pivot_close = df.pivot(index='date', columns='ticker', values='close')
        corr_matrix = pivot_close.corr().reset_index().melt(id_vars='ticker', var_name='ticker2', value_name='correlation')
        corr_matrix.to_sql('correlation_matrix', engine, if_exists='replace', index=False)

        st.success("✅ All summary tables uploaded to PostgreSQL successfully!")

    except Exception as e:
        st.error(f"❌ Error uploading summary tables: {e}")
