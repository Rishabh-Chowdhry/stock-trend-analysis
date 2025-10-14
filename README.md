Equity Lens – Glassmorphic Executive Dashboard
A single-file, production-ready Dash application that delivers a responsive, WCAG-friendly, glassmorphic equity-analysis cockpit without external assets.
✨ Highlights
Table
Copy
Feature	Description
Glassmorphism UI	Frosted-glass cards, soft shadows, light-mode only
Full Chart Stack	Price + MAs, Candlestick + Bollinger, MACD, RSI, Sector Sunburst, Treemap, Correlation Heat-map, Drawdown, YTD vs Benchmark, Top-Gainers & Volatility scatters
Mobile-first	Responsive down to 320 px; touch-friendly tables
Zero External Files	Google Fonts is the only remote dependency
WCAG 2.2 Compliant	Colour-contrast & keyboard nav ready
Flat-File Data	Auto-caches Polygon.io daily bars; falls back to realistic mock data if API unavailable
One-Click Export	CSV / Excel download for every table
ARIMA Forecast	1-day ahead close-price forecast baked into score

# 1. Clone or copy app.py
# 2. Install deps
pip install dash==2.16.0 dash-bootstrap-components==1.5.0 plotly==5.19.0 pandas==2.2.0 statsmodels==0.14.0

# 3. Run
python app.py

Open http://localhost:8050 – done.

Chart Catalogue
Table
Copy
Chart	Purpose
Price + Moving Averages	20 / 50 / 200-day ribbon
Candlestick + Bollinger	20-period, 2-σ bands
MACD	Histogram + signal line
RSI	14-period with 30/70 levels
Sector Sunburst	Market-cap weighted returns
Treemap	Portfolio-weight by cap
Correlation Heat-map	20-asset Pearson matrix
Underwater Drawdown	Peak-to-trough equity curve
YTD vs Benchmark	Simulated S&P 500 drift
Top Gainers	Horizontal bar + mini-table
High Volatility	Risk-return scatter + mini-table
Investment Suggestions	5-y CAGR + ARIMA score, sortable
