import os, dash, base64, warnings, numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
px.defaults.template = "plotly_white"  # light mode only

# ------------------------------------------------------------------
# 1.  CONFIG
# ------------------------------------------------------------------
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "demo")
CACHE_DIR = "polygon_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2.  TICKER UNIVERSE
# ------------------------------------------------------------------
def sp500_tickers(limit=20):
    try:
        return [t.replace(".", "-") for t in
                pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()[:limit]]
    except:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "NFLX", "AMD",
                "JPM", "V", "WMT", "CRM", "ORCL", "ADBE", "PYPL", "INTC", "QCOM", "AVGO", "SHOP"]

STOCKS = sp500_tickers()

# ------------------------------------------------------------------
# 3.  DATA LAYER  –  local cache + mock fallback
# ------------------------------------------------------------------
class DataMgr:
    def __init__(self):
        self.cache_dir = CACHE_DIR

    def fetch(self, ticker: str, days: int = 30) -> pd.DataFrame:
        ticker = ticker.upper()
        csv_path = os.path.join(self.cache_dir, f"{ticker}_daily.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, usecols=["date", "open", "high", "low", "close", "volume"],
                                 parse_dates=["date"]).rename(columns={"date": "Date"}).set_index("Date").sort_index()
                cut = df.last(f"{days}D")
                return cut if not cut.empty else df.tail(days)
            except:
                pass
        return self._mock(ticker, days)

    def _mock(self, ticker: str, days: int) -> pd.DataFrame:
        end = datetime.now()
        dr = pd.date_range(end=end, periods=days*2, freq="D")
        base = {"AAPL": 150, "MSFT": 300, "NVDA": 450, "AMZN": 130, "META": 350,
                "TSLA": 180, "GOOGL": 140, "NFLX": 500, "AMD": 120, "JPM": 150,
                "V": 250, "WMT": 60, "CRM": 200, "ORCL": 110, "ADBE": 520,
                "PYPL": 60, "INTC": 35, "QCOM": 130, "AVGO": 1100, "SHOP": 70}.get(ticker, 100)
        np.random.seed(abs(hash(ticker)) % 5000)
        r = np.random.normal(0.0005, 0.02, len(dr))
        px_vec = base * (1 + np.cumsum(r))
        df = pd.DataFrame({
            "Date": dr,
            "Open": px_vec * (1 - np.random.uniform(0, 0.01, len(dr))),
            "High": px_vec * (1 + np.random.uniform(0, 0.02, len(dr))),
            "Low": px_vec * (1 - np.random.uniform(0, 0.02, len(dr))),
            "Close": px_vec,
            "Volume": np.random.randint(1e6, 1e7, len(dr))
        }).set_index("Date").sort_index()
        return df.last(f"{days}D") if len(df) > days else df

    def metrics(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 2:
            return {"Return %": 0, "Volatility (Std Dev)": 0}
        df = df.dropna(subset=["Close"])
        c0, c1 = df["Close"].iloc[0], df["Close"].iloc[-1]
        ret = (c1 - c0) / c0 * 100 if c0 else 0
        vol = float(df["Close"].pct_change().std()) if len(df) > 1 else 0
        return {"Return %": ret, "Volatility (Std Dev)": vol}

    def arima_fc(self, df: pd.DataFrame) -> float:
        recent = pd.to_numeric(df["Close"].dropna().tail(30), errors="coerce").dropna()
        if len(recent) < 10:
            return 0
        try:
            model = sm.tsa.ARIMA(recent, order=(1, 1, 0)).fit()
            val = float(model.forecast(steps=1).iloc[-1])
            return (val / recent.iloc[-1]) - 1
        except:
            return (recent.iloc[-1] / recent.iloc[-2]) - 1 if len(recent) >= 2 else 0

    def suggestions(self, stocks, years=5):
        sug = []
        for t in stocks:
            df = self.fetch(t, days=365)
            if df.empty or len(df) < 2:
                continue
            first, last = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])

            # Fix for complex numbers in CAGR calculation
            if first <= 0:
                cagr = 0
            else:
                cagr_calc = ((last / first) ** (1 / years) - 1) * 100
                cagr = cagr_calc.real if isinstance(cagr_calc, complex) else cagr_calc

            vol = float(df["Close"].pct_change().tail(30).std()) if len(df) > 1 else 0
            fc = self.arima_fc(df) * 100
            score = cagr + fc - vol * 100

            cagr_safe = float(cagr.real) if isinstance(cagr, complex) else float(cagr)
            fc_safe = float(fc.real) if isinstance(fc, complex) else float(fc)
            vol_safe = float(vol.real) if isinstance(vol, complex) else float(vol)
            score_safe = float(score.real) if isinstance(score, complex) else float(score)

            sug.append({"Stock": t, "CAGR %": round(cagr_safe, 2),
                        "Forecast %": round(fc_safe, 2), "Volatility": round(vol_safe, 4),
                        "Score": round(score_safe, 2)})

        if not sug:
            default = [{"Stock": s, "CAGR %": 8.0, "Forecast %": 1.5,
                        "Volatility": 0.02, "Score": 7.5} for s in stocks[:5]]
            sug_df = pd.DataFrame(default)
            return sug_df, sug_df.iloc[0]["Stock"], f"{sug_df.iloc[0]['Stock']} has the highest composite score."

        sug_df = pd.DataFrame(sug).sort_values("Score", ascending=False)
        return sug_df, sug_df.iloc[0]["Stock"], f"{sug_df.iloc[0]['Stock']} has the highest composite score."

DM = DataMgr()
SUG_DF, TOP_S, REASON = DM.suggestions(STOCKS)
if SUG_DF.empty:
    SUG_DF = pd.DataFrame({"Stock": ["AAPL"], "CAGR %": [8.0], "Forecast %": [1.5],
                           "Volatility": [0.02], "Score": [7.5]})
    TOP_S = "AAPL"

# ------------------------------------------------------------------
# 4.  UI  –  semantic / glassmorphic
# ------------------------------------------------------------------
def glass_card(children, header=None, style=None):
    return dbc.Card(
        dbc.CardBody([html.H5(header, className="mb-3")] +
                     (children if isinstance(children, list) else [children])),
        className="glass-card mb-4", style=style
    )

def kpi_card(title, value, id_name):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-title mb-1 text-muted fw-normal"),
            html.H3(value, className="mb-0 fw-bold", id=id_name)
        ]),
        className="glass-card mb-3", style={"height": "120px", "width": "100%"}
    )

def create_gainers_chart(gainers_data):
    """Create a horizontal bar chart for top gainers"""
    if not gainers_data:
        return px.bar(title="No data available")

    df = pd.DataFrame(gainers_data)
    df = df.head(8)

    colors = ['#ff4d4d' if x < 0 else '#2ecc71' for x in df['Return %']]

    fig = px.bar(df, y='Stock', x='Return %',
                 title="Top Performers",
                 color='Return %',
                 color_continuous_scale=['#ff4d4d', '#f39c12', '#2ecc71'],
                 orientation='h')

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Return %",
        yaxis_title="",
        showlegend=False
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Return: %{x:.2f}%<extra></extra>"
    )

    return fig

def create_volatility_chart(volatility_data):
    """Create a scatter plot for high volatility stocks"""
    if not volatility_data:
        return px.scatter(title="No data available")

    df = pd.DataFrame(volatility_data)
    df = df.head(10)

    fig = px.scatter(df, x='Volatility (Std Dev)', y='Return %',
                     size='Volatility (Std Dev)',
                     color='Return %',
                     hover_name='Stock',
                     title="Risk-Return Profile",
                     color_continuous_scale='Viridis')

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Volatility",
        yaxis_title="Return %",
        showlegend=False
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig.add_vline(x=df['Volatility (Std Dev)'].median(), line_dash="dash", line_color="gray", opacity=0.7)

    return fig

header = dbc.Container(
    dbc.Row([
        dbc.Col(html.H1("Equity Lens", className="logo fw-bolder m-0"), width="auto"),
        dbc.Col(html.P(REASON, className="mb-0 text-muted"), width="auto", className="d-flex align-items-end")
    ], justify="between", align="end"),
    fluid=True, className="py-4"
)

controls = dbc.Row([
    dbc.Col(dcc.Dropdown(id="ticker", options=[{"label": s, "value": s} for s in STOCKS],
                         value=STOCKS[0], clearable=False, className="glass-select"), md=4),
    dbc.Col(dcc.Dropdown(id="period", options=[{"label": "1 D", "value": "1"},
                                               {"label": "5 D", "value": "5"},
                                               {"label": "1 M", "value": "30"},
                                               {"label": "1 Y", "value": "365"}],
                         value="30", clearable=False, className="glass-select"), md=3),
    dbc.Col(dbc.Button("⟳  Refresh", id="refresh", color="light", outline=True, size="sm", className="h-100"), md=2),
    dbc.Col([
        dbc.Button("⬇ CSV", id="btn-csv", color="success", size="sm", className="me-2"),
        dbc.Button("⬇ Excel", id="btn-xlsx", color="info", size="sm"),
        dcc.Download(id="download1")
    ], md=3)
], className="g-2 mb-4")

main_grid = html.Div([
    dbc.Row([
        dbc.Col(kpi_card("Top Pick", TOP_S, "kpi-pick"), md=3, sm=6, xs=12),
        dbc.Col(kpi_card("CAGR (5 y)", f"{SUG_DF.iloc[0]['CAGR %']}%", "kpi-cagr"), md=3, sm=6, xs=12),
        dbc.Col(kpi_card("Next-Day FC", f"{SUG_DF.iloc[0]['Forecast %']}%", "kpi-fc"), md=3, sm=6, xs=12),
        dbc.Col(kpi_card("Score", f"{SUG_DF.iloc[0]['Score']}", "kpi-score"), md=3, sm=6, xs=12)
    ], className="g-3 mb-4"),
    dbc.Row([
        dbc.Col(glass_card(dcc.Graph(id="line", config={"displayModeBar": False}, style={"height": "60vh"}), "Price Trend + Moving Averages"), md=6, sm=12),
        dbc.Col(glass_card(dcc.Graph(id="candle", config={"displayModeBar": False}, style={"height": "60vh"}), "Candlestick + Bollinger Bands"), md=6, sm=12)
    ], className="g-3 mb-4"),
    dbc.Row([
        dbc.Col(glass_card(dcc.Graph(id="bar", config={"displayModeBar": False}, style={"height": "60vh"}), "MACD"), md=6, sm=12),
        dbc.Col(glass_card(dcc.Graph(id="scatter3d", config={"displayModeBar": False}, style={"height": "60vh"}), "RSI (14)"), md=6, sm=12)
    ], className="g-3 mb-4"),
    dbc.Row([
        dbc.Col(glass_card(dcc.Graph(id="sunburst", config={"displayModeBar": False}, style={"height": "60vh"}), "Sector Allocation"), md=6, sm=12),
        dbc.Col(glass_card(dcc.Graph(id="treemap", config={"displayModeBar": False}, style={"height": "60vh"}), "Portfolio Weight"), md=6, sm=12)
    ], className="g-3 mb-4"),
    dbc.Row([
        dbc.Col(glass_card(dcc.Graph(id="heatmap", config={"displayModeBar": False}, style={"height": "60vh"}), "Correlation Heat-map"), md=6, sm=12),
        dbc.Col([
            glass_card([
                dcc.Graph(id="gainers-chart", config={"displayModeBar": False}, style={"height": "30vh"}),
                html.Div([
                    dash_table.DataTable(
                        id="table-gl",
                        page_size=5,
                        style_cell={"textAlign": "center", "fontSize": "0.85rem", "padding": "6px 10px", "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
                        style_header={'backgroundColor': 'rgba(30, 136, 229, 0.15)', 'fontWeight': 'bold', 'fontSize': '0.9rem', 'textAlign': 'center'},
                        style_data={'backgroundColor': 'rgba(255, 255, 255, 0.5)', 'border': '1px solid rgba(240, 240, 240, 0.8)'},
                        style_data_conditional=[
                            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(245, 248, 250, 0.6)"},
                            {"if": {"column_id": "Return %", "filter_query": '{Return %} > 0'}, "color": "#2ecc71", "fontWeight": "bold"},
                            {"if": {"column_id": "Return %", "filter_query": '{Return %} < 0'}, "color": "#e74c3c", "fontWeight": "bold"}
                        ]
                    )
                ], className="table-responsive", style={"maxHeight": "15vh", "overflowY": "auto"})
            ], "Top Gainers"),
            glass_card([
                dcc.Graph(id="volatility-chart", config={"displayModeBar": False}, style={"height": "30vh"}),
                html.Div([
                    dash_table.DataTable(
                        id="table-vol",
                        page_size=8,
                        style_cell={"textAlign": "center", "fontSize": "0.85rem", "padding": "6px 10px"},
                        style_header={'backgroundColor': 'rgba(156, 39, 176, 0.15)', 'fontWeight': 'bold'},
                        style_data={'backgroundColor': 'rgba(255, 255, 255, 0.5)'},
                        style_data_conditional=[{"if": {"column_id": "Volatility (Std Dev)"}, "color": "#9c27b0", "fontWeight": "bold"}]
                    )
                ], className="table-responsive", style={"maxHeight": "15vh", "overflowY": "auto"})
            ], "High Volatility")
        ], md=6, sm=12)
    ], className="g-3 mb-4"),
    glass_card([
        dash_table.DataTable(
            id="table-sug",
            page_size=10,
            style_cell={"textAlign": "center", "fontSize": "0.9rem", "padding": "8px 12px"},
            style_header={'backgroundColor': 'rgba(76, 175, 80, 0.15)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(245, 248, 250, 0.6)"},
                {"if": {"column_id": "Score"}, "fontWeight": "bold", "color": "#1976d2"}
            ]),
        html.A("⬇  suggestions.csv", id="down-sug", className="btn btn-sm btn-outline-dark mt-2", download="suggestions.csv", href="")
    ], "Investment Suggestions (5 y CAGR + ARIMA)", style={"width": "100%"})
], className="main-container")

# Initialize the Dash app with callback exceptions suppressed
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.layout = dbc.Container([header, controls, html.Hr(className="my-2"), main_grid,
                            dcc.Interval(id="interval", interval=60_000, n_intervals=0),
                            dcc.Download(id="download"), html.Div(id="dummy")],
                           fluid=True, className="main-container")

# ------------------------------------------------------------------
# 5.  CALLBACKS  –  COMPLETE CHART SUITE
# ------------------------------------------------------------------
@app.callback(
    [Output("line", "figure"), Output("candle", "figure"), Output("bar", "figure"),
     Output("scatter3d", "figure"), Output("sunburst", "figure"), Output("treemap", "figure"),
     Output("heatmap", "figure"), Output("table-gl", "data"), Output("table-gl", "columns"),
     Output("table-vol", "data"), Output("table-vol", "columns"), Output("table-sug", "data"),
     Output("down-sug", "href"), Output("gainers-chart", "figure"), Output("volatility-chart", "figure")],
    Input("ticker", "value"), Input("period", "value"), Input("interval", "n_intervals"), Input("refresh", "n_clicks")
)
def update(ticker, period, n, clk):
    df = DM.fetch(ticker, days=int(period))

    # 1.  Line chart – price trend + MAs
    if not df.empty:
        df["MA20"] = df.Close.rolling(20).mean()
        df["MA50"] = df.Close.rolling(50).mean()
        df["MA200"] = df.Close.rolling(200).mean()
        line = px.line(df, x=df.index, y=["Close", "MA20", "MA50", "MA200"],
                       title=f"{ticker} – last {period} day(s)",
                       color_discrete_map={"Close": "#1f77b4", "MA20": "#ff7f0e", "MA50": "#2ca02c", "MA200": "#d62728"})
    else:
        line = px.line(title="No data")

    # 2.  Candlestick + Bollinger
    if not df.empty:
        df["BBm"] = df.Close.rolling(20).mean()
        df["BBu"] = df.BBm + 2 * df.Close.rolling(20).std()
        df["BBl"] = df.BBm - 2 * df.Close.rolling(20).std()
        candle = go.Figure()
        candle.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Candle"))
        candle.add_trace(go.Scatter(x=df.index, y=df.BBu, line=dict(color="lightblue"), name="BB upper"))
        candle.add_trace(go.Scatter(x=df.index, y=df.BBl, fill="tonexty", fillcolor="rgba(100,181,246,0.2)",
                                    line=dict(color="lightblue"), name="BB lower", hoverinfo="skip"))
        candle.add_trace(go.Scatter(x=df.index, y=df.BBm, line=dict(color="orange", dash="dot"), name="BB mid"))
        candle.update_layout(title="Candlestick + Bollinger Bands (20, 2)", xaxis_rangeslider_visible=False)
    else:
        candle = go.Figure()

    # 3.  MACD
    if not df.empty:
        ema12 = df.Close.ewm(span=12).mean()
        ema26 = df.Close.ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df.MACD.ewm(span=9).mean()
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df.MACD, line=dict(color="#1f77b4"), name="MACD"))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df.Signal, line=dict(color="#ff7f0e"), name="Signal"))
        macd_fig.add_bar(x=df.index, y=df.MACD - df.Signal, name="Histogram", marker_color="rgba(150,150,150,0.5)")
        macd_fig.update_layout(title="MACD")
        bar = macd_fig
    else:
        bar = px.bar(title="No data")

    # 4.  RSI
    if not df.empty:
        delta = df.Close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        rsi_fig = px.line(df, x=df.index, y="RSI", title="RSI (14)")
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        scatter3d = rsi_fig
    else:
        scatter3d = px.line(title="No data")

    # 5.  Sunburst Chart - Sector Allocation
    sectors = {
        "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "ADBE", "CRM", "ORCL", "PYPL", "INTC", "QCOM", "AVGO"],
        "Consumer Cyclical": ["AMZN", "TSLA", "META", "NFLX", "SHOP"],
        "Financial Services": ["JPM", "V"],
        "Consumer Defensive": ["WMT"],
        "Other": ["AMD"]
    }

    sector_data = []
    for sector, stocks in sectors.items():
        for stock in stocks:
            if stock in STOCKS:
                df_stock = DM.fetch(stock, days=30)
                if not df_stock.empty:
                    return_pct = ((df_stock["Close"].iloc[-1] - df_stock["Close"].iloc[0]) / df_stock["Close"].iloc[0]) * 100
                    sector_data.append({
                        "Sector": sector,
                        "Stock": stock,
                        "Return %": return_pct,
                        "Market Cap": np.random.uniform(10, 1000)
                    })

    sector_df = pd.DataFrame(sector_data)
    if not sector_df.empty:
        sunburst = px.sunburst(
            sector_df,
            path=['Sector', 'Stock'],
            values='Market Cap',
            color='Return %',
            color_continuous_scale='RdYlGn',
            title="Sector Allocation & Performance"
        )
    else:
        sunburst = px.sunburst(title="No sector data available")

    # 6.  Treemap Chart - Portfolio Weight
    if not sector_df.empty:
        treemap = px.treemap(
            sector_df,
            path=['Sector', 'Stock'],
            values='Market Cap',
            color='Return %',
            color_continuous_scale='RdYlGn',
            title="Portfolio Weight by Market Cap"
        )
    else:
        treemap = px.treemap(title="No portfolio data available")

    # 7.  Correlation heat-map
    big = pd.concat([DM.fetch(s, days=30).assign(Stock=s) for s in STOCKS[:20]]) if STOCKS else pd.DataFrame()
    if not big.empty:
        corr_data = big.pivot_table(values="Close", index="Date", columns="Stock")
        if not corr_data.empty and len(corr_data.columns) > 1:
            corr = corr_data.corr()
            heatmap = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix", color_continuous_scale="RdBu_r")
        else:
            heatmap = px.imshow(np.zeros((1, 1)), title="Not enough data for correlation matrix")
    else:
        heatmap = px.imshow(np.zeros((1, 1)), title="No data")

    # 8.  Gainers and Volatility tables
    mdf = pd.DataFrame([DM.metrics(DM.fetch(s, days=30)) | {"Stock": s} for s in STOCKS])

    # Top gainers table
    gl = mdf.sort_values("Return %", ascending=False).head(10)
    gl_columns = [{"name": "Stock", "id": "Stock"},
                  {"name": "Return %", "id": "Return %"},
                  {"name": "Volatility", "id": "Volatility (Std Dev)"}]

    # High volatility table
    vol = mdf.sort_values("Volatility (Std Dev)", ascending=False).head(10)
    vol_columns = [{"name": "Stock", "id": "Stock"},
                   {"name": "Volatility", "id": "Volatility (Std Dev)"},
                   {"name": "Return %", "id": "Return %"}]

    # Suggestions table data
    sug_csv = "data:text/csv;base64," + base64.b64encode(SUG_DF.to_csv(index=False).encode()).decode()

    # Create charts for gainers and volatility
    gainers_chart = create_gainers_chart(gl.to_dict('records'))
    volatility_chart = create_volatility_chart(vol.to_dict('records'))

    return (
        line, candle, bar, scatter3d, sunburst, treemap, heatmap,
        gl.to_dict("records"), gl_columns,
        vol.to_dict("records"), vol_columns,
        SUG_DF.to_dict("records"), sug_csv,
        gainers_chart, volatility_chart
    )

@app.callback(
    Output("download", "data"),
    Input("btn-csv", "n_clicks"),
    Input("btn-xlsx", "n_clicks"),
    State("table-sug", "data"),
    prevent_initial_call=True
)
def download(csv_clicks, xlsx_clicks, data):
    if not data:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(data)
    if dash.ctx.triggered_id == "btn-csv":
        return dcc.send_data_frame(df.to_csv, "stock_data.csv", index=False)
    return dcc.send_data_frame(df.to_excel, "stock_data.xlsx", index=False)

# ------------------------------------------------------------------
# 6.  INLINE CSS  –  glassmorphism
# ------------------------------------------------------------------
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #e0f7fa 0%, #f5f7fa 100%);
            color: #222;
        }
        .main-container {
            min-height: 100vh;
            padding: 1rem 2rem;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 16px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            height: 100%; /* Ensure all cards have the same height */
            width: 100%; /* Ensure all cards have the same width */
        }
        .logo {
            font-weight: 700;
            font-size: 2rem;
            letter-spacing: 1px;
            color: #111;
        }
        .btn-outline-light {
            color: #333;
            border-color: rgba(0, 0, 0, 0.25);
        }
        .btn-outline-light:hover {
            background: rgba(0, 0, 0, 0.05);
        }
        hr {
            border-color: rgba(0, 0, 0, 0.08);
        }
        .Select-control, .Select-menu-outer {
            background: rgba(255, 255, 255, 0.65) !important;
            border: 1px solid rgba(0, 0, 0, 0.15);
            border-radius: 8px;
        }
        .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
            background-color: transparent;
        }
        .dash-spreadsheet .dash-header {
            background: rgba(255, 255, 255, 0.4);
        }
        .js-plotly-plot .plotly {
            background: transparent !important;
        }
        .js-plotly-plot .main-svg {
            background: transparent !important;
        }

        /* WCAG 2.2 Compliant Colors */
        .card-title { color: #1a3c5e; }
        .fw-bold { color: #0d2b45; }
        .text-muted { color: #5a6771; }

        /* Responsive Adjustments */
        @media (max-width: 1400px) {
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th,
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                padding: 4px 8px;
                font-size: 0.8rem;
            }
        }
        @media (max-width: 1200px) {
            .main-container { padding: 1rem 1.5rem; }
            .glass-card { height: auto; }
        }
        @media (max-width: 992px) {
            .main-container { padding: 0.75rem 1rem; }
            .glass-card { margin-bottom: 1rem; height: auto; }
        }
        @media (max-width: 768px) {
            .main-container { padding: 0.5rem 0.75rem; }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th,
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                padding: 2px 4px;
                font-size: 0.7rem;
            }
            .glass-card { height: auto; }
        }
        @media (max-width: 576px) {
            .main-container { padding: 0.25rem 0.5rem; }
            .logo { font-size: 1.5rem; }
            .glass-card { height: auto; }
        }

        /* Custom Scrollbar */
        .table-responsive::-webkit-scrollbar {
            height: 6px;
            width: 6px;
        }
        .table-responsive::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.05);
            border-radius: 3px;
        }
        .table-responsive::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
        }
        .table-responsive::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 0, 0, 0.3);
        }
    </style>
</head>
<body>
    {%app_entry%}
<footer>
        {%config%}
        {%scripts%}
        {%renderer%}
</footer>
</body>
</html>
'''

# ------------------------------------------------------------------
# Add this line after app initialization
server = app.server

# Replace the existing __main__ block with:
if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")
else:
    # This is needed for Vercel
    application = server