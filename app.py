"""
🚀 Crypto GA Portfolio Optimizer Dashboard
Beautiful Streamlit dashboard for genetic algorithm portfolio optimization.

Architecture: Binance API → Data Collector → GA Optimizer → Backtester → [Dashboard]

Deployable web application supporting:
- Streamlit Cloud
- Docker (any cloud: AWS, GCP, Azure, DigitalOcean)
- Heroku / Render / Railway
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
import os
from config import (
    SYMBOLS, ASSET_NAMES, POPULATION_SIZE, NUM_GENERATIONS,
    CROSSOVER_RATE, MUTATION_RATE, MAX_WEIGHT, MIN_WEIGHT,
    INITIAL_CAPITAL, RISK_FREE_RATE
)
from data_collector import get_data
from ga_engine import optimize_portfolio
from backtester import run_backtest, compare_with_equal_weight

# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Crypto GA Portfolio Optimizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6fd8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: rgba(255,255,255,0.7);
        margin: 0.5rem 0 0;
        font-size: 0.95rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-value {
        color: #fff;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-value.positive { color: #00e676; }
    .metric-value.negative { color: #ff5252; }
    .metric-value.neutral { color: #7c4dff; }
    
    /* Weight allocation bar */
    .weight-bar {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .weight-bar .asset-name {
        font-weight: 600;
        color: #fff;
        font-size: 0.9rem;
        min-width: 50px;
    }
    .weight-bar .weight-pct {
        color: #7c4dff;
        font-weight: 700;
        font-size: 1.1rem;
        min-width: 60px;
        text-align: right;
    }
    
    /* Section divider */
    .section-title {
        color: #fff;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(124, 77, 255, 0.3);
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-running {
        background: rgba(255, 167, 38, 0.2);
        color: #ffa726;
        border: 1px solid rgba(255, 167, 38, 0.3);
    }
    .status-complete {
        background: rgba(0, 230, 118, 0.2);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #fff;
        font-size: 1.1rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark theme plotly charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════

CHART_TEMPLATE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,26,46,0.6)',
    font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor='rgba(26,26,46,0.95)', font_size=13, font_family='Inter'),
)

ASSET_COLORS = {
    'BTC': '#F7931A',
    'ETH': '#627EEA',
    'SOL': '#9945FF',
    'BNB': '#F3BA2F',
    'ADA': '#0033AD',
    'BTC/USDT': '#F7931A',
    'ETH/USDT': '#627EEA',
    'SOL/USDT': '#9945FF',
    'BNB/USDT': '#F3BA2F',
    'ADA/USDT': '#0033AD',
}


def create_donut_chart(weights, labels, title="Portfolio Allocation"):
    """Create a beautiful donut chart for portfolio weights."""
    colors = [ASSET_COLORS.get(l, '#7c4dff') for l in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=weights * 100,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='rgba(26,26,46,1)', width=3)),
        textinfo='label+percent',
        textfont=dict(size=14, family='Inter', color='white'),
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>',
        pull=[0.03] * len(labels),
    )])
    
    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text=title, font=dict(size=16, color='white')),
        showlegend=False,
        height=350,
        annotations=[dict(
            text='<b>GA<br>Optimized</b>',
            x=0.5, y=0.5, font_size=14,
            font_color='rgba(255,255,255,0.5)',
            showarrow=False, font_family='Inter'
        )]
    )
    return fig


def create_performance_chart(opt_values, bench_values, title="Portfolio Performance"):
    """Create a dual-line chart comparing optimized vs benchmark."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=opt_values.index, y=opt_values.values,
        name='GA Optimized',
        line=dict(color='#7c4dff', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(124, 77, 255, 0.08)',
        hovertemplate='<b>GA Optimized</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=bench_values.index, y=bench_values.values,
        name='Equal Weight',
        line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dot'),
        hovertemplate='<b>Equal Weight</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text=title, font=dict(size=16, color='white')),
        height=400,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=12, color='rgba(255,255,255,0.7)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        yaxis_title='Portfolio Value (USD)',
    )
    return fig


def create_fitness_chart(best_history, avg_history, diversity_history):
    """Create a multi-axis chart showing GA convergence."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=['Fitness Evolution', 'Population Diversity'],
        row_heights=[0.65, 0.35]
    )
    
    gens = list(range(len(best_history)))
    
    # Best fitness
    fig.add_trace(go.Scatter(
        x=gens, y=best_history,
        name='Best Fitness',
        line=dict(color='#00e676', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 230, 118, 0.05)',
    ), row=1, col=1)
    
    # Average fitness
    fig.add_trace(go.Scatter(
        x=gens, y=avg_history,
        name='Avg Fitness',
        line=dict(color='rgba(255,167,38,0.7)', width=1.5, dash='dot'),
    ), row=1, col=1)
    
    # Diversity
    fig.add_trace(go.Scatter(
        x=gens, y=diversity_history,
        name='Diversity',
        line=dict(color='#ff6fd8', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255, 111, 216, 0.05)',
    ), row=2, col=1)
    
    fig.update_layout(
        **CHART_TEMPLATE,
        height=450,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=11, color='rgba(255,255,255,0.7)'),
            bgcolor='rgba(0,0,0,0)'
        ),
    )
    fig.update_xaxes(title_text='Generation', row=2, col=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=2, col=1)
    
    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color='rgba(255,255,255,0.7)', family='Inter')
    
    return fig


def create_drawdown_chart(portfolio_values):
    """Create a drawdown visualization chart."""
    cumulative = portfolio_values / portfolio_values.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (running_max - cumulative) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=-drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.15)',
        line=dict(color='#ff5252', width=1.5),
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text='Drawdown Analysis', font=dict(size=16, color='white')),
        height=280,
        yaxis_title='Drawdown (%)',
        showlegend=False
    )
    return fig


def create_correlation_heatmap(returns_df):
    """Create a correlation heatmap of asset returns."""
    corr = returns_df.corr()
    labels = [s.replace('/USDT', '') for s in corr.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[
            [0, '#ff5252'],
            [0.5, '#1a1a2e'],
            [1, '#00e676']
        ],
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=14, color='white'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text='Corr', font=dict(color='rgba(255,255,255,0.7)')),
            tickfont=dict(color='rgba(255,255,255,0.5)')
        )
    ))
    
    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text='Asset Correlation Matrix', font=dict(size=16, color='white')),
        height=380,
    )
    return fig


def create_returns_distribution(daily_returns):
    """Create a histogram of daily portfolio returns."""
    returns_pct = daily_returns * 100
    
    fig = go.Figure()
    
    # Negative returns
    neg_mask = returns_pct < 0
    fig.add_trace(go.Histogram(
        x=returns_pct[neg_mask],
        name='Loss Days',
        marker_color='rgba(255, 82, 82, 0.6)',
        nbinsx=40,
    ))
    
    # Positive returns
    pos_mask = returns_pct >= 0
    fig.add_trace(go.Histogram(
        x=returns_pct[pos_mask],
        name='Profit Days',
        marker_color='rgba(0, 230, 118, 0.6)',
        nbinsx=40,
    ))
    
    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text='Daily Returns Distribution', font=dict(size=16, color='white')),
        height=300,
        barmode='overlay',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=11, color='rgba(255,255,255,0.7)'),
            bgcolor='rgba(0,0,0,0)'
        ),
    )
    return fig


def create_asset_performance_chart(prices_df):
    """Create normalized price performance chart for each asset."""
    fig = go.Figure()
    
    normalized = prices_df / prices_df.iloc[0] * 100
    
    for col in normalized.columns:
        name = col.replace('/USDT', '')
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col].values,
            name=name,
            line=dict(color=ASSET_COLORS.get(col, ASSET_COLORS.get(name, '#7c4dff')), width=2),
            hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Performance: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        **CHART_TEMPLATE,
        title=dict(text='Individual Asset Performance (Normalized)', font=dict(size=16, color='white')),
        height=380,
        yaxis_title='Performance (Base=100)',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=12, color='rgba(255,255,255,0.8)'),
            bgcolor='rgba(0,0,0,0)'
        ),
    )
    return fig


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧬 GA Parameters")
    
    pop_size = st.slider("Population Size", 50, 500, POPULATION_SIZE, 10)
    num_gens = st.slider("Generations", 50, 1000, NUM_GENERATIONS, 50)
    cross_rate = st.slider("Crossover Rate", 0.5, 1.0, CROSSOVER_RATE, 0.05)
    mut_rate = st.slider("Mutation Rate", 0.01, 0.50, MUTATION_RATE, 0.01)
    
    st.markdown("## 📊 Portfolio Constraints")
    max_w = st.slider("Max Weight per Asset", 0.2, 1.0, MAX_WEIGHT, 0.05)
    min_w = st.slider("Min Weight per Asset", 0.0, 0.2, MIN_WEIGHT, 0.01)
    
    st.markdown("## 💰 Backtest Settings")
    init_capital = st.number_input("Initial Capital ($)", 1000, 1000000, INITIAL_CAPITAL, 1000)
    risk_free = st.slider("Risk-Free Rate (%)", 0.0, 10.0, RISK_FREE_RATE * 100, 0.5) / 100
    
    st.markdown("## 📡 Data")
    force_refresh = st.checkbox("Force Refresh Data", value=False)
    
    st.markdown("---")
    run_button = st.button("🚀 Run Optimization", use_container_width=True, type="primary")

# ══════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 Crypto GA Portfolio Optimizer</h1>
    <p>Genetic Algorithm-based portfolio optimization for cryptocurrency assets — powered by real Binance market data</p>
</div>
""", unsafe_allow_html=True)

# ===== Data Loading =====
@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(force=False):
    return get_data(use_cache=True, force_refresh=force)

# Architecture diagram
with st.expander("📐 System Architecture", expanded=False):
    st.markdown("""
    ```
    Binance API → Data Collector → GA Optimizer → Backtester → Dashboard
         │              │                │              │           │
     ccxt lib      fetch OHLCV    Sharpe + MDD     Simulate    Streamlit
                    365 days      Tournament Sel.  Rebalance    Plotly
                   5 assets      BLX-α Crossover   Metrics     Charts
                                 Gaussian Mut.
    ```
    
    **Fitness Function**: `fitness = Sharpe - 0.5 × MaxDrawdown - penalty(constraints)`
    
    **Constraints**: `Σ weights = 1`, `weight_i ≤ 40%`, `weight_i ≥ 5%`
    """)

# Show data section before optimization
try:
    with st.spinner("📡 Loading market data from Binance..."):
        returns_df, prices_df, raw_data = load_market_data(force_refresh)
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Trading Days</div>
            <div class="metric-value neutral">{len(returns_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Assets</div>
            <div class="metric-value neutral">{len(SYMBOLS)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        date_range = f"{returns_df.index[0].strftime('%b %Y')} — {returns_df.index[-1].strftime('%b %Y')}"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Date Range</div>
            <div class="metric-value neutral" style="font-size:1rem;">{date_range}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Asset performance  
    st.plotly_chart(create_asset_performance_chart(prices_df), use_container_width=True)
    
    # Correlation matrix
    st.plotly_chart(create_correlation_heatmap(returns_df), use_container_width=True)
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("💡 Make sure you have internet access and the `ccxt` library installed. Run: `pip install -r requirements.txt`")
    st.stop()

# ===== Run GA Optimization =====
if run_button:
    st.markdown('<div class="section-title">🧬 Running Genetic Algorithm</div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # GA callback for progress
    generation_data = {'best': [], 'avg': [], 'div': []}
    
    def ga_callback(gen, best, avg_fitness, diversity):
        progress = (gen + 1) / num_gens
        progress_bar.progress(progress)
        if gen % 10 == 0:
            status_text.markdown(
                f'<span class="status-badge status-running">⏳ Generation {gen+1}/{num_gens}</span> '
                f'&nbsp; Best Fitness: **{best.fitness:.4f}** &nbsp; | &nbsp; '
                f'Sharpe: **{best.sharpe:.3f}** &nbsp; | &nbsp; '
                f'MaxDD: **{best.max_drawdown:.2%}**',
                unsafe_allow_html=True
            )
    
    # Run optimization
    start_time = time.time()
    
    ga_result = optimize_portfolio(
        returns_df, SYMBOLS,
        callback=ga_callback,
        population_size=pop_size,
        num_generations=num_gens,
        crossover_rate=cross_rate,
        mutation_rate=mut_rate,
        max_weight=max_w,
        min_weight=min_w,
        risk_free_rate=risk_free,
    )
    
    elapsed = time.time() - start_time
    
    progress_bar.progress(1.0)
    status_text.markdown(
        f'<span class="status-badge status-complete">✅ Complete</span> '
        f'&nbsp; {num_gens} generations in **{elapsed:.1f}s**',
        unsafe_allow_html=True
    )
    
    best = ga_result.best_individual
    weights = best.weights
    
    # Store results in session state
    st.session_state['ga_result'] = ga_result
    st.session_state['weights'] = weights
    st.session_state['elapsed'] = elapsed

# ===== Display Results =====
if 'ga_result' in st.session_state:
    ga_result = st.session_state['ga_result']
    weights = st.session_state['weights']
    elapsed = st.session_state['elapsed']
    best = ga_result.best_individual
    
    # ── Optimized Weights ──
    st.markdown('<div class="section-title">🎯 Optimal Portfolio Allocation</div>', unsafe_allow_html=True)
    
    col_donut, col_weights = st.columns([1, 1])
    
    with col_donut:
        fig_donut = create_donut_chart(weights, ASSET_NAMES, "GA-Optimized Allocation")
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col_weights:
        st.markdown("<br>", unsafe_allow_html=True)
        for i, (name, w) in enumerate(sorted(zip(ASSET_NAMES, weights), key=lambda x: -x[1])):
            color = ASSET_COLORS.get(name, '#7c4dff')
            pct = w * 100
            bar_width = pct / max(weights) * 100
            st.markdown(f"""
            <div class="weight-bar">
                <span class="asset-name" style="color: {color};">● {name}</span>
                <div style="flex:1; margin: 0 1rem; background: rgba(255,255,255,0.05); border-radius: 4px; height: 8px; overflow:hidden;">
                    <div style="width: {bar_width}%; height: 100%; background: linear-gradient(90deg, {color}, {color}88); border-radius: 4px;"></div>
                </div>
                <span class="weight-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align: right; margin-top: 0.5rem; color: rgba(255,255,255,0.4); font-size: 0.8rem;">
            Total: {np.sum(weights)*100:.1f}% &nbsp;|&nbsp; Fitness: {best.fitness:.4f}
        </div>
        """, unsafe_allow_html=True)
    
    # ── GA Convergence ──
    st.markdown('<div class="section-title">📈 GA Evolution & Convergence</div>', unsafe_allow_html=True)
    
    fig_fitness = create_fitness_chart(
        ga_result.best_fitness_history,
        ga_result.avg_fitness_history,
        ga_result.diversity_history
    )
    st.plotly_chart(fig_fitness, use_container_width=True)
    
    # ── Backtesting ──
    st.markdown('<div class="section-title">📊 Backtest Results</div>', unsafe_allow_html=True)
    
    with st.spinner("Running backtest..."):
        opt_result, bench_result = compare_with_equal_weight(returns_df, weights, init_capital)
    
    # Key metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics = [
        ("Total Return", f"{opt_result.total_return:+.1%}", "positive" if opt_result.total_return > 0 else "negative"),
        ("Annual Return", f"{opt_result.annual_return:+.1%}", "positive" if opt_result.annual_return > 0 else "negative"),
        ("Sharpe Ratio", f"{opt_result.sharpe_ratio:.3f}", "positive" if opt_result.sharpe_ratio > 0 else "negative"),
        ("Max Drawdown", f"{opt_result.max_drawdown:.1%}", "negative"),
        ("Sortino Ratio", f"{opt_result.sortino_ratio:.3f}", "positive" if opt_result.sortino_ratio > 0 else "negative"),
        ("Win Rate", f"{opt_result.win_rate:.1%}", "positive" if opt_result.win_rate > 0.5 else "negative"),
    ]
    
    for col, (label, value, cls) in zip([col1, col2, col3, col4, col5, col6], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {cls}">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance chart
    fig_perf = create_performance_chart(
        opt_result.portfolio_values,
        bench_result.portfolio_values,
        f"Portfolio Performance (${init_capital:,.0f} initial)"
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Drawdown chart
    col_dd, col_dist = st.columns(2)
    
    with col_dd:
        fig_dd = create_drawdown_chart(opt_result.portfolio_values)
        st.plotly_chart(fig_dd, use_container_width=True)
    
    with col_dist:
        fig_dist = create_returns_distribution(opt_result.daily_returns)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # ── Comparison Table ──
    st.markdown('<div class="section-title">⚖️ GA Optimized vs Equal Weight</div>', unsafe_allow_html=True)
    
    comparison_data = {
        'Metric': ['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio',
                    'Max Drawdown', 'Calmar Ratio', 'Win Rate', 'VaR (95%)', 'CVaR (95%)',
                    'Best Day', 'Worst Day'],
        'GA Optimized': [
            f"{opt_result.total_return:+.2%}", f"{opt_result.annual_return:+.2%}",
            f"{opt_result.annual_volatility:.2%}", f"{opt_result.sharpe_ratio:.3f}",
            f"{opt_result.sortino_ratio:.3f}", f"{opt_result.max_drawdown:.2%}",
            f"{opt_result.calmar_ratio:.3f}", f"{opt_result.win_rate:.1%}",
            f"{opt_result.var_95:.2%}", f"{opt_result.cvar_95:.2%}",
            f"{opt_result.best_day:+.2%}", f"{opt_result.worst_day:+.2%}",
        ],
        'Equal Weight': [
            f"{bench_result.total_return:+.2%}", f"{bench_result.annual_return:+.2%}",
            f"{bench_result.annual_volatility:.2%}", f"{bench_result.sharpe_ratio:.3f}",
            f"{bench_result.sortino_ratio:.3f}", f"{bench_result.max_drawdown:.2%}",
            f"{bench_result.calmar_ratio:.3f}", f"{bench_result.win_rate:.1%}",
            f"{bench_result.var_95:.2%}", f"{bench_result.cvar_95:.2%}",
            f"{bench_result.best_day:+.2%}", f"{bench_result.worst_day:+.2%}",
        ]
    }
    
    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
    
    # ── Top Portfolios from Final Population ──
    st.markdown('<div class="section-title">🏆 Top 10 Portfolios from Final Population</div>', unsafe_allow_html=True)
    
    top_data = []
    for i, ind in enumerate(ga_result.final_population[:10]):
        row = {'Rank': f"#{i+1}", 'Fitness': f"{ind.fitness:.4f}",
               'Sharpe': f"{ind.sharpe:.3f}", 'Return': f"{ind.annual_return:+.1%}",
               'Volatility': f"{ind.annual_volatility:.1%}", 'MaxDD': f"{ind.max_drawdown:.1%}"}
        for j, name in enumerate(ASSET_NAMES):
            row[name] = f"{ind.weights[j]:.1%}"
        top_data.append(row)
    
    st.dataframe(pd.DataFrame(top_data), use_container_width=True, hide_index=True)
    
    # ── Runtime Info ──
    st.markdown(f"""
    <div style="text-align:center; color:rgba(255,255,255,0.3); font-size:0.8rem; margin-top:2rem; padding:1rem;">
        🧬 GA completed {ga_result.generations_run} generations with population size {pop_size} in {elapsed:.1f}s &nbsp;|&nbsp;
        Crossover: {cross_rate:.0%} &nbsp;|&nbsp; Mutation: {mut_rate:.0%} &nbsp;|&nbsp;
        Constraints: [{min_w:.0%}, {max_w:.0%}]
    </div>
    """, unsafe_allow_html=True)

else:
    # No results yet - show instructions
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 2rem; color: rgba(255,255,255,0.5);">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🧬</div>
        <h3 style="color: rgba(255,255,255,0.7); font-weight: 600;">Ready to Optimize</h3>
        <p>Configure GA parameters in the sidebar, then click <b>🚀 Run Optimization</b> to find the optimal portfolio allocation.</p>
        <p style="font-size: 0.85rem; margin-top: 1rem;">
            The algorithm will evolve {num_gens} generations of {pop_size} portfolios,<br>
            optimizing for risk-adjusted returns across {len(SYMBOLS)} crypto assets.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem; margin-top:3rem; padding:1.5rem; border-top: 1px solid rgba(255,255,255,0.05);">
    🧬 Crypto GA Portfolio Optimizer &nbsp;•&nbsp; Powered by Genetic Algorithm &nbsp;•&nbsp; Data from Binance API<br>
    Built with Streamlit, Plotly, ccxt &nbsp;•&nbsp; © 2026
</div>
""", unsafe_allow_html=True)
