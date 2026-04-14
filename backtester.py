"""
Backtester Module
Simulates portfolio performance using historical data
and the GA-optimized weights.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from config import INITIAL_CAPITAL, REBALANCE_FREQUENCY


@dataclass
class BacktestResult:
    """Results from backtesting a portfolio."""
    portfolio_values: pd.Series = None
    daily_returns: pd.Series = None
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    weights_over_time: pd.DataFrame = None


def run_backtest(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
    rebalance_days: int = REBALANCE_FREQUENCY,
    risk_free_rate: float = 0.04
) -> BacktestResult:
    """
    Run a comprehensive backtest of the portfolio.
    
    Args:
        returns_df: DataFrame of daily returns (dates x assets)
        weights: Optimized portfolio weights
        initial_capital: Starting capital in USD
        rebalance_days: How often to rebalance (in days)
        risk_free_rate: Annual risk-free rate
        
    Returns:
        BacktestResult with all performance metrics
    """
    result = BacktestResult()
    
    n_days = len(returns_df)
    n_assets = len(weights)
    
    # ===== Simulate with rebalancing =====
    portfolio_value = initial_capital
    current_weights = weights.copy()
    values = [portfolio_value]
    
    # Track weights over time
    weight_records = [{'date': returns_df.index[0], **{returns_df.columns[i]: weights[i] for i in range(n_assets)}}]
    
    for day in range(n_days):
        daily_ret = returns_df.iloc[day].values
        
        # Portfolio return for this day
        port_ret = np.sum(current_weights * daily_ret)
        portfolio_value *= (1 + port_ret)
        values.append(portfolio_value)
        
        # Update weights due to price changes (drift)
        current_weights = current_weights * (1 + daily_ret)
        total = np.sum(current_weights)
        if total > 0:
            current_weights = current_weights / total
        
        # Rebalance periodically
        if (day + 1) % rebalance_days == 0:
            current_weights = weights.copy()
            weight_records.append({
                'date': returns_df.index[day],
                **{returns_df.columns[i]: weights[i] for i in range(n_assets)}
            })
    
    # ===== Compute metrics =====
    dates = [returns_df.index[0] - pd.Timedelta(days=1)] + list(returns_df.index)
    portfolio_series = pd.Series(values, index=dates[:len(values)])
    daily_returns = portfolio_series.pct_change().dropna()
    
    # Total and annualized return
    total_return = (portfolio_value - initial_capital) / initial_capital
    n_years = n_days / 365
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    
    # Volatility
    annual_volatility = daily_returns.std() * np.sqrt(365)
    
    # Sharpe Ratio
    if annual_volatility > 0:
        sharpe = (annual_return - risk_free_rate) / annual_volatility
    else:
        sharpe = 0.0
    
    # Max Drawdown
    cumulative = portfolio_series / portfolio_series.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = drawdown.max()
    
    # Max Drawdown Duration
    is_dd = drawdown > 0
    dd_groups = (~is_dd).cumsum()
    dd_durations = is_dd.groupby(dd_groups).sum()
    max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0
    
    # Calmar Ratio
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    
    # Sortino Ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 1e-10
    sortino = (annual_return - risk_free_rate) / downside_std
    
    # Win Rate
    win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0.0
    
    # VaR and CVaR (95%)
    var_95 = np.percentile(daily_returns, 5)
    cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else var_95
    
    # ===== Populate result =====
    result.portfolio_values = portfolio_series
    result.daily_returns = daily_returns
    result.total_return = total_return
    result.annual_return = annual_return
    result.annual_volatility = annual_volatility
    result.sharpe_ratio = sharpe
    result.max_drawdown = max_drawdown
    result.max_drawdown_duration = max_dd_duration
    result.calmar_ratio = calmar
    result.sortino_ratio = sortino
    result.win_rate = win_rate
    result.best_day = daily_returns.max()
    result.worst_day = daily_returns.min()
    result.var_95 = var_95
    result.cvar_95 = cvar_95
    result.weights_over_time = pd.DataFrame(weight_records)
    
    return result


def compare_with_equal_weight(returns_df, optimized_weights, initial_capital=INITIAL_CAPITAL):
    """
    Compare GA-optimized portfolio with equal-weight benchmark.
    
    Returns:
        tuple: (optimized_result, equal_weight_result)
    """
    n_assets = len(optimized_weights)
    equal_weights = np.ones(n_assets) / n_assets
    
    optimized = run_backtest(returns_df, optimized_weights, initial_capital)
    benchmark = run_backtest(returns_df, equal_weights, initial_capital)
    
    return optimized, benchmark


def compare_with_btc_only(returns_df, optimized_weights, initial_capital=INITIAL_CAPITAL):
    """
    Compare GA-optimized portfolio with BTC-only allocation.
    Note: This uses the first asset as BTC (convention).
    """
    n_assets = returns_df.shape[1]
    btc_weights = np.zeros(n_assets)
    btc_weights[0] = 1.0  # All in BTC
    
    optimized = run_backtest(returns_df, optimized_weights, initial_capital)
    btc_only = run_backtest(returns_df, btc_weights, initial_capital)
    
    return optimized, btc_only
