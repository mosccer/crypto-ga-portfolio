"""
Data Collector Module
Fetches historical OHLCV data from Binance via ccxt library
and computes daily returns for portfolio optimization.
"""

import ccxt
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from config import SYMBOLS, EXCHANGE, TIMEFRAME, DATA_LIMIT


def get_exchange():
    """Create and return a ccxt exchange instance."""
    exchange_class = getattr(ccxt, EXCHANGE)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    return exchange


def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=DATA_LIMIT):
    """
    Fetch OHLCV data for a single symbol from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candlestick timeframe (e.g., '1d')
        limit: Number of candles to fetch
        
    Returns:
        DataFrame with columns: time, open, high, low, close, volume
    """
    exchange = get_exchange()
    print(f"  📥 Fetching {symbol}...")
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('time')
    
    return df


def fetch_all_data(symbols=SYMBOLS, timeframe=TIMEFRAME, limit=DATA_LIMIT):
    """
    Fetch OHLCV data for all symbols.
    
    Returns:
        dict: {symbol: DataFrame}
    """
    data = {}
    for sym in symbols:
        data[sym] = fetch_ohlcv(sym, timeframe, limit)
    return data


def compute_returns(data, symbols=SYMBOLS):
    """
    Compute daily percentage returns from close prices.
    
    Args:
        data: dict of {symbol: DataFrame with 'close' column}
        symbols: list of symbols to include
        
    Returns:
        DataFrame of daily returns with symbols as columns
    """
    prices = pd.concat([data[s]['close'] for s in symbols], axis=1)
    prices.columns = symbols
    returns = prices.pct_change().dropna()
    return returns


def compute_prices(data, symbols=SYMBOLS):
    """
    Extract close prices for all symbols.
    
    Returns:
        DataFrame of close prices
    """
    prices = pd.concat([data[s]['close'] for s in symbols], axis=1)
    prices.columns = symbols
    prices = prices.dropna()
    return prices


def save_data_cache(data, cache_dir='data_cache'):
    """Save fetched data to local cache for offline use."""
    os.makedirs(cache_dir, exist_ok=True)
    
    for symbol, df in data.items():
        filename = symbol.replace('/', '_') + '.csv'
        filepath = os.path.join(cache_dir, filename)
        df.to_csv(filepath)
    
    # Save metadata
    meta = {
        'symbols': list(data.keys()),
        'fetched_at': datetime.now().isoformat(),
        'rows': {s: len(df) for s, df in data.items()}
    }
    with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  💾 Data cached to {cache_dir}/")


def load_data_cache(cache_dir='data_cache'):
    """Load data from local cache."""
    meta_path = os.path.join(cache_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return None
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    data = {}
    for symbol in meta['symbols']:
        filename = symbol.replace('/', '_') + '.csv'
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col='time', parse_dates=True)
            data[symbol] = df
    
    print(f"  📂 Loaded cached data (fetched: {meta['fetched_at']})")
    return data


def get_data(use_cache=True, force_refresh=False):
    """
    Main entry point for data collection.
    Uses cache if available, otherwise fetches from Binance.
    
    Args:
        use_cache: Whether to try loading from cache first
        force_refresh: Force re-fetch even if cache exists
        
    Returns:
        tuple: (returns DataFrame, prices DataFrame, raw data dict)
    """
    data = None
    
    if use_cache and not force_refresh:
        data = load_data_cache()
    
    if data is None:
        print("🌐 Fetching data from Binance...")
        data = fetch_all_data()
        if use_cache:
            save_data_cache(data)
    
    returns = compute_returns(data)
    prices = compute_prices(data)
    
    print(f"  ✅ Data ready: {len(returns)} days, {len(SYMBOLS)} assets")
    return returns, prices, data


if __name__ == '__main__':
    returns, prices, data = get_data(force_refresh=True)
    print("\n📊 Returns Statistics:")
    print(returns.describe())
    print("\n📈 Latest Prices:")
    print(prices.tail())
