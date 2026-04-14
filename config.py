"""
Configuration for Crypto GA Portfolio Optimizer
"""

# ===== Crypto Assets =====
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
ASSET_NAMES = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA']

# ===== Data Settings =====
EXCHANGE = 'binance'
TIMEFRAME = '1d'
DATA_LIMIT = 365  # days of historical data
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate

# ===== GA Parameters =====
POPULATION_SIZE = 200
NUM_GENERATIONS = 300
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
MUTATION_SCALE = 0.05  # std dev for gaussian mutation
TOURNAMENT_SIZE = 5
ELITISM_COUNT = 10  # top N individuals preserved each generation

# ===== Portfolio Constraints =====
MAX_WEIGHT = 0.40   # max 40% per asset (prevent all-in)
MIN_WEIGHT = 0.05   # min 5% per asset (ensure diversification)

# ===== Fitness Function Weights =====
DRAWDOWN_PENALTY = 0.5  # penalty multiplier for max drawdown
CONSTRAINT_PENALTY = 10.0  # penalty for violating constraints

# ===== Backtesting =====
INITIAL_CAPITAL = 10000  # USD
REBALANCE_FREQUENCY = 30  # days
