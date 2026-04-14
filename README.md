# 🧬 Crypto GA Portfolio Optimizer

A **deployable web application** that uses **Genetic Algorithms (GA)** to optimize cryptocurrency portfolio allocation. Built with Streamlit, powered by real-time Binance market data.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

```
Binance API → Data Collector → GA Optimizer → Backtester → Dashboard
     │              │                │              │           │
 ccxt lib      fetch OHLCV    Sharpe + MDD     Simulate    Streamlit
               365 days      Tournament Sel.  Rebalance    Plotly
              5 assets      BLX-α Crossover   Metrics     Charts
                            Gaussian Mut.
```

## ✨ Features

- **Genetic Algorithm Optimization** — Real-valued chromosome encoding with BLX-α crossover and Gaussian mutation
- **5 Crypto Assets** — BTC, ETH, SOL, BNB, ADA portfolio allocation
- **Fitness Function** — `Sharpe - 0.5 × MaxDrawdown - penalty(constraints)`
- **Interactive Dashboard** — Beautiful dark-themed charts with Plotly
- **Backtesting Engine** — Comprehensive metrics (Sharpe, Sortino, Calmar, VaR, CVaR)
- **Real-time Data** — Fetches from Binance via ccxt with local caching

---

## 🚀 Quick Start (Local)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/crypto-ga-portfolio.git
cd crypto-ga-portfolio

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (ง่ายที่สุด / Easiest)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set:
   - **Main file**: `app.py`
   - **Python version**: 3.11
5. Click **Deploy** ✅

> Streamlit Cloud is **free** for public apps!

### Option 2: Render

1. Push code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Deploy ✅

### Option 3: Docker (Any Cloud)

```bash
# Build
docker build -t crypto-ga-portfolio .

# Run locally
docker run -p 8501:8501 crypto-ga-portfolio

# Push to registry (example: Docker Hub)
docker tag crypto-ga-portfolio YOUR_USERNAME/crypto-ga-portfolio
docker push YOUR_USERNAME/crypto-ga-portfolio
```

Then deploy the Docker image to:
- **AWS** (ECS, App Runner, Lightsail)
- **GCP** (Cloud Run, GKE)
- **Azure** (Container Apps, ACI)
- **DigitalOcean** (App Platform)

### Option 4: Heroku

```bash
# Login
heroku login

# Create app
heroku create crypto-ga-portfolio

# Deploy
git push heroku main
```

### Option 5: Railway

1. Go to [railway.app](https://railway.app)
2. **New Project** → **Deploy from GitHub repo**
3. It auto-detects the `Procfile`
4. Deploy ✅

---

## 📁 Project Structure

```
crypto-ga-portfolio/
├── app.py                 # 🎨 Streamlit dashboard (main entry)
├── config.py              # ⚙️ GA parameters & settings
├── ga_engine.py           # 🧬 Genetic Algorithm engine
├── data_collector.py      # 📡 Binance data fetcher (ccxt)
├── backtester.py          # 📊 Portfolio backtester
├── requirements.txt       # 📦 Python dependencies
├── Dockerfile             # 🐳 Docker deployment
├── Procfile               # 🚂 Heroku/Render deployment
├── runtime.txt            # 🐍 Python version spec
├── setup.sh               # 🔧 Streamlit Cloud setup
├── .streamlit/
│   └── config.toml        # 🎨 Streamlit theme & server config
├── .gitignore             # Git ignore rules
├── .dockerignore          # Docker ignore rules
├── data_cache/            # 💾 Cached market data
└── README.md              # 📖 This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOLS` | BTC, ETH, SOL, BNB, ADA | Crypto assets to optimize |
| `POPULATION_SIZE` | 200 | GA population size |
| `NUM_GENERATIONS` | 300 | Number of GA generations |
| `CROSSOVER_RATE` | 0.85 | BLX-α crossover probability |
| `MUTATION_RATE` | 0.15 | Gaussian mutation probability |
| `MAX_WEIGHT` | 0.40 | Max weight per asset (40%) |
| `MIN_WEIGHT` | 0.05 | Min weight per asset (5%) |
| `INITIAL_CAPITAL` | $10,000 | Backtest starting capital |

---

## 📊 Performance Metrics

The backtester computes:
- **Total & Annual Return**
- **Sharpe & Sortino Ratio**
- **Maximum Drawdown & Duration**
- **Calmar Ratio**
- **Value at Risk (VaR 95%)**
- **Conditional VaR (CVaR 95%)**
- **Win Rate, Best/Worst Day**

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit + Custom CSS |
| Charts | Plotly |
| GA Engine | NumPy (custom implementation) |
| Data | ccxt (Binance API) |
| Backtest | Pandas + NumPy |
| Deploy | Docker / Streamlit Cloud / Render |

---

## 📄 License

MIT License © 2026
