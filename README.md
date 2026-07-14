# 🏦 Institutional Crypto Quant Framework

**MoE AI & Deterministic Execution**

An asynchronous, multi-agent cryptocurrency trading architecture that separates deterministic order-flow mathematics from AI-driven macro reasoning.

Traditional trading bots that use Large Language Models (LLMs) for direct numerical time-series forecasting fail due to the chaotic and adversarial nature of asset prices. This system solves that with a strict **separation of concerns**: high-frequency, deterministic order-flow math runs in pure Python, while non-deterministic macro-synthesis and reasoning are offloaded to an adversarial Mixture-of-Experts (MoE) LLM cluster.

---

## 🏗️ Core Architecture

The system is divided into four fully decoupled, event-driven microservices that communicate over a zero-latency Redis message bus.

### 1. The Nervous System — Data & Microstructure
A flawless, zero-latency data ingestion pipeline that feeds everything downstream.
- **Async WebSockets** — native `asyncio` connections to exchange L2/L3 data feeds for live trades
- **Lee-Ready Tick Test** — since Level 2 feeds don't explicitly tag aggressors, this algorithm classifies aggressive buyers vs. sellers
- **Order Flow Mathematics** — real-time Cumulative Volume Delta (CVD) calculations to detect passive liquidity absorption
- **State Management** — high-frequency data states are pushed continuously to a local Redis server using Hashes and Sorted Sets

### 2. The Vectorized TA Engine — The Senses
LLMs suffer from "number blindness" and hallucinate when fed raw floating-point arrays, so this engine translates market data into language the AI can reason about.
- **Multi-Timeframe Matrix** — a cron-scheduled service uses `pandas-ta` to compute indicators across 1m, 5m, 15m, 1h, and 4h timeframes
- **Synthetic Dollar Bars** — normalizes volatility by slicing candles on traded notional value (e.g., $1M exchanged) rather than fixed time intervals
- **Semantic Tagging** — converts raw indicator math into geometric state vectors (e.g., `DEEP_PULLBACK`, `OVEREXTENDED`) to eliminate LLM mathematical hallucination

### 3. The MoE Brain — AI Orchestrator
A localized "board of directors" built on an adversarial debate topology.
- **Pipeline A (Alpha Hunter)** — when flat, a Bull Agent and Bear Agent run a 3-round debate over current market context; a Master LLM synthesizes the debate into a directional Conviction Score and a Semantic Invalidation Condition
- **Pipeline B (Thesis Validator)** — when a position is open, the action space shifts to `[Hold, Trim, Abort]`; the AI continuously audits the original thesis against live data to detect macro decay

### 4. The Muscle — Deterministic Risk Engine
The LLM is the strategist, but the Python risk engine is the executioner — the AI never has direct access to capital sizing.
- **Fractional Kelly Sizing** — maps the AI's conviction score to a Quarter-Kelly allocation
- **Volatility Scaling** — allocation is inversely scaled against the real-time 14-period ATR to keep dollar-risk flat
- **Expected Shortfall (CVaR)** — computes worst-1% tail risk and vetoes any trade that breaches the global drawdown limit
- **Toxic Flow Evacuation (VPIN)** — triggers an emergency market sell, overriding standard stops, if a severe bearish sweep (distribution) is detected
- **Hybrid Exits** — at +2x ATR, mechanically banks 50% profit and moves the stop to breakeven, leaving the remaining 50% ("the runner") for the AI to manage

---

## 🛠️ Installation & Setup

This project uses **Poetry** for strict dependency locking and **Docker** for zero-latency state management.

### Prerequisites
- Python >= 3.12
- Poetry
- Docker Desktop

### 1. Initialize the workspace

```bash
git clone https://github.com/yourusername/btc-hedge-fund.git
cd btc-hedge-fund
python -m poetry install
```

### 2. Deploy the infrastructure layer

Spin up the local Redis memory bus used for high-frequency CVD arrays:

```bash
docker run --name btc-hedge-redis -p 6379:6379 -d redis:alpine
```

### 3. Environment configuration

Create a `.env` file inside `config/` with your exchange and LLM API keys:

```ini
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
GEMINI_API_KEY=your_ai_key
REDIS_URL=redis://localhost:6379
```

---

## 🚀 Running the System

Monitoring real-time data, computing math, compressing context, managing risk, and running the AI all happen concurrently. Open five terminal tabs and run:

| # | Service | Command |
|---|---------|---------|
| 1 | Order Flow Stream | `python -m poetry run python src/data_ingestion/nervous_system.py` |
| 2 | Vectorized Math Engine | `python -m poetry run python src/data_ingestion/ta_engine.py` |
| 3 | Risk Firewall | `python -m poetry run python src/execution/order_router.py` |
| 4 | Schema Compressor | `python -m poetry run python src/reasoning/context_builder.py` |
| 5 | MoE AI Brain | `python -m poetry run python src/reasoning/ai_orchestrator.py` |

> **Note:** In production (Sprint 5), these services are tied together via a Master Process Manager (`main.py`) or an async task broker such as Celery.

---

## 🔮 Future Roadmap: Knowledge Distillation

Relying indefinitely on commercial cloud APIs represents a systemic platform vulnerability. Once the paper-trading validation phase generates a large telemetry database (10,000+ logged rows) of profitable AI reasoning paths, the pipeline will transition to knowledge distillation:

- **QLoRA** (Parameter-Efficient Fine-Tuning) to distill the reasoning abstractions of the cloud-based "Teacher" model
- Into an ultra-low-latency, localized 8B-parameter "Student" model (e.g., Llama 3 8B) running on dedicated GPU hardware

---

## ⚠️ Disclaimer

This system is an open-source proof-of-concept for **educational and research purposes only**. Cryptocurrency markets are highly volatile and inherently adversarial. The creator assumes no liability for financial losses. Do not deploy live capital without extensive forward paper-trading.
