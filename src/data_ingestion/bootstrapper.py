import asyncio
import aiohttp
import socket
import time
import os
import json
import pandas as pd
import pandas_ta as ta
import numpy as np
from redis.asyncio import Redis
from dotenv import load_dotenv

# Load environment configuration
load_dotenv(dotenv_path="config/.env")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"

class MarketBootstrapper:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.symbol = "BTCUSDT"
        self.dollar_bar_threshold = 1_000_000.0  # $1M per bar (must match nervous_system.py)
        
    async def fetch_klines(self, session, interval, limit=1000, end_time=None):
        """Fetches historical OHLCV data from Binance REST API."""
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }
        if end_time:
            params["endTime"] = end_time
            
        async with session.get(BINANCE_REST_URL, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"[ERROR] Binance API returned {response.status} for {interval}")
                return []

    def _compute_krs_zones(self, df: pd.DataFrame, pivot_strength=15, merge_threshold=0.001, max_zones=10):
        """Identifies key structural pivots and merges closely packed liquidity zones."""
        if len(df) < pivot_strength * 2 + 1:
            return [], []

        # Find Pivot Highs
        rolling_max = df['high'].rolling(window=2*pivot_strength+1, center=True).max()
        is_pivot_high = df['high'] == rolling_max
        
        # Find Pivot Lows
        rolling_min = df['low'].rolling(window=2*pivot_strength+1, center=True).min()
        is_pivot_low = df['low'] == rolling_min

        # Process Resistance Zones
        res_boxes = []
        last_res = None
        for price in df['high'][is_pivot_high].dropna():
            if last_res is not None and abs(price - last_res) / last_res <= merge_threshold:
                continue
            res_boxes.append(price)
            last_res = price
            if len(res_boxes) > max_zones:
                res_boxes.pop(0)

        # Process Support Zones
        sup_boxes = []
        last_sup = None
        for price in df['low'][is_pivot_low].dropna():
            if last_sup is not None and abs(price - last_sup) / last_sup <= merge_threshold:
                continue
            sup_boxes.append(price)
            last_sup = price
            if len(sup_boxes) > max_zones:
                sup_boxes.pop(0)

        res_boxes = [round(p, 2) for p in res_boxes]
        sup_boxes = [round(p, 2) for p in sup_boxes]
        
        return res_boxes, sup_boxes

    def compute_technical_indicators(self, df: pd.DataFrame) -> dict:
        """Applies vectorized TA to a DataFrame and returns the latest state."""
        if df.empty or len(df) < 200:
            return {}

        pd.options.mode.chained_assignment = None

        # Momentum
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Trend: Simple Moving Averages (SMA)
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=100, append=True)
        df.ta.sma(length=200, append=True)

        # Trend: Exponential Moving Averages (EMA)
        df.ta.ema(length=10, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=100, append=True)
        df.ta.ema(length=200, append=True)
        
        # Volatility
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        # Custom KRS Zones
        res_zones, sup_zones = self._compute_krs_zones(df)

        # Get the absolute latest row
        latest = df.iloc[-1]
        
        ta_state = {
            "rsi_14": float(latest.get("RSI_14", 50.0)),
            "macd": float(latest.get("MACD_12_26_9", 0.0)),
            "macd_hist": float(latest.get("MACDh_12_26_9", 0.0)),
            
            "sma_10": float(latest.get("SMA_10", latest["close"])),
            "sma_20": float(latest.get("SMA_20", latest["close"])),
            "sma_50": float(latest.get("SMA_50", latest["close"])),
            "sma_100": float(latest.get("SMA_100", latest["close"])),
            "sma_200": float(latest.get("SMA_200", latest["close"])),

            "ema_10": float(latest.get("EMA_10", latest["close"])),
            "ema_20": float(latest.get("EMA_20", latest["close"])),
            "ema_50": float(latest.get("EMA_50", latest["close"])),
            "ema_100": float(latest.get("EMA_100", latest["close"])),
            "ema_200": float(latest.get("EMA_200", latest["close"])),
            
            "atr_14": float(latest.get("ATRr_14", 0.0)),
            "bb_upper": float(latest.get("BBU_20_2.0", 0.0)),
            "bb_lower": float(latest.get("BBL_20_2.0", 0.0)),
            "live_price": float(latest["close"]),
            
            "krs_res_zones": json.dumps(res_zones),
            "krs_sup_zones": json.dumps(sup_zones)
        }
        
        # Clean up NaNs
        clean_state = {}
        for k, v in ta_state.items():
            if isinstance(v, str):
                clean_state[k] = v
            else:
                clean_state[k] = 0.0 if pd.isna(v) else v
                
        return clean_state

    async def bootstrap_time_bars(self, session):
        """Warms up standard timeframes: 1m, 5m, 15m, 1h, 4h."""
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        print("[BOOTSTRAP] Fetching standard timeframes to warm up Moving Averages...")
        
        for tf in timeframes:
            data = await self.fetch_klines(session, tf, limit=500)
            if not data: continue
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_vol', 'taker_buy_quote', 'ignore'])
            df = df.astype(float)
            
            ta_state = self.compute_technical_indicators(df)
            
            await self.redis.hset(f"btc_market_state:ta:{tf}", mapping=ta_state)
            print(f" -> {tf.upper()} Indicators calculated and loaded into Redis.")

            if tf == '1m':
                print("[BOOTSTRAP] Seeding 1-minute historical CVD state array...")
                for _, row in df.tail(60).iterrows():
                    vol = row['volume']
                    taker_buy = row['taker_buy_vol']
                    taker_sell = vol - taker_buy
                    approx_cvd = taker_buy - taker_sell
                    
                    minute_data = {
                        "timestamp": int(row['timestamp']) / 1000,
                        "open": row['open'], "high": row['high'],
                        "low": row['low'], "close": row['close'],
                        "volume_btc": vol,
                        "net_volume_delta": approx_cvd,
                        "flag": "NONE"
                    }
                    score = int(row['timestamp']) / 1000
                    await self.redis.zadd("btc_market_state:1m_history", {json.dumps(minute_data): score})

    async def bootstrap_dollar_bars(self, session):
        """Synthetically reconstructs historical Dollar Bars from 1m data."""
        print("[BOOTSTRAP] Reconstructing historical Microstructure Dollar Bars...")
        
        all_1m_data = []
        end_time = None
        for _ in range(3): 
            data = await self.fetch_klines(session, '1m', limit=1000, end_time=end_time)
            if not data: break
            all_1m_data = data + all_1m_data
            end_time = int(data[0][0]) - 1   
            await asyncio.sleep(0.5)         

        df_1m = pd.DataFrame(all_1m_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_vol', 'taker_buy_quote', 'ignore'])
        df_1m = df_1m.astype(float)
        
        dollar_bars = []
        current_bar = {'open': None, 'high': float('-inf'), 'low': float('inf'), 'close': None, 'volume': 0, 'quote_vol': 0, 'cvd': 0}
        
        for _, row in df_1m.iterrows():
            if current_bar['open'] is None: current_bar['open'] = row['open']
            current_bar['high'] = max(current_bar['high'], row['high'])
            current_bar['low'] = min(current_bar['low'], row['low'])
            current_bar['close'] = row['close']
            
            current_bar['volume'] += row['volume']
            current_bar['quote_vol'] += row['quote_asset_volume']
            
            taker_buy = row['taker_buy_vol']
            taker_sell = row['volume'] - taker_buy
            current_bar['cvd'] += (taker_buy - taker_sell)
            
            if current_bar['quote_vol'] >= self.dollar_bar_threshold:
                bar_data = {
                    'timestamp': row['timestamp'],
                    'open': current_bar['open'],
                    'high': current_bar['high'],
                    'low': current_bar['low'],
                    'close': current_bar['close'],
                    'volume': current_bar['volume'],
                    'cvd': current_bar['cvd']
                }
                dollar_bars.append(bar_data)
                
                score_timestamp = float(row['timestamp']) / 1000.0
                await self.redis.zadd("btc_market_state:dollar_bars", {json.dumps(bar_data): score_timestamp})
                
                current_bar = {'open': row['close'], 'high': row['close'], 'low': row['close'], 'close': row['close'], 'volume': 0, 'quote_vol': 0, 'cvd': 0}

        df_db = pd.DataFrame(dollar_bars)
        if not df_db.empty:
            ta_state = self.compute_technical_indicators(df_db)
            await self.redis.hset("btc_market_state:ta:dollar_bar", mapping=ta_state)
            print(f" -> Constructed {len(df_db)} historical Dollar Bars. State loaded into Redis.")
        else:
            print("[WARNING] Failed to construct historical Dollar Bars.")

    async def run(self):
        """Executes the pre-flight checklist."""
        print("===================================================")
        print("🚀 INITIATING PRE-FLIGHT QUANT BOOTSTRAPPER")
        print("===================================================")
        
        await self.redis.flushdb()
        print("[SYSTEM] Redis memory wiped clean.")
        
        connector = aiohttp.TCPConnector(family=socket.AF_INET, resolver=aiohttp.ThreadedResolver())
        async with aiohttp.ClientSession(connector=connector) as session:
            await self.bootstrap_time_bars(session)
            await self.bootstrap_dollar_bars(session)
            
        print("===================================================")
        print("✅ BOOTSTRAP COMPLETE. READY FOR LIVE INGESTION.")
        print("===================================================")


if __name__ == "__main__":
    bootstrapper = MarketBootstrapper()
    try:
        asyncio.run(bootstrapper.run())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Bootstrapper aborted.")