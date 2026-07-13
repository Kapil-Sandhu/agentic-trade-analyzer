import asyncio
import aiohttp
import socket
import os
import json
import pandas as pd
import pandas_ta as ta
from redis.asyncio import Redis
from dotenv import load_dotenv

# Load environment configuration
load_dotenv(dotenv_path="config/.env")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"

class VectorizedTAEngine:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.symbol = "BTCUSDT"
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.update_interval = 5.0  # Calculate fresh TA every 5 seconds

    def _compute_krs_zones(self, df: pd.DataFrame, pivot_strength=15, merge_threshold=0.001, max_zones=10):
        """
        Python port of the Custom KRS Support and Resistance TradingView Indicator.
        Identifies key structural pivots and merges closely packed liquidity zones.
        """
        if len(df) < pivot_strength * 2 + 1:
            return [], []

        # Find Pivot Highs (Strict local maxima over the window)
        rolling_max = df['high'].rolling(window=2*pivot_strength+1, center=True).max()
        is_pivot_high = df['high'] == rolling_max
        
        # Find Pivot Lows (Strict local minima over the window)
        rolling_min = df['low'].rolling(window=2*pivot_strength+1, center=True).min()
        is_pivot_low = df['low'] == rolling_min

        # Process Resistance Zones
        res_boxes = []
        last_res = None
        # Dropna() handles the NaN edges created by center=True rolling
        for price in df['high'][is_pivot_high].dropna():
            # Merge Logic: If it's too close to the last zone, ignore the new one
            if last_res is not None and abs(price - last_res) / last_res <= merge_threshold:
                continue
            res_boxes.append(price)
            last_res = price
            # Cap the maximum tracked active zones to prevent memory/context bloat
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

        # Return cleanly rounded float lists
        res_boxes = [round(p, 2) for p in res_boxes]
        sup_boxes = [round(p, 2) for p in sup_boxes]
        
        return res_boxes, sup_boxes

    def _compute_ta(self, df: pd.DataFrame) -> dict:
        """Applies vectorized TA to a DataFrame and extracts the current active state."""
        if df.empty or len(df) < 200:
            return {}

        # Suppress SettingWithCopyWarning for clean terminal
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

        # Custom KRS Support & Resistance Zones
        res_zones, sup_zones = self._compute_krs_zones(df)

        # Grab the absolute latest row (the currently forming live candle)
        latest = df.iloc[-1]
        
        ta_state = {
            "rsi_14": float(latest.get("RSI_14", 50.0)),
            "macd": float(latest.get("MACD_12_26_9", 0.0)),
            "macd_hist": float(latest.get("MACDh_12_26_9", 0.0)),
            
            # SMAs
            "sma_10": float(latest.get("SMA_10", latest["close"])),
            "sma_20": float(latest.get("SMA_20", latest["close"])),
            "sma_50": float(latest.get("SMA_50", latest["close"])),
            "sma_100": float(latest.get("SMA_100", latest["close"])),
            "sma_200": float(latest.get("SMA_200", latest["close"])),

            # EMAs
            "ema_10": float(latest.get("EMA_10", latest["close"])),
            "ema_20": float(latest.get("EMA_20", latest["close"])),
            "ema_50": float(latest.get("EMA_50", latest["close"])),
            "ema_100": float(latest.get("EMA_100", latest["close"])),
            "ema_200": float(latest.get("EMA_200", latest["close"])),
            
            # Volatility
            "atr_14": float(latest.get("ATRr_14", 0.0)),
            "bb_upper": float(latest.get("BBU_20_2.0", 0.0)),
            "bb_lower": float(latest.get("BBL_20_2.0", 0.0)),
            "live_price": float(latest["close"]),
            
            # KRS Zones (JSON stringified for Redis Hash storage)
            "krs_res_zones": json.dumps(res_zones),
            "krs_sup_zones": json.dumps(sup_zones)
        }
        
        # Clean NaNs in case of early calculation anomalies, while preserving strings
        clean_state = {}
        for k, v in ta_state.items():
            if isinstance(v, str):
                clean_state[k] = v
            else:
                clean_state[k] = 0.0 if pd.isna(v) else v
                
        return clean_state

    async def _update_standard_timeframes(self, session):
        """Fetches standard OHLCV from Binance and updates Redis."""
        for tf in self.timeframes:
            params = {"symbol": self.symbol, "interval": tf, "limit": 250} # 250 is enough for 200-EMA
            
            try:
                async with session.get(BINANCE_REST_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_vol', 'taker_buy_quote', 'ignore'])
                        df = df.astype(float)
                        
                        ta_state = self._compute_ta(df)
                        if ta_state:
                            await self.redis.hset(f"btc_market_state:ta:{tf}", mapping=ta_state)
            except Exception as e:
                print(f"[ERROR] Failed to fetch {tf} from Binance: {e}")

    async def _update_dollar_bars(self):
        """Pulls the proprietary Dollar Bars from Redis memory and calculates TA."""
        try:
            # Fetch the last 300 dollar bars from Redis Sorted Set
            raw_bars = await self.redis.zrevrange("btc_market_state:dollar_bars", 0, 299)
            if not raw_bars or len(raw_bars) < 200:
                return # Not enough live bars formed yet to calculate a 200-EMA

            # Reverse back to chronological order (oldest to newest)
            raw_bars.reverse()
            
            parsed_bars = [json.loads(b) for b in raw_bars]
            df = pd.DataFrame(parsed_bars)
            
            # Ensure columns match expectations for pandas_ta
            if not df.empty:
                ta_state = self._compute_ta(df)
                if ta_state:
                    await self.redis.hset("btc_market_state:ta:dollar_bar", mapping=ta_state)
                    
        except Exception as e:
             print(f"[ERROR] Dollar Bar TA calculation failed: {e}")

    async def run(self):
        print("[TA ENGINE] Vectorized Technical Analysis Microservice Online.")
        print(f"[TA ENGINE] Updating indicators across all timeframes every {self.update_interval} seconds...")
        
        # Bypass Windows DNS bug
        connector = aiohttp.TCPConnector(family=socket.AF_INET, resolver=aiohttp.ThreadedResolver())
        
        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                start_time = asyncio.get_event_loop().time()
                
                # Run both tasks concurrently
                await asyncio.gather(
                    self._update_standard_timeframes(session),
                    self._update_dollar_bars()
                )
                
                # Sleep exactly until the next interval to prevent CPU burnout
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0.0, self.update_interval - elapsed)
                await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    engine = VectorizedTAEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\n[SYSTEM] TA Engine Offline.")