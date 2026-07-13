import asyncio
import websockets
import json
import time
import os
import math
from dotenv import load_dotenv
from redis.asyncio import Redis

# Load environment configuration
load_dotenv(dotenv_path="config/.env")
BINANCE_WS_URL = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws/btcusdt@trade")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ==========================================
# 1. THE STREAMING MATH ENGINE (O(1) Memory)
# ==========================================
class StreamingEWMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mean = 0.0
        self.variance = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.mean = x
            self.variance = 0.0
            self.initialized = True
            return 0.0

        old_mean = self.mean
        self.mean = old_mean + self.alpha * (x - old_mean)
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * (x - old_mean)**2)
        
        std_dev = math.sqrt(self.variance)
        return (x - self.mean) / std_dev if std_dev > 0 else 0.0

# ==========================================
# 2. THE DUAL-CLOCK INGESTION LAYER
# ==========================================
class MarketMicrostructureEngine:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        
        # Buffer increased to 50,000 to survive extreme flash crash volatility
        self.message_queue = asyncio.Queue(maxsize=50000) 
        
        # --- 1. CHRONOLOGICAL CLOCK STATE ---
        self.current_minute = int(time.time() // 60)
        self.minute_open = None
        self.minute_high = float('-inf')
        self.minute_low = float('inf')
        self.minute_close = None
        self.minute_volume_bought = 0.0
        self.minute_volume_sold = 0.0
        self.minute_cvd = 0.0
        self.minute_ticks = 0

        # --- 2. DOLLAR CLOCK STATE ---
        # Set to 50k for rapid testing. Change to 1_000_000.0 for production!
        self.DOLLAR_BAR_THRESHOLD = 1_000_000.0  
        self.current_bar_dollars = 0.0
        self.bar_open = None
        self.bar_high = float('-inf')
        self.bar_low = float('inf')
        self.bar_close = None
        self.bar_volume = 0.0
        self.bar_cvd = 0.0
        self.bar_ticks = 0
        self.bar_start_time = time.time()

        # Multi-Scale EWMV (Order Flow MACD)
        self.fast_cvd_tracker = StreamingEWMA(alpha=0.2)
        self.slow_cvd_tracker = StreamingEWMA(alpha=0.02)
        self.divergence_tracker = StreamingEWMA(alpha=0.1)
        
        # Dynamic Thresholding (Volatility of Volatility)
        self.z_score_volatility = StreamingEWMA(alpha=0.05)

    async def _websocket_reader(self):
        async for websocket in websockets.connect(BINANCE_WS_URL):
            print("[NETWORK] Institutional Dual-Clock Engine Online. Streaming ticks...")
            try:
                async for message in websocket:
                    if self.message_queue.full():
                        await self.message_queue.get() 
                    await self.message_queue.put(message)
            except websockets.ConnectionClosed:
                print("[NETWORK] WebSocket dropped. Reconnecting...")
                continue
            except Exception as e:
                print(f"[ERROR] WebSocket: {e}")

    async def _order_flow_processor(self):
        while True:
            message = await self.message_queue.get()
            trade_data = json.loads(message)
            
            price = float(trade_data['p'])
            volume = float(trade_data['q'])
            trade_minute = int(trade_data['E']) // 60000 
            is_buyer_maker = trade_data['m']
            notional_value = price * volume
            side = -1 if is_buyer_maker else 1
            
            # --- PUSH RAW TICK TO REDIS STREAM FOR ML HUNTER ---
            tick_payload = {
                "price": str(price),
                "volume": str(volume),
                "timestamp_ms": str(trade_data['E']), 
                "side": str(side)
            }
            # Maxlen 100,000 prevents RAM from overflowing while giving ML plenty of history
            await self.redis.xadd("btc_market_state:raw_ticks", tick_payload, maxlen=100000)

            # ==========================================
            # CLOCK 1: TIME BAR AGGREGATION
            # ==========================================
            if trade_minute > self.current_minute:
                await self._process_completed_minute_bar()
                self.current_minute = trade_minute

            if self.minute_open is None: self.minute_open = price
            self.minute_high = max(self.minute_high, price)
            self.minute_low = min(self.minute_low, price)
            self.minute_close = price
            self.minute_ticks += 1
            self.minute_cvd += (volume * side)
            if side == 1: self.minute_volume_bought += volume
            else: self.minute_volume_sold += volume

            # ==========================================
            # CLOCK 2: DOLLAR BAR AGGREGATION (Fractional Splitting)
            # ==========================================
            remaining_notional = notional_value
            remaining_volume = volume
            
            while self.current_bar_dollars + remaining_notional >= self.DOLLAR_BAR_THRESHOLD:
                needed_notional = self.DOLLAR_BAR_THRESHOLD - self.current_bar_dollars
                fraction = needed_notional / notional_value if notional_value > 0 else 0
                allocated_volume = volume * fraction
                allocated_cvd = (volume * side) * fraction
                
                if self.bar_open is None: self.bar_open = price
                self.bar_high = max(self.bar_high, price)
                self.bar_low = min(self.bar_low, price)
                self.bar_close = price
                
                self.current_bar_dollars += needed_notional
                self.bar_volume += allocated_volume
                self.bar_cvd += allocated_cvd
                if self.bar_ticks == 0: self.bar_ticks = 1 
                
                await self._process_completed_dollar_bar(carry_price=price)
                
                remaining_notional -= needed_notional
                remaining_volume -= allocated_volume

            if remaining_notional > 0:
                if self.bar_open is None: self.bar_open = price
                self.bar_high = max(self.bar_high, price)
                self.bar_low = min(self.bar_low, price)
                self.bar_close = price
                
                self.current_bar_dollars += remaining_notional
                self.bar_volume += remaining_volume
                self.bar_cvd += (remaining_volume * side)
                self.bar_ticks += 1

            await self.redis.hset("btc_market_state:live", "current_price", str(price))
            self.message_queue.task_done()

    async def _process_completed_minute_bar(self):
        timestamp = self.current_minute * 60
        minute_data = {
            "timestamp": timestamp,
            "open": self.minute_open, "high": self.minute_high,
            "low": self.minute_low, "close": self.minute_close,
            "volume_btc": round(self.minute_volume_bought + self.minute_volume_sold, 2),
            "cvd_close": round(self.minute_cvd, 2),
            "net_volume_delta": round(self.minute_volume_bought - self.minute_volume_sold, 2),
            "ticks": self.minute_ticks
        }
        bar_json = json.dumps(minute_data)
        await self.redis.zadd("btc_market_state:1m_history", {bar_json: timestamp})
        
        self.minute_open = None
        self.minute_high = float('-inf')
        self.minute_low = float('inf')
        self.minute_close = None
        self.minute_volume_bought = 0.0
        self.minute_volume_sold = 0.0
        self.minute_cvd = 0.0
        self.minute_ticks = 0

    async def _process_completed_dollar_bar(self, carry_price):
        end_time = time.time()
        duration_seconds = end_time - self.bar_start_time
        
        fast_z = self.fast_cvd_tracker.update(self.bar_cvd)
        slow_z = self.slow_cvd_tracker.update(self.bar_cvd)
        
        divergence = self.fast_cvd_tracker.mean - self.slow_cvd_tracker.mean
        div_z_score = self.divergence_tracker.update(divergence)

        self.z_score_volatility.update(abs(div_z_score))
        vol_std_dev = math.sqrt(self.z_score_volatility.variance)
        dynamic_threshold = max(2.0, self.z_score_volatility.mean + (2 * vol_std_dev))

        anomaly_flag = "NONE"
        if div_z_score > dynamic_threshold: anomaly_flag = "BULLISH_REGIME_BREAKOUT"
        elif div_z_score < -dynamic_threshold: anomaly_flag = "BEARISH_REGIME_BREAKDOWN"

        dollar_bar_data = {
            "timestamp_end": end_time,
            "duration_seconds": round(duration_seconds, 3),
            "ticks": self.bar_ticks,
            "open": self.bar_open, "high": self.bar_high,
            "low": self.bar_low, "close": self.bar_close,
            "volume_btc": round(self.bar_volume, 2),
            "cvd_close": round(self.bar_cvd, 2),
            "div_z_score": round(div_z_score, 2),
            "dynamic_threshold": round(dynamic_threshold, 2),
            "regime_flag": anomaly_flag
        }

        bar_json = json.dumps(dollar_bar_data)
        await self.redis.zadd("btc_market_state:dollar_bars", {bar_json: end_time})
        
        self.current_bar_dollars = 0.0
        self.bar_open = carry_price
        self.bar_high = carry_price
        self.bar_low = carry_price
        self.bar_close = carry_price
        self.bar_volume = 0.0
        self.bar_cvd = 0.0
        self.bar_ticks = 0
        self.bar_start_time = time.time()

    async def run(self):
        await asyncio.gather(self._websocket_reader(), self._order_flow_processor())

if __name__ == "__main__":
    engine = MarketMicrostructureEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Ingestion shut down gracefully.")