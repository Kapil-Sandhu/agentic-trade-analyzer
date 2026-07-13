import asyncio
import json
import os
import time
from dotenv import load_dotenv
from redis.asyncio import Redis
from pydantic import BaseModel, Field

# Load configurations
load_dotenv(dotenv_path="config/.env")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ==========================================
# 1. THE PYDANTIC AI SCHEMA 
# ==========================================
class TrendAverages(BaseModel):
    sma_10: float; sma_20: float; sma_50: float; sma_100: float; sma_200: float
    ema_10: float; ema_20: float; ema_50: float; ema_100: float; ema_200: float

class TechnicalState(BaseModel):
    rsi: float = Field(description="RSI 14 value")
    rsi_condition: str = Field(description="OVERBOUGHT (>70), OVERSOLD (<30), or NEUTRAL")
    macd_histogram: float = Field(description="MACD Histogram value")
    macd_trend: str = Field(description="BULLISH (Hist > 0) or BEARISH (Hist < 0)")
    trend_structure: str = Field(description="Pre-calculated market regime")
    distance_from_200_ema: float = Field(description="Percentage distance from 200 EMA")
    moving_averages: TrendAverages = Field(description="All calculated SMAs and EMAs")
    bollinger_position: str = Field(description="Price relative to Bollinger Bands")
    bollinger_upper: float = Field(description="Upper Band (Resistance)")
    bollinger_lower: float = Field(description="Lower Band (Support)")
    atr_14: float = Field(description="Average True Range (Volatility)")
    nearest_support: float = Field(description="Closest KRS Support level below price")
    nearest_resistance: float = Field(description="Closest KRS Resistance level above price")

class MarketSummary(BaseModel):
    timestamp_minute: int = Field(description="Current minute timestamp")
    live_price: float = Field(description="Current absolute price of BTC")
    cvd_1h_trend: str = Field(description="Direction of Cumulative Volume Delta (UP/DOWN/FLAT)")
    net_volume_delta_1h: float = Field(description="Sum of volume deltas over the last hour")
    absorption_anomalies_count: int = Field(description="Number of absorption flags fired in the last hour")
    primary_regime: str = Field(description="Pre-computed dominant flow regime")
    ta_1m: TechnicalState
    ta_5m: TechnicalState
    ta_15m: TechnicalState
    ta_1h: TechnicalState
    ta_4h: TechnicalState
    ta_dollar_bar: TechnicalState

# ==========================================
# 2. THE COMPRESSION ENGINE
# ==========================================
class ContextBuilder:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    def _round(self, value):
        """Strictly enforces 2 decimal points for clean LLM ingestion."""
        try:
            return round(float(value), 2)
        except (ValueError, TypeError):
            return 0.0

    def _extract_ta_state(self, raw_state: dict, live_price: float) -> TechnicalState:
        if not raw_state:
            empty_ma = TrendAverages(sma_10=live_price, sma_20=live_price, sma_50=live_price, sma_100=live_price, sma_200=live_price,
                                     ema_10=live_price, ema_20=live_price, ema_50=live_price, ema_100=live_price, ema_200=live_price)
            return TechnicalState(rsi=50.0, rsi_condition="NEUTRAL", macd_histogram=0.0, macd_trend="NEUTRAL",
                                  trend_structure="UNKNOWN", distance_from_200_ema=0.0, moving_averages=empty_ma, 
                                  bollinger_position="UNKNOWN", bollinger_upper=live_price, bollinger_lower=live_price, 
                                  atr_14=0.0, nearest_support=0.0, nearest_resistance=float('inf'))
            
        rsi = self._round(raw_state.get('rsi_14', 50.0))
        rsi_condition = "OVERBOUGHT" if rsi >= 70 else "OVERSOLD" if rsi <= 30 else "NEUTRAL"
        
        macd_hist = self._round(raw_state.get('macd_hist', 0.0))
        macd_trend = "BULLISH" if macd_hist > 0 else "BEARISH"

        ma = TrendAverages(
            sma_10=self._round(raw_state.get('sma_10', live_price)), sma_20=self._round(raw_state.get('sma_20', live_price)),
            sma_50=self._round(raw_state.get('sma_50', live_price)), sma_100=self._round(raw_state.get('sma_100', live_price)),
            sma_200=self._round(raw_state.get('sma_200', live_price)), ema_10=self._round(raw_state.get('ema_10', live_price)),
            ema_20=self._round(raw_state.get('ema_20', live_price)), ema_50=self._round(raw_state.get('ema_50', live_price)),
            ema_100=self._round(raw_state.get('ema_100', live_price)), ema_200=self._round(raw_state.get('ema_200', live_price))
        )
        
        dist_200_ema_pct = self._round(((live_price - ma.ema_200) / ma.ema_200) * 100)
        
        if live_price > ma.ema_200:
            if dist_200_ema_pct > 5.0 and rsi > 70:
                trend_structure = "UPTREND OVEREXTENDED (Price >5% from Mean. High Reversal Risk)"
            elif live_price < ma.ema_20 and live_price >= ma.ema_50:
                trend_structure = "UPTREND PULLBACK (Price mean-reverting to 50 EMA support)"
            elif live_price < ma.ema_50:
                trend_structure = "DEEP PULLBACK (Micro-trend broken, testing Macro Mean)"
            elif dist_200_ema_pct >= 1.0:
                trend_structure = "UPTREND ALIGNED (Healthy extension. Safe to Trend-Follow)"
            else:
                trend_structure = "CONSOLIDATION (Price clamped near Macro Mean)"
        else:
            if dist_200_ema_pct < -5.0 and rsi < 30:
                trend_structure = "DOWNTREND OVEREXTENDED (Price <-5% from Mean. High Squeeze Risk)"
            elif live_price > ma.ema_20 and live_price <= ma.ema_50:
                trend_structure = "DOWNTREND PULLBACK (Price mean-reverting to 50 EMA resistance)"
            elif live_price > ma.ema_50:
                trend_structure = "DEEP RELIEF (Micro-trend broken, testing Macro Mean)"
            elif dist_200_ema_pct <= -1.0:
                trend_structure = "DOWNTREND ALIGNED (Healthy negative extension. Safe to Trend-Follow)"
            else:
                trend_structure = "CONSOLIDATION (Price clamped near Macro Mean)"

        bb_upper = self._round(raw_state.get('bb_upper', live_price))
        bb_lower = self._round(raw_state.get('bb_lower', live_price))
        atr_14 = self._round(raw_state.get('atr_14', 0.0))
        
        if live_price >= bb_upper:
            bollinger_position = "AT UPPER BAND (Overextended/Resistance)"
        elif live_price <= bb_lower:
            bollinger_position = "AT LOWER BAND (Oversold/Support)"
        else:
            bollinger_position = "MID BAND (Neutral)"

        try:
            res_zones = json.loads(raw_state.get('krs_res_zones', '[]'))
            sup_zones = json.loads(raw_state.get('krs_sup_zones', '[]'))
        except:
            res_zones, sup_zones = [], []

        nearest_res = float('inf')
        nearest_sup = 0.0
        
        for z in res_zones:
            if z > live_price and z < nearest_res:
                nearest_res = z
        for z in sup_zones:
            if z < live_price and z > nearest_sup:
                nearest_sup = z
                
        return TechnicalState(
            rsi=rsi, rsi_condition=rsi_condition, macd_histogram=macd_hist, macd_trend=macd_trend,
            trend_structure=trend_structure, distance_from_200_ema=dist_200_ema_pct, moving_averages=ma,
            bollinger_position=bollinger_position, bollinger_upper=bb_upper, bollinger_lower=bb_lower,
            atr_14=atr_14, nearest_support=self._round(nearest_sup), nearest_resistance=self._round(nearest_res if nearest_res != float('inf') else 0.0)
        )

    async def _fetch_and_compress(self):
        current_minute = int(time.time() // 60)
        sixty_mins_ago = current_minute - 60

        raw_history = await self.redis.zrange("btc_market_state:1m_history", start=sixty_mins_ago, end=current_minute, byscore=True)
        live_price_raw = await self.redis.hget("btc_market_state:live", "current_price")
        live_price = self._round(live_price_raw) if live_price_raw else 64000.0

        ta_1m_raw = await self.redis.hgetall("btc_market_state:ta:1m")
        ta_5m_raw = await self.redis.hgetall("btc_market_state:ta:5m")
        ta_15m_raw = await self.redis.hgetall("btc_market_state:ta:15m")
        ta_1h_raw = await self.redis.hgetall("btc_market_state:ta:1h")
        ta_4h_raw = await self.redis.hgetall("btc_market_state:ta:4h")
        ta_db_raw = await self.redis.hgetall("btc_market_state:ta:dollar_bar")

        total_delta = 0.0
        anomalies = 0
        cvd_start = 0.0
        cvd_end = 0.0

        if raw_history:
            first_tick = json.loads(raw_history[0])
            last_tick = json.loads(raw_history[-1])
            cvd_start = first_tick.get("cvd_close", 0.0)
            cvd_end = last_tick.get("cvd_close", 0.0)
            
            for item in raw_history:
                data = json.loads(item)
                total_delta += data.get("net_volume_delta", 0.0)
                if data.get("flag") == "BULLISH_ABSORPTION":
                    anomalies += 1

        cvd_trend = "FLAT"
        if cvd_end > cvd_start + 10: cvd_trend = "UP"
        elif cvd_end < cvd_start - 10: cvd_trend = "DOWN"
        
        regime = "BEARISH_FLOW"
        if cvd_trend == "UP" and anomalies > 0: regime = "BULLISH_FLOW"
        elif anomalies >= 3: regime = "HEAVY_ACCUMULATION"

        summary = MarketSummary(
            timestamp_minute=current_minute,
            live_price=live_price,
            cvd_1h_trend=cvd_trend,
            net_volume_delta_1h=self._round(total_delta),
            absorption_anomalies_count=anomalies,
            primary_regime=regime,
            ta_1m=self._extract_ta_state(ta_1m_raw, live_price),
            ta_5m=self._extract_ta_state(ta_5m_raw, live_price),
            ta_15m=self._extract_ta_state(ta_15m_raw, live_price),
            ta_1h=self._extract_ta_state(ta_1h_raw, live_price),
            ta_4h=self._extract_ta_state(ta_4h_raw, live_price),
            ta_dollar_bar=self._extract_ta_state(ta_db_raw, live_price)
        )

        return summary.model_dump_json(indent=2)

    async def run_scheduler(self):
        print("[CONTEXT BUILDER] Online. Compressing memory for LLM consumption...")
        while True:
            try:
                clean_json_context = await self._fetch_and_compress()
                await self.redis.set("ai_context:latest_1h_summary", clean_json_context)
            except Exception as e:
                print(f"[ERROR] Context formulation failed: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    builder = ContextBuilder()
    try:
        asyncio.run(builder.run_scheduler())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Context Builder offline.")