import asyncio
import json
import os
import math
import re
import numpy as np
import ccxt.async_support as ccxt
from redis.asyncio import Redis
from dotenv import load_dotenv

# Load configurations
load_dotenv(dotenv_path="config/.env")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

class ExecutionEngine:
    def __init__(self):
        # 1. Memory Layer Connection
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        
        # 2. Exchange Layer Connection (CCXT)
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
        })
        # CRITICAL: Force Sandbox mode to prevent real capital loss during development
        self.exchange.set_sandbox_mode(True) 
        
        # 3. Institutional Risk Parameters
        self.STARTING_EQUITY = 1000.0     # $1,000 Testnet Bankroll
        self.current_equity = self.STARTING_EQUITY
        self.peak_equity = self.STARTING_EQUITY
        self.GLOBAL_DRAWDOWN_LIMIT = 0.05 # 5% Hard Circuit Breaker (Halt Trading)
        
        self.HISTORICAL_WIN_RATE = 0.55   # (p) for Kelly Math
        self.HISTORICAL_RISK_REWARD = 1.5 # (b) for Kelly Math
        
        # Risk Flags
        self.trading_halted = False

    async def _calculate_expected_shortfall(self) -> float:
        """Calculates 99% Expected Shortfall (CVaR) over the trailing Dollar Bars."""
        try:
            raw_bars = await self.redis.zrevrange("btc_market_state:dollar_bars", 0, 300)
            if not raw_bars or len(raw_bars) < 100:
                return 0.0
                
            closes = np.array([json.loads(b)["close"] for b in raw_bars], dtype=float)
            closes = closes[::-1] # Chronological order
            
            # Calculate log returns
            returns = np.diff(np.log(closes))
            
            if len(returns) == 0: return 0.0
            
            # Find the worst 1% of returns (Left Tail)
            percentile_1 = np.percentile(returns, 1)
            tail_losses = returns[returns <= percentile_1]
            
            if len(tail_losses) == 0: return 0.0
            
            # ES is the absolute mean of the worst 1% crashes
            expected_shortfall = abs(np.mean(tail_losses))
            return float(expected_shortfall)
            
        except Exception as e:
            print(f"[RISK WARN] Expected Shortfall calculation failed: {e}")
            return 0.0

    def _calculate_dynamic_position_size(self, ai_conviction: float, live_price: float, atr: float) -> tuple[float, float]:
        """
        Computes Volatility-Scaled Fractional Kelly.
        Returns: (Notional Trade Size in USDT, Risk per Trade in USDT)
        """
        # 1. Base Fractional Kelly (Quarter-Kelly)
        p = self.HISTORICAL_WIN_RATE
        q = 1.0 - p
        b = self.HISTORICAL_RISK_REWARD
        kelly_fraction = max(0.0, p - (q / b))
        quarter_kelly = kelly_fraction / 4.0
        
        # 2. Scale by AI Conviction
        target_risk_pct = quarter_kelly * ai_conviction
        
        # 3. Volatility Scaling (ATR)
        # We target the risk so that hitting a 1.5x ATR stop loss exactly equals our target_risk_pct
        atr_pct = atr / live_price
        stop_loss_distance_pct = 1.5 * atr_pct
        
        if stop_loss_distance_pct <= 0: return 0.0, 0.0
        
        # How much leverage/size do we need so that a SL hit loses exactly target_risk_pct?
        # Size = (Equity * Risk_Pct) / Stop_Loss_Pct
        risk_dollars = self.current_equity * target_risk_pct
        notional_size_usdt = risk_dollars / stop_loss_distance_pct
        
        # Cap notional size to 1x leverage (no margin trading for baseline safety)
        final_notional = min(notional_size_usdt, self.current_equity)
        
        return final_notional, risk_dollars

    def _parse_gmm_regime(self, regime_str: str) -> str:
        """Parses the 'Q:0.10 | N:0.80 | C:0.10' string into the dominant regime."""
        try:
            matches = re.findall(r'([QNC]):([0-9.]+)', regime_str)
            probs = {k: float(v) for k, v in matches}
            dominant = max(probs, key=probs.get)
            if dominant == 'C': return "CHAOTIC"
            elif dominant == 'N': return "NORMAL"
            return "QUIET"
        except:
            return "NORMAL"

    async def _execute_virtual_trade(self, side: str, amount_usdt: float, price: float):
        """Simulates API execution latency and success."""
        btc_amount = amount_usdt / price
        print(f" -> [EXCHANGE] Routing {side.upper()} order for {btc_amount:.4f} BTC (${amount_usdt:.2f})...")
        await asyncio.sleep(0.15) # Simulate execution latency
        print(" -> [EXCHANGE] Order Filled.")
        return btc_amount

    async def run_poller(self):
        """The core Risk Management and Execution FSM."""
        print("===================================================")
        print("🛡️ INITIATING INSTITUTIONAL RISK ENGINE")
        print("===================================================")
        
        while True:
            try:
                # -------------------------------------------------------------
                # 1. SIPHON GLOBAL STATE
                # -------------------------------------------------------------
                live_price_raw = await self.redis.hget("btc_market_state:live", "current_price")
                if not live_price_raw:
                    await asyncio.sleep(1)
                    continue
                live_price = float(live_price_raw)
                
                # Siphon Microstructure & TA for Risk Math
                ml_footprints = await self.redis.hgetall("btc_market_state:tick_ml_footprints")
                ta_db_raw = await self.redis.hgetall("btc_market_state:ta:dollar_bar")
                atr_14 = float(ta_db_raw.get("atr_14", live_price * 0.01)) if ta_db_raw else (live_price * 0.01)
                
                # Siphon AI Mandate & Active Portfolio
                ai_signal_raw = await self.redis.hget("btc_market_state:ai_mandate", "latest_signal")
                active_trade_raw = await self.redis.hget("btc_market_state:portfolio", "active_trade")
                current_dollar_bars = await self.redis.zcard("btc_market_state:dollar_bars")

                # -------------------------------------------------------------
                # 2. CIRCUIT BREAKER (GLOBAL DRAWDOWN)
                # -------------------------------------------------------------
                if self.current_equity > self.peak_equity:
                    self.peak_equity = self.current_equity
                    
                drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                if drawdown > self.GLOBAL_DRAWDOWN_LIMIT and not self.trading_halted:
                    print(f"\n🚨 [KILL SWITCH ACTIVATED] Global Drawdown hit {drawdown*100:.2f}%. Trading Halted.")
                    self.trading_halted = True
                    # If we had a live position here, we would market sell it instantly.
                
                if self.trading_halted:
                    await asyncio.sleep(5)
                    continue

                # -------------------------------------------------------------
                # 3. POSITION MANAGER (IN_POSITION STATE)
                # -------------------------------------------------------------
                if active_trade_raw:
                    trade = json.loads(active_trade_raw)
                    side = trade["side"]
                    entry = trade["entry_price"]
                    sl = trade["sl"]
                    tp_banker = trade["tp_banker"]
                    state = trade["state"]
                    entry_bar_count = trade.get("entry_bar_count", current_dollar_bars)
                    
                    unrealized_pnl_pct = (live_price - entry) / entry if side == "LONG" else (entry - live_price) / entry
                    
                    # --- TIER 1A: HARD STOP LOSS (Math Floor) ---
                    if (side == "LONG" and live_price <= sl) or (side == "SHORT" and live_price >= sl):
                        print(f"\n🛑 [RISK] Hard Stop Loss hit at ${live_price:.2f}. Trade Liquidated.")
                        await self._execute_virtual_trade("sell" if side == "LONG" else "buy", trade["notional_size"], live_price)
                        self.current_equity += trade["notional_size"] * unrealized_pnl_pct
                        await self.redis.hdel("btc_market_state:portfolio", "active_trade")
                        continue
                    
                    # --- TIER 1B: THE BANKER (Deterministic Scale-Out) ---
                    if state == "FULL":
                        hit_tp = (side == "LONG" and live_price >= tp_banker) or (side == "SHORT" and live_price <= tp_banker)
                        if hit_tp:
                            print(f"\n💰 [RISK] +2x ATR Target Hit (${live_price:.2f}). Executing 'Banker' 50% Trim.")
                            await self._execute_virtual_trade("sell" if side == "LONG" else "buy", trade["notional_size"] / 2.0, live_price)
                            
                            # Realize 50% profit
                            self.current_equity += (trade["notional_size"] / 2.0) * unrealized_pnl_pct
                            
                            # Shift to 'Runner' and move SL to Breakeven
                            trade["state"] = "TRIMMED"
                            trade["sl"] = entry 
                            trade["notional_size"] /= 2.0
                            await self.redis.hset("btc_market_state:portfolio", "active_trade", json.dumps(trade))
                            continue

                    # --- TIER 2A: TOXIC FLOW EVACUATION (VPIN/Sweep Check) ---
                    sweep_flag = ml_footprints.get("markup_sweep", "NONE") if ml_footprints else "NONE"
                    div_flag = ml_footprints.get("divergence", "NONE") if ml_footprints else "NONE"
                    
                    is_toxic_long = side == "LONG" and ("BEARISH" in sweep_flag or "DISTRIBUTION" in div_flag)
                    is_toxic_short = side == "SHORT" and ("BULLISH" in sweep_flag or "ACCUMULATION" in div_flag)
                    
                    if is_toxic_long or is_toxic_short:
                        print(f"\n☣️ [RISK EVACUATION] Toxic Flow Detected against position ({sweep_flag} / {div_flag}). Aborting trade!")
                        await self._execute_virtual_trade("sell" if side == "LONG" else "buy", trade["notional_size"], live_price)
                        self.current_equity += trade["notional_size"] * unrealized_pnl_pct
                        await self.redis.hdel("btc_market_state:portfolio", "active_trade")
                        continue

                    # --- TIER 2B: VOLATILITY CHOKE (GMM Check) ---
                    gmm_regime = self._parse_gmm_regime(ml_footprints.get("regime", "")) if ml_footprints else "NORMAL"
                    if gmm_regime == "CHAOTIC" and not trade.get("choked", False):
                        print(f"\n🌪️ [RISK CHOKE] GMM Regime shifted to CHAOTIC. Tightening Stop Loss to 0.5x ATR.")
                        choke_distance = 0.5 * atr_14
                        trade["sl"] = (live_price - choke_distance) if side == "LONG" else (live_price + choke_distance)
                        trade["choked"] = True
                        if state == "FULL":
                            print(f" -> Defensive 50% Trim executed due to volatility.")
                            await self._execute_virtual_trade("sell" if side == "LONG" else "buy", trade["notional_size"] / 2.0, live_price)
                            self.current_equity += (trade["notional_size"] / 2.0) * unrealized_pnl_pct
                            trade["state"] = "TRIMMED"
                            trade["notional_size"] /= 2.0
                        await self.redis.hset("btc_market_state:portfolio", "active_trade", json.dumps(trade))
                        continue

                    # --- TIER 2C: TIME-DECAY TRIM ---
                    bars_passed = current_dollar_bars - entry_bar_count
                    if bars_passed > 50 and state == "FULL":
                        print(f"\n⏳ [RISK DECAY] 50 Dollar Bars passed with no target hit. Opportunity cost too high. Trimming 50%.")
                        await self._execute_virtual_trade("sell" if side == "LONG" else "buy", trade["notional_size"] / 2.0, live_price)
                        self.current_equity += (trade["notional_size"] / 2.0) * unrealized_pnl_pct
                        trade["state"] = "TRIMMED"
                        trade["sl"] = entry # Move to breakeven
                        trade["notional_size"] /= 2.0
                        await self.redis.hset("btc_market_state:portfolio", "active_trade", json.dumps(trade))
                        continue

                    # --- TIER 3: THESIS VALIDATOR (AI Pipeline B Output) ---
                    if ai_signal_raw:
                        ai_signal = json.loads(ai_signal_raw)
                        action = ai_signal.get("direction", "HOLD")
                        
                        if action == "ABORT":
                            print(f"\n🧠 [AI THESIS AUDIT] AI Mandates ABORT: {ai_signal.get('reasoning')}")
                            await self._execute_virtual_trade("sell" if side == "LONG" else "buy", trade["notional_size"], live_price)
                            self.current_equity += trade["notional_size"] * unrealized_pnl_pct
                            await self.redis.hdel("btc_market_state:portfolio", "active_trade")
                            await self.redis.hdel("btc_market_state:ai_mandate", "latest_signal")
                            
                        elif action == "REDUCE_EXPOSURE" and state == "FULL":
                            print(f"\n🧠 [AI THESIS AUDIT] AI Mandates REDUCE_EXPOSURE. Momentum slowing.")
                            await self._execute_virtual_trade("sell" if side == "LONG" else "buy", trade["notional_size"] / 2.0, live_price)
                            self.current_equity += (trade["notional_size"] / 2.0) * unrealized_pnl_pct
                            trade["state"] = "TRIMMED"
                            trade["sl"] = entry 
                            trade["notional_size"] /= 2.0
                            await self.redis.hset("btc_market_state:portfolio", "active_trade", json.dumps(trade))
                            await self.redis.hdel("btc_market_state:ai_mandate", "latest_signal")
                            
                        elif action in ["HOLD", "LONG", "SHORT", "NEUTRAL"]:
                            # Purge processed/irrelevant signals
                            await self.redis.hdel("btc_market_state:ai_mandate", "latest_signal")
                
                # -------------------------------------------------------------
                # 4. SIGNAL PROCESSOR (FLAT STATE - HUNT FOR ENTRIES)
                # -------------------------------------------------------------
                elif ai_signal_raw:
                    try:
                        ai_signal = json.loads(ai_signal_raw)
                        direction = ai_signal.get("direction", "NEUTRAL")
                        conviction = float(ai_signal.get("conviction_score", 0.0))
                        
                        if direction in ["LONG", "SHORT"]:
                            print(f"\n🎯 [SIGNAL] Processing {direction} Mandate (Conviction: {conviction})")
                            
                            # FILTER 1: Expected Shortfall (Tail Risk) Check
                            es_99 = await self._calculate_expected_shortfall()
                            if es_99 > 0.05:
                                print(f" ❌ [RISK REJECT] Expected Shortfall (CVaR) is {es_99*100:.2f}%. Market is too fat-tailed. Suppressing Trade.")
                                await self.redis.hdel("btc_market_state:ai_mandate", "latest_signal")
                                continue
                                
                            # FILTER 2: Fractional Kelly Sizing
                            notional_size, risk_dollars = self._calculate_dynamic_position_size(conviction, live_price, atr_14)
                            if notional_size < 10.0: # Binance minimum
                                print(f" ❌ [RISK REJECT] Calculated trade size (${notional_size:.2f}) below exchange minimums.")
                                await self.redis.hdel("btc_market_state:ai_mandate", "latest_signal")
                                continue
                                
                            # EXECUTE TRADE
                            print(f" ✅ [RISK APPROVED] ES: {es_99*100:.2f}% | Sizing: ${notional_size:.2f} (Risking ${risk_dollars:.2f})")
                            await self._execute_virtual_trade("buy" if direction == "LONG" else "sell", notional_size, live_price)
                            
                            # Construct Institutional Active Trade State
                            sl_price = (live_price - (1.5 * atr_14)) if direction == "LONG" else (live_price + (1.5 * atr_14))
                            tp_banker = (live_price + (2.0 * atr_14)) if direction == "LONG" else (live_price - (2.0 * atr_14))
                            
                            trade_state = {
                                "side": direction,
                                "entry_price": live_price,
                                "notional_size": notional_size,
                                "sl": round(sl_price, 2),
                                "tp_banker": round(tp_banker, 2),
                                "state": "FULL",
                                "choked": False,
                                "entry_bar_count": current_dollar_bars,
                                "reasoning": ai_signal.get("reasoning", ""),
                                "invalidation_condition": ai_signal.get("invalidation_condition", "")
                            }
                            
                            # Lock State into Memory for Pipeline B
                            await self.redis.hset("btc_market_state:portfolio", "active_trade", json.dumps(trade_state))
                            
                        # Clean up memory
                        await self.redis.hdel("btc_market_state:ai_mandate", "latest_signal")
                        
                    except json.JSONDecodeError:
                        await self.redis.hdel("btc_market_state:ai_mandate", "latest_signal")

            except Exception as e:
                print(f"[ROUTER GLOBAL ERROR] {e}")

            # Extreme low latency loop (10ms) for high-speed stop loss & logic catching
            await asyncio.sleep(0.01)

if __name__ == "__main__":
    engine = ExecutionEngine()
    try:
        asyncio.run(engine.run_poller())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Risk Engine shutting down.")
        asyncio.run(engine.exchange.close())