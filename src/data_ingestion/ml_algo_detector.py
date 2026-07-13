import asyncio
import os
import time
import math
import numpy as np
from scipy.signal import lombscargle
from scipy import stats
from sklearn.mixture import GaussianMixture
from redis.asyncio import Redis
from dotenv import load_dotenv
from collections import deque

load_dotenv(dotenv_path="config/.env")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TWAP_WINDOW_MS      = 3_600_000   # 1-hour rolling window for TWAP detection (ms)
TWAP_MIN_BUCKETS    = 60          # Need at least 60 seconds of data to run Lomb-Scargle
ICEBERG_TICK_WINDOW = 1000        # Number of ticks for VCP iceberg scan
GMM_REFIT_INTERVAL  = 15.0        # Seconds between GMM refits
GMM_MIN_SECONDS     = 60          # Minimum seconds of data before first GMM fit
# BTC tick size on Binance is $0.01 → bin to $0.01 precision
# FIX #1: was * 10 / 10.0 ($0.10 bins) → changed to * 100 / 100.0 ($0.01 bins)
PRICE_BIN_PRECISION = 100.0

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING EWMA  (shared utility — identical copy lives in nervous_system.py)
# ─────────────────────────────────────────────────────────────────────────────
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
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * (x - old_mean) ** 2)
        std_dev = math.sqrt(self.variance)
        return (x - self.mean) / std_dev if std_dev > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HUNTER
# ─────────────────────────────────────────────────────────────────────────────
class TickLevelAlgoHunter:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.last_stream_id = "0-0"
        self.tick_window = deque(maxlen=100_000)

        self.gmm = GaussianMixture(n_components=3, random_state=42)
        self.regime_probs = [0.0, 1.0, 0.0]          # default: Normal market
        self._prev_regime_probs = [0.0, 1.0, 0.0]    # FIX #3: keep last stable probs
        self.gmm_fitted = False                        # FIX #3: track whether GMM ever converged

        # FIX #6: initialize to time.time() so the 15-second cooldown starts NOW,
        # not from Unix epoch (which made the first refit fire immediately on startup
        # before enough data was available).
        self.gmm_update_timer = time.time()

        self.ats_tracker = StreamingEWMA(alpha=0.01)

        print("[INIT] Institutional Tick-Level Hunter Initialized.")

    # ─────────────────────────────────────────────────────────────────────────
    # TICK INGESTION
    # ─────────────────────────────────────────────────────────────────────────
    async def _slurp_ticks(self):
        try:
            stream_data = await self.redis.xread(
                {"btc_market_state:raw_ticks": self.last_stream_id},
                count=10_000,
                block=10,
            )
            if stream_data:
                _, messages = stream_data[0]
                for msg_id, payload in messages:
                    # Sentinel price == 0.0 → clean test reset
                    if float(payload.get("price", 1.0)) == 0.0:
                        print("\n🔄 [MEMORY PURGE] Clean Test Reset Detected. Wiping internal RAM...")
                        self.tick_window.clear()
                        self.ats_tracker = StreamingEWMA(alpha=0.01)
                        self.regime_probs = [0.0, 1.0, 0.0]
                        self._prev_regime_probs = [0.0, 1.0, 0.0]
                        self.gmm_fitted = False
                        self.gmm_update_timer = time.time()  # restart cooldown
                        self.last_stream_id = msg_id
                        continue

                    vol = float(payload["volume"])
                    self.tick_window.append({
                        "price":        float(payload["price"]),
                        "volume":       vol,
                        "timestamp_ms": int(payload["timestamp_ms"]),
                        "side":         int(payload["side"]),
                    })
                    self.ats_tracker.update(vol)
                    self.last_stream_id = msg_id

        # FIX #5: log Redis / network errors instead of silently swallowing them.
        except Exception as e:
            print(f"[ERROR] _slurp_ticks: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # MARKET REGIME  (GMM — 3 states: Quiet / Normal / Chaotic)
    # ─────────────────────────────────────────────────────────────────────────
    def _update_market_regime(self, volumes, timestamps):
        if time.time() - self.gmm_update_timer < GMM_REFIT_INTERVAL:
            return

        try:
            t_sec = (timestamps - timestamps[0]) / 1000.0
            bins = np.arange(0, t_sec[-1] + 1.0, 1.0)
            vol_per_sec, _ = np.histogram(t_sec, bins=bins, weights=volumes)

            if len(vol_per_sec) < GMM_MIN_SECONDS:
                return

            X = vol_per_sec.reshape(-1, 1)
            self.gmm.fit(X)

            # Sort components by mean volume (low → high = Quiet → Normal → Chaotic)
            sorted_indices = np.argsort(self.gmm.means_.flatten())
            sorted_means   = self.gmm.means_.flatten()[sorted_indices]

            # ── FIX #3: GMM component flip guard ────────────────────────────
            # After a refit, the component ordering can randomly swap if the
            # market is ambiguous. We validate that the three regimes are
            # meaningfully separated before accepting the new probs.
            # Heuristic: each successive mean must be at least 20% larger than
            # the previous one. If not, discard this refit and keep last stable.
            if self.gmm_fitted:
                if sorted_means[0] <= 0 or \
                   sorted_means[1] < sorted_means[0] * 1.2 or \
                   sorted_means[2] < sorted_means[1] * 1.2:
                    print(f"[GMM] Refit rejected — components not well-separated: {sorted_means.round(2)}. Keeping prior probs.")
                    self.gmm_update_timer = time.time()
                    return
            # ────────────────────────────────────────────────────────────────

            current_state_probs = self.gmm.predict_proba(X[-1].reshape(1, -1))[0]
            new_probs = [
                float(current_state_probs[sorted_indices[0]]),  # Quiet
                float(current_state_probs[sorted_indices[1]]),  # Normal
                float(current_state_probs[sorted_indices[2]]),  # Chaotic
            ]

            self._prev_regime_probs = self.regime_probs
            self.regime_probs       = new_probs
            self.gmm_fitted         = True
            self.gmm_update_timer   = time.time()

            print(f"[GMM] Regime updated → Q:{new_probs[0]:.2f} N:{new_probs[1]:.2f} C:{new_probs[2]:.2f} | Means: {sorted_means.round(2)}")

        # FIX #5: log instead of silently resetting to defaults
        except Exception as e:
            print(f"[ERROR] _update_market_regime: {e}")
            # Only fall back to default if we never had a stable fit
            if not self.gmm_fitted:
                self.regime_probs = [0.0, 1.0, 0.0]

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1 — ICEBERG DETECTION  (Volume Concentration Profile)
    # ─────────────────────────────────────────────────────────────────────────
    def hunt_icebergs(self, prices, volumes, sides):
        recent_p = prices[-ICEBERG_TICK_WINDOW:]
        recent_v = volumes[-ICEBERG_TICK_WINDOW:]
        recent_s = sides[-ICEBERG_TICK_WINDOW:]

        if len(recent_p) < ICEBERG_TICK_WINDOW:
            return "NONE"

        # ── FIX #1: Price bin precision ─────────────────────────────────────
        # BTC minimum tick on Binance = $0.01.
        # Old code used * 10 / 10.0  → $0.10 bins (10x too wide → false positives).
        # Fixed to * 100 / 100.0     → $0.01 bins (matches real tick size).
        rounded_prices = np.round(recent_p * PRICE_BIN_PRECISION) / PRICE_BIN_PRECISION
        # ────────────────────────────────────────────────────────────────────

        mode_price = stats.mode(rounded_prices, keepdims=True).mode[0]

        # VCP: what fraction of total volume is concentrated at the mode price?
        at_mode_mask = (rounded_prices == mode_price)
        vol_at_mode  = np.sum(recent_v[at_mode_mask])
        total_vol    = np.sum(recent_v)
        vcp          = vol_at_mode / total_vol if total_vol > 0 else 0.0

        p_q, p_n, p_c   = self.regime_probs
        dynamic_mult     = (p_q * 5.0) + (p_n * 10.0) + (p_c * 25.0)
        # Guard: if ATS tracker hasn't warmed up yet, use a sensible floor
        ats_mean         = self.ats_tracker.mean if self.ats_tracker.initialized else 1.0
        dynamic_vol_thr  = max(20.0, dynamic_mult * ats_mean)

        if vcp > 0.95 and vol_at_mode > dynamic_vol_thr:
            # side convention (from nervous_system.py):
            #   side = +1  →  seller is maker (passive sell / aggressive buy)
            #   side = -1  →  buyer is maker  (passive buy  / aggressive sell)
            # dominant_side > 0 → buyers hitting a passive sell iceberg → BEARISH iceberg
            # dominant_side < 0 → sellers hitting a passive buy iceberg  → BULLISH iceberg
            dominant_side = np.sum(recent_s[at_mode_mask])
            label = "BEARISH" if dominant_side > 0 else "BULLISH"
            return f"{label}_ICEBERG @ ${mode_price:.2f} | VCP:{vcp:.2%} | Vol:{vol_at_mode:.2f} (Mult:{dynamic_mult:.1f}x)"

        return "NONE"

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2 — TWAP HEARTBEAT  (Lomb-Scargle on 1-second CVD buckets)
    # ─────────────────────────────────────────────────────────────────────────
    def hunt_twap_heartbeat(self, all_timestamps, all_volumes, all_sides):
        # ── FIX #2: Filter by TIME not tick count ───────────────────────────
        # Old code used timestamps[-5000:] → undefined real-time window.
        # A slow session: 5000 ticks = several hours → periods search meaningless.
        # A fast session: 5000 ticks = 90 seconds    → can't detect 5-min TWAP.
        # Fix: always use a fixed 1-hour rolling window as specified in the PDF.
        now_ms    = all_timestamps[-1]
        cutoff_ms = now_ms - TWAP_WINDOW_MS
        mask      = all_timestamps >= cutoff_ms
        timestamps = all_timestamps[mask]
        volumes    = all_volumes[mask]
        sides      = all_sides[mask]

        if len(timestamps) < 100:  # need some ticks to bin
            return False, 0.0
        # ────────────────────────────────────────────────────────────────────

        # Bin directional volume into 1-second buckets (up to 3,600 buckets for 1hr)
        t_sec = (timestamps - timestamps[0]) / 1000.0
        bins  = np.arange(0, t_sec[-1] + 1.0, 1.0)

        directional_vol       = volumes * sides
        cvd_per_sec, bin_edges = np.histogram(t_sec, bins=bins, weights=directional_vol)

        t_signals = bin_edges[:-1]  # left edge of each bucket (seconds)

        # Need at least 60 seconds of data and non-zero variance to run Lomb-Scargle
        if len(cvd_per_sec) < TWAP_MIN_BUCKETS or np.std(cvd_per_sec) == 0:
            return False, 0.0

        y_standardized = (cvd_per_sec - np.mean(cvd_per_sec)) / np.std(cvd_per_sec)

        # Search for periodic components between 10 seconds and 300 seconds
        periods       = np.linspace(10.0, 300.0, 300)
        angular_freqs = 2.0 * np.pi / periods

        try:
            power    = lombscargle(t_signals, y_standardized, angular_freqs)
            peak_idx = np.argmax(power)
            # FAP power > 12.0 is a statistically robust threshold (Signal-to-Noise)
            if power[peak_idx] > 12.0:
                return True, float(periods[peak_idx])
        except Exception as e:
            print(f"[ERROR] hunt_twap_heartbeat lombscargle: {e}")

        return False, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3 — REGIME SHIFTS  (Sweeps & Divergences via Z-scored CVD velocity)
    # ─────────────────────────────────────────────────────────────────────────
    def hunt_regime_shifts(self, prices, volumes, sides):
        if len(prices) < 2000:
            return "NONE", "NONE"

        tick_cvd      = np.cumsum(volumes * sides)
        cvd_vel_array   = tick_cvd[100:] - tick_cvd[:-100]
        price_vel_array = prices[100:] - prices[:-100]

        cvd_mean, cvd_std     = np.mean(cvd_vel_array), np.std(cvd_vel_array)
        price_mean, price_std = np.mean(price_vel_array), np.std(price_vel_array)

        cvd_z   = (cvd_vel_array[-1] - cvd_mean)   / cvd_std   if cvd_std   > 0 else 0.0
        price_z = (price_vel_array[-1] - price_mean) / price_std if price_std > 0 else 0.0

        p_q, p_n, p_c = self.regime_probs
        dynamic_z     = (p_q * 2.0) + (p_n * 3.0) + (p_c * 4.5)

        recent_cvd_vel   = cvd_vel_array[-1]
        recent_price_vel = price_vel_array[-1]

        # ── FIX #4: Replace hardcoded absolute thresholds with std-relative ones ──
        # Old: abs(recent_cvd_vel) > 20.0  and  abs(recent_price_vel) > 10.0
        # Problem: at BTC ~$100k these fire on normal noise (0.01% price move).
        # Fix: use 1-sigma of the respective velocity distribution as the floor.
        # This automatically scales with current market activity.
        cvd_abs_floor   = max(cvd_std   * 1.0, 1e-6)   # at least 1σ of CVD velocity
        price_abs_floor = max(price_std * 1.0, 1e-6)   # at least 1σ of price velocity

        cvd_anomaly   = abs(cvd_z)   > dynamic_z and abs(recent_cvd_vel)   > cvd_abs_floor
        price_anomaly = abs(price_z) > dynamic_z and abs(recent_price_vel) > price_abs_floor
        # ─────────────────────────────────────────────────────────────────────

        markup_flag = "NONE"
        if cvd_anomaly and price_anomaly:
            if recent_cvd_vel > 0 and recent_price_vel > 0:
                markup_flag = f"BULLISH_SWEEP (Z: {cvd_z:.1f})"
            elif recent_cvd_vel < 0 and recent_price_vel < 0:
                markup_flag = f"BEARISH_SWEEP (Z: {cvd_z:.1f})"

        dist_flag = "NONE"
        if cvd_anomaly:
            # Divergence: heavy sell flow but price NOT falling → passive buyers absorbing
            if recent_cvd_vel < 0 and recent_price_vel > -price_abs_floor:
                dist_flag = "DISTRIBUTION"
            # Divergence: heavy buy flow but price NOT rising → passive sellers absorbing
            elif recent_cvd_vel > 0 and recent_price_vel < price_abs_floor:
                dist_flag = "ACCUMULATION"

        return markup_flag, dist_flag

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    async def run_hunter(self):
        print("[WORKER] Tick-Level Algorithm Hunter Online. Slurping firehose...")
        last_heartbeat = time.time()

        while True:
            await self._slurp_ticks()

            # ── Heartbeat status ──────────────────────────────────────────────
            if time.time() - last_heartbeat > 10.0:
                buffer_size = len(self.tick_window)
                if buffer_size < 5000:
                    print(f"⏳ [STATUS] Buffer filling: {buffer_size}/5000 ticks. Waiting for critical mass...")
                else:
                    print(f"📡 [STATUS] Buffer full ({buffer_size} ticks). Hunting actively. Market is quiet...")
                last_heartbeat = time.time()

            if len(self.tick_window) >= 5000:
                ticks      = list(self.tick_window)
                prices     = np.array([t["price"]        for t in ticks])
                volumes    = np.array([t["volume"]       for t in ticks])
                timestamps = np.array([t["timestamp_ms"] for t in ticks])
                sides      = np.array([t["side"]         for t in ticks])

                self._update_market_regime(volumes, timestamps)

                iceberg_state          = self.hunt_icebergs(prices, volumes, sides)
                twap_active, twap_period = self.hunt_twap_heartbeat(timestamps, volumes, sides)
                markup_state, dist_state = self.hunt_regime_shifts(prices, volumes, sides)

                state_map = {
                    "regime":       f"Q:{self.regime_probs[0]:.2f} | N:{self.regime_probs[1]:.2f} | C:{self.regime_probs[2]:.2f}",
                    "iceberg":      iceberg_state,
                    "twap_detected": str(twap_active),
                    "twap_period":  str(round(twap_period, 2)),
                    "markup_sweep": markup_state,
                    "divergence":   dist_state,
                    "timestamp":    str(time.time()),
                }
                await self.redis.hset("btc_market_state:tick_ml_footprints", mapping=state_map)

                flags_active = any(
                    s != "NONE" and s != "False"
                    for s in [iceberg_state, str(twap_active), markup_state, dist_state]
                )
                if flags_active:
                    print(f"🎯 [HUNT] Iceberg: {iceberg_state} | TWAP: {twap_active} ({twap_period:.1f}s) | Sweep: {markup_state} | Div: {dist_state}")
                    last_heartbeat = time.time()  # suppress double-print during active signal

            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(TickLevelAlgoHunter().run_hunter())