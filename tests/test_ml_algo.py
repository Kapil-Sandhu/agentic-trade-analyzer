import asyncio
import time
import random
import os
from redis.asyncio import Redis
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

class MarketManipulator:
    def __init__(self):
        self.redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.synth_time_ms = int(time.time() * 1000) - (3600 * 1000) 
        self.current_price = 65000.0

    async def _push_ticks(self, num_ticks, price_change_range, volume_range, side_override=None, time_step_ms=10):
        pipeline = self.redis.pipeline()
        for _ in range(num_ticks):
            self.synth_time_ms += time_step_ms
            self.current_price += random.uniform(*price_change_range)
            vol = random.uniform(*volume_range)
            side = side_override if side_override else random.choice([1, -1])

            payload = {
                "price": str(round(self.current_price, 2)), "volume": str(round(vol, 3)),
                "timestamp_ms": str(self.synth_time_ms), "side": str(side)
            }
            pipeline.xadd("btc_market_state:raw_ticks", payload, maxlen=100000)
        
        await pipeline.execute()
        await asyncio.sleep(0.05)

    async def generate_baseline_noise(self):
        print("\n[PHASE 1] Injecting 6,000 ticks of boring, random market noise...")
        await self.redis.delete("btc_market_state:raw_ticks") 
        
        # ==========================================
        # THE POISON PILL (Clears ML RAM before test)
        # ==========================================
        payload = {"price": "0.0", "volume": "0", "timestamp_ms": "0", "side": "0"}
        await self.redis.xadd("btc_market_state:raw_ticks", payload, maxlen=100000)

        self.synth_time_ms = int(time.time() * 1000) - (3600 * 1000) # Reset clock
        await self._push_ticks(6000, (-0.5, 0.5), (0.01, 0.2), time_step_ms=15)
        print("✅ Baseline generated. Wait 3 seconds...\n")
        await asyncio.sleep(3)

    async def trigger_bullish_iceberg(self):
        print("🥶 [ATTACK] Injecting Bullish Iceberg...")
        self.current_price = 65000.0 
        await self._push_ticks(1000, (0.0, 0.0), (1.0, 5.0), side_override=-1, time_step_ms=5)

    async def trigger_markup_sweep(self):
        print("🚀 [ATTACK] Injecting Bullish Sweep (Markup)...")
        await self._push_ticks(150, (0.1, 0.8), (0.5, 2.0), side_override=1, time_step_ms=5)

    async def trigger_distribution(self):
        print("🐋 [ATTACK] Injecting Bearish Distribution...")
        await self._push_ticks(150, (0.0, 0.1), (1.0, 3.0), side_override=-1, time_step_ms=5)

    async def trigger_twap(self):
        print("⏱️ [ATTACK] Injecting TWAP Algorithm (20 Minute Session)...")
        await self.redis.delete("btc_market_state:raw_ticks") 
        
        # THE POISON PILL (Fixes time travel bug for TWAP)
        payload = {"price": "0.0", "volume": "0", "timestamp_ms": "0", "side": "0"}
        await self.redis.xadd("btc_market_state:raw_ticks", payload, maxlen=100000)
        
        self.synth_time_ms = int(time.time() * 1000) - (3600 * 1000)
        
        for _ in range(80): 
            await self._push_ticks(100, (-0.1, 0.1), (0.01, 0.1), time_step_ms=149)
            await self._push_ticks(10, (0.0, 0.2), (2.0, 5.0), side_override=1, time_step_ms=10)

    async def interactive_menu(self):
        # We wipe it instantly on boot so old Binance data doesn't trigger alerts
        await self.generate_baseline_noise()

        while True:
            print("="*50 + "\n 🧪 SYNTHETIC ALGORITHM INJECTOR\n" + "="*50)
            print("1. Trigger Bullish Iceberg\n2. Trigger Bullish Sweep (Markup)")
            print("3. Trigger Bearish Distribution\n4. Trigger 15-Second TWAP Heartbeat\n5. Generate Normal Noise (Clear Flags)\n0. Exit")
            
            choice = input("\nSelect attack to inject: ")
            
            if choice == '1': await self.trigger_bullish_iceberg()
            elif choice == '2': await self.trigger_markup_sweep()
            elif choice == '3': await self.trigger_distribution()
            elif choice == '4': await self.trigger_twap()
            elif choice == '5': await self.generate_baseline_noise()
            elif choice == '0': break
            
            print("\n✅ Injection Complete. Check your ml_algo_detector terminal!")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(MarketManipulator().interactive_menu())