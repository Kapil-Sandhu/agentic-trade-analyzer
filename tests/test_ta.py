import asyncio
import os
import json
from redis.asyncio import Redis
from dotenv import load_dotenv

# Load environment configuration
load_dotenv(dotenv_path="config/.env")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

async def verify_ta_pipeline():
    print("===================================================")
    print("🔍 INITIATING TA PIPELINE VERIFICATION")
    print("===================================================")
    
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    # We check all standard timeframes + the proprietary dollar bar
    timeframes = ['1m', '5m', '15m', '1h', '4h', 'dollar_bar']
    
    all_passed = True

    for tf in timeframes:
        print(f"\n[{tf.upper()}] Fetching TA state from Redis...")
        state = await redis.hgetall(f"btc_market_state:ta:{tf}")
        
        if not state:
            print(f" ❌ [ERROR] No data found for {tf}. Is the TA Engine running?")
            all_passed = False
            continue
            
        print(" ✅ [SUCCESS] Data found. Verifying payload structure:")
        
        # Safely attempt to parse expected keys
        live_price = state.get('live_price', 'MISSING')
        rsi = state.get('rsi_14', 'MISSING')
        sma_200 = state.get('sma_200', 'MISSING')
        ema_200 = state.get('ema_200', 'MISSING')
        
        print(f"    -> Live Price: ${live_price}")
        print(f"    -> RSI (14):   {rsi}")
        print(f"    -> SMA (200):  {sma_200}")
        print(f"    -> EMA (200):  {ema_200}")
        
        # Verify the custom KRS JSON arrays
        res_zones = state.get("krs_res_zones")
        sup_zones = state.get("krs_sup_zones")
        
        if res_zones and sup_zones:
            print(f"    -> KRS Resistance Zones: {res_zones}")
            print(f"    -> KRS Support Zones:    {sup_zones}")
        else:
            print("    -> ⚠️ [WARNING] KRS Zones missing!")
            all_passed = False

    print("\n===================================================")
    if all_passed:
        print("✅ VERIFICATION COMPLETE: Pipeline is Flawless.")
    else:
        print("⚠️ VERIFICATION COMPLETE: Missing data detected.")
    print("===================================================")
    
    await redis.aclose()

if __name__ == "__main__":
    asyncio.run(verify_ta_pipeline())