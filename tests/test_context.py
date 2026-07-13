import asyncio
import json
import sys
import os

# Add project root to python path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reasoning.context_builder import ContextBuilder

async def verify_context_output():
    print("===================================================")
    print("🧠 INITIATING LLM CONTEXT VERIFICATION")
    print("===================================================")

    builder = ContextBuilder()

    try:
        print("[SYSTEM] Fetching and compressing data from Redis...")
        
        # Trigger the exact compression function the AI will use
        json_payload = await builder._fetch_and_compress()
        
        # Parse the JSON back to a Python dictionary to verify its keys
        data = json.loads(json_payload)
        
        print("\n✅ [SUCCESS] Payload Generated! Verifying Schema...")
        
        # Check Core Flow Data
        print(f"\n[ORDER FLOW]")
        print(f"  -> CVD Trend: {data.get('cvd_1h_trend')}")
        print(f"  -> Anomalies: {data.get('absorption_anomalies_count')}")
        print(f"  -> Regime:    {data.get('primary_regime')}")
        
        print(f"\n[TIMEFRAMES]")
        # Check all timeframes including the new 1m and 5m
        timeframes = ["ta_1m", "ta_5m", "ta_15m", "ta_1h", "ta_4h", "ta_dollar_bar"]
        all_passed = True
        
        for tf in timeframes:
            if tf in data:
                tf_data = data[tf]
                trend = tf_data.get("trend_structure", "MISSING")
                
                # Check for the deep moving averages block
                mas = tf_data.get("moving_averages", {})
                sma_200 = mas.get("sma_200", "MISSING")
                
                print(f"  ✅ {tf.upper()}: {trend}")
                if sma_200 == "MISSING":
                    print(f"     ⚠️ WARNING: moving_averages block is missing!")
                    all_passed = False
            else:
                print(f"  ❌ [ERROR] {tf.upper()} missing from payload!")
                all_passed = False
        
        print("\n===================================================")
        print("📜 FINAL JSON PAYLOAD FOR LLM:")
        print("===================================================")
        print(json_payload)  # Print the exact string the LLM will see
        
        print("\n===================================================")
        if all_passed:
            print("✅ VERIFICATION COMPLETE: Ready for AI Agents.")
        else:
            print("⚠️ VERIFICATION COMPLETE: Missing data detected.")
        print("===================================================")
        
    except Exception as e:
        print(f"\n❌ [ERROR] Failed to build context: {e}")
    finally:
        await builder.redis.aclose()

if __name__ == "__main__":
    asyncio.run(verify_context_output())