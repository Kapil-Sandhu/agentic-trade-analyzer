import asyncio
import json
import os
from redis.asyncio import Redis
from datetime import datetime

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

async def monitor():
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    print("Starting Redis Market Monitor...")

    while True:
        # Clear the terminal for a clean dashboard look
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*60)
        print(" 🕵️‍♂️ QUANT TRADING DESK - LIVE MEMORY MONITOR ")
        print("="*60)

        try:
            # 1. Fetch Live State
            live_state = await redis.hgetall("btc_market_state:live")
            if live_state:
                price = float(live_state.get('current_price', 0))
                print(f"🟢 LIVE PRICE: ${price:,.2f}")
            else:
                print("🔴 LIVE PRICE: WAITING FOR DATA FROM NERVOUS SYSTEM...")

            # 2. Fetch Latest Minute Bar (Time Clock)
            # zrevrange gets the highest scoring (newest) item from the sorted set
            minute_bars = await redis.zrevrange("btc_market_state:1m_history", 0, 0)
            print("\n⏰ LATEST 1-MINUTE TIME BAR:")
            if minute_bars:
                bar = json.loads(minute_bars[0])
                print(f"   -> Time:  {datetime.fromtimestamp(bar['timestamp']).strftime('%H:%M:%S')}")
                print(f"   -> Close: ${bar['close']:,.2f}")
                print(f"   -> Vol:   {bar['volume_btc']} BTC")
                print(f"   -> CVD:   {bar['cvd_close']}")
                print(f"   -> Ticks: {bar['ticks']}")
            else:
                print("   -> Waiting for the clock to hit the top of the minute...")

            # 3. Fetch Latest Dollar Bar (Dollar Clock)
            dollar_bars = await redis.zrevrange("btc_market_state:dollar_bars", 0, 0)
            print("\n💰 LATEST DOLLAR BAR:")
            if dollar_bars:
                bar = json.loads(dollar_bars[0])
                flag = bar.get('regime_flag', 'NONE')
                flag_str = f"🚨 {flag}" if flag != "NONE" else "✅ NORMAL"
                
                print(f"   -> Time:  {datetime.fromtimestamp(bar['timestamp_end']).strftime('%H:%M:%S')} (Duration: {bar['duration_seconds']}s)")
                print(f"   -> Close: ${bar['close']:,.2f}")
                print(f"   -> Vol:   {bar['volume_btc']} BTC")
                print(f"   -> Fast Z: {bar['fast_cvd_mean']} | Slow Z: {bar['slow_cvd_mean']}")
                print(f"   -> State: {flag_str}")
            else:
                print("   -> Waiting for Dollar threshold to be breached...")

        except Exception as e:
            print(f"\n[ERROR] Could not read from Redis. Is the server running? Error: {e}")

        print("\n" + "="*60)
        print(f"Last Dashboard Refresh: {datetime.now().strftime('%H:%M:%S')}")
        
        # Refresh the dashboard every 1 second
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\nMonitor shut down gracefully.")