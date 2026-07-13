import asyncio
import websockets
import json
from redis.asyncio import Redis

async def verify_system():
    print("Initializing System Verification Sequence...\n")
    
    # Test 1: Redis Connection
    print("[1] Testing Local Redis Infrastructure...")
    try:
        redis_client = Redis(host='localhost', port=6379, decode_responses=True)
        await redis_client.ping()
        print(" -> SUCCESS: Redis is online and responding to async pings.\n")
    except Exception as e:
        print(f" -> FAILED: Could not connect to Redis. Error: {e}\n")
        return

    # Test 2: Binance WebSocket Connection
    print("[2] Testing External WebSocket (Binance)...")
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    try:
        async with websockets.connect(uri) as websocket:
            print(" -> SUCCESS: WebSocket connected. Waiting for live tick...")
            message = await websocket.recv()
            trade_data = json.loads(message)
            live_price = trade_data['p']
            print(f" -> SUCCESS: Received live BTC order. Price: ${live_price}\n")
    except Exception as e:
        print(f" -> FAILED: Could not connect to Binance. Error: {e}\n")
        return

    # Test 3: Redis Read/Write Operations
    print("[3] Testing High-Speed Redis Memory I/O...")
    try:
        await redis_client.hset("system_test", "last_price", live_price)
        retrieved_price = await redis_client.hget("system_test", "last_price")
        if retrieved_price == live_price:
            print(f" -> SUCCESS: Wrote and retrieved ${retrieved_price} from Redis memory.\n")
        else:
            print(" -> FAILED: Data mismatch in Redis.\n")
    except Exception as e:
        print(f" -> FAILED: Redis I/O error: {e}\n")
        return

    print("==================================================")
    print("🚀 ALL SYSTEMS NOMINAL. ENVIRONMENT FULLY VERIFIED.")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(verify_system())