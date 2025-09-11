import asyncio
import websockets
import json
import time

async def test_client(host='localhost', port=8765, duration=3000):
    uri = f"ws://{host}:{port}"
    print(f"🔗 Connecting to: {uri}")
    
    count = 0
    start_time = time.time()
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connection successful")
            
            while time.time() - start_time < duration:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=200.0)
                    data = json.loads(message)
                    count += 1
                    
                    if count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = count / elapsed
                        humans = data.get('human_count', 0)
                        delay = (time.time() - data.get('timestamp', 0)) * 1000
                        
                        print(f"#{count:4d} | Humans:{humans} | FPS:{fps:5.1f} | Delay:{delay:5.1f}ms")
                        
                except asyncio.TimeoutError:
                    print("⚠️ Timeout")
                    continue
                    
    except (ConnectionRefusedError, OSError) as e:
        print("❌ Connection failed")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Statistics
    elapsed = time.time() - start_time
    avg_fps = count / elapsed if elapsed > 0 else 0
    print(f"\n Statistics: {count} messages | {elapsed:.1f}s | Avg FPS {avg_fps:.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--duration', type=int, default=30)
    args = parser.parse_args()
    
    try:
        asyncio.run(test_client(args.host, args.port, args.duration))
    except KeyboardInterrupt:
        print("\n Test stopped")
