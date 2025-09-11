import asyncio
import argparse
import logging

from src.config import Config
from src.server import MotionCaptureServer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='MotionPlay Server - Real-time Motion Capture WebSocket Server')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', 
                        help='Computing device (cuda/cpu, default: cuda, auto-selects best available)')
    parser.add_argument('--host', default='localhost', help='WebSocket server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket server port (default: 8765)')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--gpu-memory', type=float, default=0.7, help='GPU memory fraction (0.1-0.9, default: 0.7)')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing (default: 2 for Mac)')
    
    return parser.parse_args()

async def main():
    server = None
    try:
        args = parse_arguments()
        
        logger.info("=== MotionPlay Motion Capture Server ===")
        
        config = Config.from_args(args)
        
        server = MotionCaptureServer(config)
        
        if not server.initialize():
            logger.error("Server initialization failed")
            return
        
        logger.info("🚀 Starting MotionPlay Server...")
        
        await server.start_server()
        
    except KeyboardInterrupt:
        logger.info("👋 User interrupted server")
    except Exception as e:
        logger.error(f"❌ Server runtime error: {e}")
    finally:
        if server:
            server.stop_server()

if __name__ == '__main__':
    asyncio.run(main())
