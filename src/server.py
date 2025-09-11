import asyncio
import websockets
import json
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MotionCaptureServer:
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.pose_estimator = None
        self.is_running = False

    def initialize(self) -> bool:
        try:
            from .camera import Camera
            from .pose_estimator import PoseEstimator

            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = Camera(
                camera_id=self.config.CAMERA_ID,
                width=self.config.CAMERA_WIDTH,
                height=self.config.CAMERA_HEIGHT,
                fps=self.config.CAMERA_FPS,
            )

            if not self.camera.initialize():
                logger.error("Camera initialization failed")
                return False

            # Initialize pose estimator
            logger.info("Initializing pose estimator...")
            self.pose_estimator = PoseEstimator(self.config)

            logger.info(f"Device info: {self.pose_estimator.get_device_info()}")
            
            if not self.pose_estimator.initialize():
                logger.error("Pose estimator initialization failed")
                return False

            logger.info("All components initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def process_frame(self) -> Optional[Dict[str, Any]]:
        try:
            # Read camera frame
            frame = self.camera.read_frame()
            if frame is None:
                return None

            # Detect humans and estimate poses
            boxes = self.pose_estimator.detect_humans(frame)
            poses_data = self.pose_estimator.estimate_poses(frame, boxes)

            logger.debug(f"Detected {len(poses_data)} humans")

            return {
                "skeletons": poses_data,
                "timestamp": time.time(),
                "human_count": len(poses_data),
                "frame_info": {"width": frame.shape[1], "height": frame.shape[0]},
            }

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return None

    async def send_data(self, websocket, data: Dict[str, Any]):
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Data sending failed: {e}")

    async def connection_handler(self, websocket):
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_info}")

        try:
            frame_count = 0
            start_time = time.time()

            while self.is_running:
                frame_data = await self.process_frame()

                if frame_data is not None:
                    await self.send_data(websocket, frame_data)
                    frame_count += 1

                    # Output statistics every 100 frames
                    if frame_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"Real-time FPS: {fps:.2f}")

                        # Reset counter for current period FPS
                        frame_count = 0
                        start_time = time.time()

                # Control frame rate
                await asyncio.sleep(1.0 / self.config.CAMERA_FPS)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_info}")
        except Exception as e:
            logger.error(f"Connection handling error: {e}")
        finally:
            logger.info(f"Cleaning up client connection: {client_info}")

    async def start_server(self):
        self.is_running = True

        logger.info("Starting WebSocket server...")
        logger.info(f"Server address: ws://{self.config.SERVER_HOST}:{self.config.SERVER_PORT}")

        try:
            async with websockets.serve(
                self.connection_handler,
                self.config.SERVER_HOST,
                self.config.SERVER_PORT,
            ):
                logger.info("Server started successfully, waiting for client connections...")
                await asyncio.Future()

        except Exception as e:
            logger.error(f"Server startup failed: {e}")

    def stop_server(self):
        logger.info("Stopping server...")
        self.is_running = False

        if self.camera:
            self.camera.release()

        logger.info("Server stopped")

