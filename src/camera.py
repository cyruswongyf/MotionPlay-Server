import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.camera_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_opened():
            logger.warning("Camera not initialized or closed")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Cannot read camera frame")
            return None
        
        return frame

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Camera {self.camera_id} released")
    
    def __enter__(self):
        if self.initialize():
            return self
        else:
            raise RuntimeError("Camera initialization failed")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
