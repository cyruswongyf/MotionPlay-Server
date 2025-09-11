from pathlib import Path
import sys

# Add 4D-Humans module path
LIBS_PATH = Path(__file__).parent.parent / "libs" / "4D-Humans"
if str(LIBS_PATH) not in sys.path:
    sys.path.append(str(LIBS_PATH))

# Import 4D-Humans related modules
from hmr2.models import DEFAULT_CHECKPOINT

class Config:
    """Application configuration class"""
    
    # Server configuration
    SERVER_HOST = "localhost"
    SERVER_PORT = 8765
    
    # Model configuration
    MODEL_CHECKPOINT = DEFAULT_CHECKPOINT
    MODEL_DEVICE = "cuda"  # Auto-detect will choose cuda on NVIDIA, cpu otherwise
    DETECTION_SCORE_THRESHOLD = 0.5
    BATCH_SIZE = 2  # Smaller batch size for Mac compatibility
    
    # Performance configuration
    GPU_MEMORY_FRACTION = 0.7  # Only relevant for CUDA
    ENABLE_MEMORY_CLEANUP = True  # Works on all platforms (CUDA/CPU)
    
    # Camera configuration
    CAMERA_ID = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Detector configuration
    DETECTOR_CONFIG_PATH = LIBS_PATH / "hmr2" / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    DETECTOR_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    DETECTOR_CHECKPOINT_LOCAL = Path(__file__).parent.parent / "libs" / "model_cache" / "detectron2" / "model_final_f05665.pkl"
    
    # Data processing configuration
    MAX_KEYPOINTS = 25  # Reduce keypoints for performance
    
    @classmethod
    def from_args(cls, args):
        instance = cls()
        
        instance.DETECTOR_CONFIG_PATH = cls.DETECTOR_CONFIG_PATH
        
        arg_mapping = {
            'device': 'MODEL_DEVICE',
            'checkpoint': 'MODEL_CHECKPOINT', 
            'camera_id': 'CAMERA_ID',
            'width': 'CAMERA_WIDTH',
            'height': 'CAMERA_HEIGHT',
            'fps': 'CAMERA_FPS',
            'host': 'SERVER_HOST',
            'port': 'SERVER_PORT',
            'gpu_memory': 'GPU_MEMORY_FRACTION',
            'batch_size': 'BATCH_SIZE'
        }
        
        for arg_name, config_attr in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    setattr(instance, config_attr, value)
        
        return instance
