import torch
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any

# Add 4D-Humans module path
LIBS_PATH = Path(__file__).parent.parent / "libs" / "4D-Humans"
if str(LIBS_PATH) not in sys.path:
    sys.path.append(str(LIBS_PATH))

try:
    from hmr2.models import load_hmr2
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    from hmr2.utils.renderer import cam_crop_to_full
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    HMR2_AVAILABLE = True
except ImportError as e:
    logging.warning(f"4D-Humans module import failed: {e}")
    HMR2_AVAILABLE = False

from .device_manager import DeviceManager

logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, config):
        self.config = config
        self.device_manager = DeviceManager(config)
        self.device = self.device_manager.get_device()
        self.model = None
        self.model_cfg = None
        self.detector = None
        
        if not HMR2_AVAILABLE:
            raise RuntimeError("4D-Humans module not available, please check installation and path settings")
    
    def initialize(self) -> bool:
        try:
            # Configure device and memory settings
            self.device_manager.configure_memory_settings()
            
            # Setup cache symbolic link
            from .utils import setup_cache_link
            if not setup_cache_link():
                logger.warning("Cannot setup cache symbolic link, models may be downloaded again")
                from hmr2.configs import CACHE_DIR_4DHUMANS
                from hmr2.models import download_models
                download_models(CACHE_DIR_4DHUMANS)

            # Load HMR2.0 model
            logger.info("Loading HMR2.0 model...")
            self.model, self.model_cfg = load_hmr2(self.config.MODEL_CHECKPOINT)
            self.model = self.device_manager.move_to_device(self.model)
            self.model.eval()
            
            logger.info("HMR2.0 model loaded successfully!")
            
            # Setup human detector
            logger.info("Setting up human detector...")
            detectron2_cfg = LazyConfig.load(str(self.config.DETECTOR_CONFIG_PATH))
            
            # Use local model if exists, otherwise remote URL
            checkpoint_path = (self.config.DETECTOR_CHECKPOINT_LOCAL 
                            if self.config.DETECTOR_CHECKPOINT_LOCAL.exists() 
                            else self.config.DETECTOR_CHECKPOINT_URL)
            detectron2_cfg.train.init_checkpoint = str(checkpoint_path)
            
            # Set detection threshold
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
            logger.info("Human detector setup completed!")
            
            return True
            
        except Exception as e:
            logger.error(f"Pose estimator initialization failed: {e}")
            return False
    
    def detect_humans(self, frame: np.ndarray) -> np.ndarray:
        """Detect humans, return boxes (N, 4) [x1, y1, x2, y2]"""
        
        try:
            det_out = self.detector(frame)
            det_instances = det_out['instances']
            
            # Filter human detection results
            valid_idx = (det_instances.pred_classes == 0) & \
                        (det_instances.scores > self.config.DETECTION_SCORE_THRESHOLD)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            
            return boxes
            
        except Exception as e:
            logger.error(f"Human detection failed: {e}")
            return np.array([])
    
    def estimate_poses(self, frame: np.ndarray, boxes: np.ndarray) -> List[Dict[str, Any]]:
        if len(boxes) == 0:
            return []
        
        logging.debug(f"Estimating poses for {len(boxes)} detected humans")
        
        try:
            # Create dataset
            dataset = ViTDetDataset(self.model_cfg, frame, boxes)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=0
            )
            
            poses_data = []
            
            for batch in dataloader:
                # Convert to device
                batch = recursive_to(batch, self.device)
                
                with torch.no_grad():
                    out = self.model(batch)
                
                # Process each person in the batch
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    pose_data = self._extract_pose_data(batch, out, n)
                    poses_data.append(pose_data)
                
                # Clean up memory after each batch
                if self.config.ENABLE_MEMORY_CLEANUP:
                    self.device_manager.clear_memory_cache()
            
            return poses_data
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return []
    
    def _extract_pose_data(self, batch: Dict, out: Dict, person_idx: int) -> Dict[str, Any]:
        try:
            # Extract and limit keypoints
            keypoints_3d = out['pred_keypoints_3d'][person_idx].detach().cpu().numpy()
            keypoints_3d = keypoints_3d[:self.config.MAX_KEYPOINTS] if len(keypoints_3d) > self.config.MAX_KEYPOINTS else keypoints_3d
            
            # Calculate world position
            scaled_focal_length = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE 
                                 * batch["img_size"][person_idx].max())
            
            position = cam_crop_to_full(
                out['pred_cam'][person_idx].unsqueeze(0),
                batch["box_center"][person_idx].unsqueeze(0),
                batch["box_size"][person_idx].unsqueeze(0),
                batch["img_size"][person_idx].unsqueeze(0),
                scaled_focal_length
            ).detach().cpu().numpy()[0]
            
            confidence = 1.0
            
            return {
                "person_id": int(batch['personid'][person_idx].item()),
                "position": position.tolist(),
                "keypoints_3d": keypoints_3d.tolist(),
                "global_orient": out['pred_smpl_params']['global_orient'][person_idx].detach().cpu().numpy().tolist(),
                "body_pose": out['pred_smpl_params']['body_pose'][person_idx].detach().cpu().numpy().tolist(),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Pose data extraction failed: {e}")
            return {
                "person_id": -1,
                "position": [0, 0, 0],
                "keypoints_3d": [],
                "global_orient": [],
                "body_pose": [],
                "confidence": 0.0
            }
    
    def get_device_info(self) -> str:
        return self.device_manager.get_device_info()
