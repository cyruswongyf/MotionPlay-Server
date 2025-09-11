import torch
import logging
import gc
from typing import Tuple

logger = logging.getLogger(__name__)

class DeviceManager:
    """Cross-platform device and memory management for PyTorch models"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._select_best_device()
        
    def _select_best_device(self) -> torch.device:
        """Smart device selection for different platforms"""
        device_preference = self.config.MODEL_DEVICE
        
        # Priority: user choice > CUDA > CPU
        if device_preference == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA GPU acceleration")
        elif device_preference == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU device (forced)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Auto-selected CUDA GPU")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device (fallback)")
        
        return device
    
    def configure_memory_settings(self):
        """Configure memory settings based on the selected device"""
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.config.GPU_MEMORY_FRACTION)
            torch.cuda.empty_cache()
            logger.info(f"CUDA memory limited to {int(self.config.GPU_MEMORY_FRACTION*100)}%")
        else:  # CPU
            torch.set_num_threads(max(1, torch.get_num_threads() // 2))
            logger.info(f"CPU threads limited to {torch.get_num_threads()}")
    
    def clear_memory_cache(self):
        """Clear memory cache across different platforms"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()  # Always run garbage collection
    
    def get_device_info(self) -> str:
        """Get detailed device information"""
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"Device: {self.device}, GPU: {gpu_name} ({gpu_memory:.1f}GB)"
        else:
            cpu_threads = torch.get_num_threads()
            return f"Device: {self.device}, CPU threads: {cpu_threads}"
    
    def get_device(self) -> torch.device:
        """Get the selected device"""
        return self.device
    
    def move_to_device(self, model):
        """Move model to the selected device with memory optimization"""
        model = model.to(self.device)
        
        # Clear cache after moving model
        if self.config.ENABLE_MEMORY_CLEANUP:
            self.clear_memory_cache()
            
        return model
