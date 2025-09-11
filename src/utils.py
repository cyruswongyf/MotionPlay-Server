import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_cache_link():
    local_cache_path = Path(__file__).parent.parent / "libs" / "model_cache" / "4DHumans"
    
    # System default cache path
    system_cache_dir = Path.home() / ".cache"
    system_4dhumans_dir = system_cache_dir / "4DHumans"
    
    try:
        # Ensure system cache directory exists
        system_cache_dir.mkdir(exist_ok=True)
        
        # If it's a symbolic link pointing to the correct location, no need to handle
        if (system_4dhumans_dir.is_symlink() and 
            system_4dhumans_dir.resolve() == local_cache_path.resolve()):
            logger.info("Cache symbolic link already correctly set")
            return True
        else:
            if system_4dhumans_dir.exists():
                system_4dhumans_dir.unlink()
        
        # Create new symbolic link
        if local_cache_path.exists():
            system_4dhumans_dir.symlink_to(local_cache_path, target_is_directory=True)
            logger.info(f"Created cache symbolic link: {system_4dhumans_dir} -> {local_cache_path}")
            return True
        else:
            logger.error(f"Pre-downloaded cache path does not exist: {local_cache_path}")
            return False
            
    except Exception as e:
        logger.error(f"Setup cache symbolic link failed: {e}")
        return False
