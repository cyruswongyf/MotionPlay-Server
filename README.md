# MotionPlay Server

A real-time human pose estimation server built with 4D-Humans and WebSocket communication for interactive motion capture applications.

## Architecture

```
MotionPlay-Server/
├── src/                    # Core server implementation
│   ├── server.py          # WebSocket server
│   ├── pose_estimator.py  # 4D-Humans integration
│   ├── camera.py          # Camera handling
│   ├── config.py          # Configuration management
│   └── utils.py           # Utility functions
├── libs/
│   ├── 4D-Humans/         # 4D-Humans library
│   └── model_cache/       # Pre-trained models (LFS)
├── start.py               # Server entry point
└── test_client.py         # WebSocket test client
```

## Quick Start

### Prerequisites

- Python 3.10 (recommended)
- Conda (for environment management)
- Git LFS (for model files)
- CUDA-compatible GPU (optional, for better performance)

### Installation

1. **Clone the repository** (with LFS support for model files):

   ```bash
   git lfs clone https://github.com/cyruswongyf/MotionPlay-Server.git
   cd MotionPlay-Server
   ```

2. **Set up Python environment**:

   ```bash
   # Create and activate conda environment
   cd libs/4D-Humans
   conda create --name motionplay python=3.10
   conda activate motionplay

   # Install dependencies
   pip install torch torchvision
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   pip install -e .[all]
   cd ../..
   ```

3. **Install additional dependencies**:
   ```bash
   conda activate motionplay
   pip install websockets opencv-python numpy
   ```

### Running the Server

1. **Start the server**:

   ```bash
   conda activate motionplay
   python start.py
   ```

2. **Test with client**:
   ```bash
   conda activate motionplay
   python test_client.py
   ```

## Models and Data

This project uses pre-trained models from 4D-Humans stored in `libs/model_cache/`. These files are managed with Git LFS:

- **SMPL model**: Human body shape model
- **HMR2 checkpoint**: Pre-trained pose estimation weights
- **Detectron2 model**: Person detection model

## Acknowledgments

- [4D-Humans](https://github.com/shubham-goel/4D-Humans) for the pose estimation model
- [SMPL](https://smpl.is.tue.mpg.de/) for the human body model
- [Detectron2](https://github.com/facebookresearch/detectron2) for person detection
