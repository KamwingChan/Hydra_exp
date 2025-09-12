# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### C++ Build (CMake)
```bash
# Build from source
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run demo with config
./demo_segmentation path/to/config.yaml
```

### Python Build & Install
```bash
# Install package in development mode
pip install -e .

# Install with extras
pip install -e ".[dev,openset,sam]"

# Run CLI commands
semantic-inference --help
```

### Testing
```bash
# Run C++ tests
cd build && ctest

# Run Python tests
pytest python/test/
pytest python/test/ -m "not slow"  # skip slow tests

# Run specific test
pytest python/test/test_openset_segmenter.py::test_specific_function
```

## Architecture Overview

This is a semantic segmentation inference library with both C++ and Python APIs, built for TensorRT acceleration and ROS/catkin integration.

### Core Components

**C++ Layer (src/)**
- `Segmenter`: Main inference interface using TensorRT models
- `ModelConfig`: Model configuration management
- `ImageUtilities`: Preprocessing (resize, normalize, color conversion)
- `ImageRecolor`: Post-processing for visualization
- TensorRT utilities for model loading and optimization

**Python Layer (python/semantic_inference/)**
- `OpenSetSegmenter`: Python wrapper for open-set segmentation
- `PatchExtractor`: Extract and process image patches
- CLI commands: `color`, `labelspace`, `model` for dataset/model utilities
- Visualization tools for segmentation results

### Model Configuration

Models defined in YAML configs under `config/models/`:
- ONNX models in `models/` directory
- TensorRT engines in `engines/` directory (pre-built)
- Standard datasets: ADE20K (150 classes), MPCAT40 (40 classes)

### Key Paths

- **Configs**: `config/models/*.yaml` - model definitions
- **Models**: `models/*.onnx` - ONNX model files
- **Engines**: `engines/*.trt` - TensorRT optimized engines
- **Resources**: `resources/` - label mappings and color schemes

### Usage Patterns

**C++ Inference**:
```cpp
Segmenter segmenter(config);  // Config from YAML
auto result = segmenter.infer(image);  // cv::Mat input
```

**Python Inference**:
```python
from semantic_inference import OpenSetSegmenter
segmenter = OpenSetSegmenter(model_config_path)
labels = segmenter(image)  # PIL.Image or np.array input
```

### Dependencies

- **C++**: OpenCV, TensorRT, CUDA, config_utilities
- **Python**: torch, torchvision, click, onnx, numpy
- **Optional**: ultralytics, CLIP, segment-anything for open-set features