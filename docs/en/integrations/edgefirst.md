---
comments: true
description: Learn how to train YOLO models that accept native camera formats (BGR, RGBA, YUYV, GREY) directly using EdgeFirst CameraAdaptor for optimized edge deployment.
keywords: EdgeFirst, CameraAdaptor, YOLO, edge deployment, BGR, RGBA, YUYV, camera formats, i.MX, edge AI, Ultralytics, color conversion
---

# EdgeFirst CameraAdaptor Integration

This branch integrates [EdgeFirst CameraAdaptor](https://github.com/EdgeFirstAI/cameraadaptor)
to enable training YOLO models that accept native camera formats directly.

## Why CameraAdaptor?

When deploying models to edge devices, there's often a mismatch between:

1. **Training format**: RGB images from standard datasets
2. **Inference format**: Native camera/hardware formats (BGR, RGBA, YUYV)

**Traditional approach**: Camera → RGB conversion → Model inference

**With CameraAdaptor**: Camera → Model inference (no conversion needed)

This eliminates color conversion overhead, reducing latency and memory bandwidth
on resource-constrained edge devices.

## How It Works

The `CameraAdaptorTransform` converts BGR training images (from cv2.imread) to the target
format during data loading. The `CameraAdaptor` layer in the model handles:

1. **Layout permutation** (NHWC ↔ NCHW) when `channels_last`/`channels_first` is enabled
2. **Alpha channel dropping** for RGBA/BGRA inputs

**Important**: The CameraAdaptor layer does NOT perform color space conversion at inference
time. Color conversion is handled during training data preprocessing, so at inference
the camera/ISP provides data directly in the target format.

## Supported Formats

| Format | Channels | Use Case |
|--------|----------|----------|
| RGB | 3 | Standard (default) |
| BGR | 3 | OpenCV pipelines, i.MX PXP output |
| RGBA | 4→3 | i.MX 8M Plus G2D output |
| BGRA | 4→3 | Graphics APIs |
| GREY | 1 | Infrared cameras, thermal imaging, depth sensors |
| YUYV | 2 | USB cameras, V4L2 devices |

## Usage

### Dataset Configuration

Add `cameraadaptor` to your dataset YAML:

```yaml
# dataset.yaml
train: /path/to/train/images
val: /path/to/val/images
names:
  0: class1
  1: class2

# EdgeFirst CameraAdaptor format
cameraadaptor: bgr  # or: rgba, bgra, grey, yuyv
```

### Model Configuration

CameraAdaptor can be added as the first backbone layer in your model YAML:

```yaml
# yolo11n-bgr.yaml
backbone:
  - [-1, 1, CameraAdaptor, [bgr]]  # First layer
  - [-1, 1, Conv, [64, 3, 2]]
  # ... rest of backbone
```

### Training

```bash
yolo train model=yolo11n.pt data=dataset.yaml
```

The training pipeline automatically:
1. Reads images in BGR format (via cv2.imread)
2. Applies `CameraAdaptorTransform` to convert BGR → target format (YUYV, RGBA, GREY, etc.)
3. Trains the model expecting that format as input

### Python API

```python
from ultralytics import YOLO

# Train with BGR format for i.MX 93 deployment
model = YOLO("yolo11n.pt")
model.train(
    data="dataset.yaml",
    # dataset.yaml includes: cameraadaptor: bgr
)

# Export for edge deployment
model.export(format="tflite", int8=True)
```

## Platform Recommendations

| Platform | Hardware | Recommended Format |
|----------|----------|-------------------|
| i.MX 93 | PXP | `bgr` |
| i.MX 8M Plus | G2D | `rgba` |
| USB Cameras | V4L2/UVC | `yuyv` |
| Thermal/IR Cameras | Sensor | `grey` |

## YOLO Version Support

CameraAdaptor works with all YOLO versions supported by Ultralytics:

| Version | Detection | Segmentation | Pose | OBB |
|---------|-----------|--------------|------|-----|
| YOLOv5 | ✅ | ✅ | ✅ | ✅ |
| YOLOv8 | ✅ | ✅ | ✅ | ✅ |
| YOLO11 | ✅ | ✅ | ✅ | ✅ |
| YOLO26 | ✅ | ✅ | ✅ | ✅ |

## EdgeFirst Studio Integration

This branch is designed to work with [EdgeFirst Studio](https://studio.edgefirst.ai)
via the [edgefirst-studio-ultralytics](https://github.com/EdgeFirstAI/edgefirst-studio-ultralytics)
package, which handles training orchestration, INT8 quantization optimization, and
model metadata embedding.

## Branch Maintenance

This branch is maintained by Au-Zone Technologies and is based on Ultralytics **v8.4.9**.

**Upstream version**: `v8.4.9` (commit `a8b639bc3`)

Changes are minimal and focused exclusively on CameraAdaptor support:

1. **CameraAdaptor module** - Import and registration in `nn/tasks.py`
2. **Data augmentation** - `CameraAdaptorTransform` in `data/augment.py`
3. **Dataset pipeline** - `cameraadaptor` config propagation in `data/dataset.py`
4. **Channel computation** - Automatic input channel detection in `data/utils.py`
5. **Dependencies** - Added `edgefirst-cameraadaptor[torch]` and `numpy<2` constraint in `pyproject.toml`

All CameraAdaptor functionality is provided by the
[edgefirst-cameraadaptor](https://github.com/EdgeFirstAI/cameraadaptor) library.

## Resources

- [EdgeFirst CameraAdaptor Documentation](https://github.com/EdgeFirstAI/cameraadaptor)
- [EdgeFirst HAL Runtime](https://github.com/EdgeFirstAI/hal)
- [EdgeFirst Studio](https://studio.edgefirst.ai)
- [FORMATS.md](https://github.com/EdgeFirstAI/cameraadaptor/blob/main/FORMATS.md) - Color format reference
- [PLATFORMS.md](https://github.com/EdgeFirstAI/cameraadaptor/blob/main/PLATFORMS.md) - Platform guidance
