# Custom Training for WebSR

This directory contains everything you need to train custom super-resolution models for WebSR and export them to the WebGPU format.

## Overview

WebSR uses Convolutional Neural Networks (CNNs) inpsired by the [Anime4K](https://github.com/bloc97/Anime4K) architecture to perform real-time video upscaling. While we provide pre-trained weights for Animation, Real Life, and 3D content, you can train your own models for specific content types or quality improvements.

## Prerequisites

### Software Requirements

- Python 3.8+
- TensorFlow 2.9+ (with GPU support recommended)
- Jupyter Notebook or JupyterLab
- NumPy, PIL, matplotlib

### Install Dependencies

```bash
pip install tensorflow jupyter numpy pillow matplotlib
```

For GPU acceleration (highly recommended):
```bash
pip install tensorflow-gpu
```

### Hardware Requirements

- **GPU**: Training on a GPU is strongly recommended. CPU training will be extremely slow.
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: Depends on your dataset size. Plan for several GB for a decent training dataset.

## Quick Start

### 1. Prepare Your Dataset

Create a dataset of high-quality images that represent the type of content you want to upscale:

- **For Animation**: Collect high-resolution anime screenshots or frames
- **For Real Life**: Use natural photos or video frames
- **For 3D/Gaming**: Gather game screenshots or 3D rendered content

**Dataset Structure:**
```
/your-dataset/
  ├── image001.png
  ├── image002.png
  ├── image003.jpg
  └── ...
```

**Recommendations:**
- Use at least 500-1000 high-quality images
- Images should be at least 512x512 pixels
- Use PNG or high-quality JPEG to avoid compression artifacts
- Ensure images represent diverse scenes, colors, and textures

### 2. Train Your Model

Open `Train_Model.ipynb` in Jupyter:

```bash
jupyter notebook Train_Model.ipynb
```

**Key steps in the notebook:**

1. **Configure dataset path** - Point to your image directory
2. **Choose network size** - Small (S), Medium (M), or Large (L)
   - **Small**: Fast, good for real-time performance (default)
   - **Medium**: Balance of quality and speed
   - **Large**: Best quality, slower inference
3. **Set training parameters** - Epochs, batch size, learning rate
4. **Run training** - Execute all cells to train the model
5. **Save checkpoint** - Model saved as `model-checkpoint.h5`

**Training Tips:**
- Start with the Small network architecture for faster iteration
- Use a batch size that fits in your GPU memory (try 8 or 16)
- Train for at least 50-100 epochs for good results
- Monitor the loss curve - it should steadily decrease
- Save checkpoints periodically during long training runs

### 3. Export Weights to JSON

Once training is complete, export the model weights:

Open `Export_Weights.ipynb`:

```bash
jupyter notebook Export_Weights.ipynb
```

**The notebook will:**
1. Load your trained model checkpoint
2. Convert TensorFlow weights to WebGPU format
3. Export to JSON file (e.g., `my-custom-weights.json`)

**Alternative: Direct Python Export**

You can also use the direct export script:

```python
from direct_webgpu_export import export_tensorflow_to_webgpu
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model-checkpoint.h5')

# Export to WebGPU JSON format
export_tensorflow_to_webgpu(
    model,
    output_file='my-custom-weights.json',
    network_name='custom/my-upscaler'
)
```

### 4. Use Your Custom Weights in WebSR

Copy your exported JSON file to your web project and use it:

```javascript
import WebSR from '@websr/websr';

const gpu = await WebSR.initWebGPU();
if(!gpu) return console.log("WebGPU not supported");

const websr = new WebSR({
    source: videoElement,
    network_name: "anime4k/cnn-2x-s",  // Use the matching architecture
    weights: await (await fetch('./my-custom-weights.json')).json(),
    gpu,
    canvas: canvasElement
});

await websr.start();
```

## File Reference

### Notebooks

- **`Train_Model.ipynb`** - Complete training pipeline
  - Dataset loading and preprocessing
  - Model architecture definition
  - Training loop with validation
  - Checkpoint saving

- **`Export_Weights.ipynb`** - Weight conversion pipeline
  - Load trained model
  - Export to JSON format for WebSR
  - Debug activation export (optional)

### Python Utilities

- **`utils.py`** - Training utilities
  - `show_images()` - Visualize training samples
  - `preview_dataset()` - Preview data pipeline
  - `degrade_blur_gaussian()` - Simulate blur degradation
  - `degrade_noise_gaussian()` - Add noise to training data
  - Various image processing functions

- **`shaderutils.py`** - GLSL shader export (for MPV/other players)
  - Converts TensorFlow models to GLSL shaders
  - Useful for integration with video players like MPV

- **`direct_webgpu_export.py`** - Direct TensorFlow → WebGPU export
  - `export_tensorflow_to_webgpu()` - Main export function
  - `export_debug_activations()` - Export intermediate outputs for debugging
  - Handles 4-channel chunking and weight format conversion

### Model Files

- **`model-checkpoint.h5`** - Example trained model checkpoint

## Network Architectures

WebSR currently supports these architectures (based on Anime4K):

### Upscaling Networks (2x)
- **cnn-2x-s** (Small): 4 Conv2D layers, ~8K parameters
- **cnn-2x-m** (Medium): 4 Conv2D layers with more filters
- **cnn-2x-l** (Large): 4 Conv2D layers with even more filters

### Restoration Networks (1x)
- **cnn-restore-s/m/l**: Same architectures but for quality enhancement without upscaling

**Architecture Pattern:**
```
Input (3 channels RGB)
  ↓
Conv2D (3x3, N filters) + ReLU
  ↓
Conv2D (3x3, N filters) + ReLU
  ↓
Conv2D (3x3, N filters) + ReLU
  ↓
Conv2D (3x3, 12 filters for 2x upscaling)
  ↓
Pixel Shuffle (2x2) + Residual Connection
  ↓
Output (upscaled image)
```

## Training Data Augmentation

The training pipeline supports various degradation techniques to make your model robust:

### Blur Degradation
```python
from utils import degrade_blur_gaussian
degraded_img = degrade_blur_gaussian(img, sigma=1.5, shape=(7, 7))
```

### Noise Addition
```python
from utils import degrade_noise_gaussian
noisy_img = degrade_noise_gaussian(img, sigma=0.01)
```

### JPEG Compression Artifacts
```python
from utils import degrade_rgb_to_yuv, degrade_yuv_to_rgb
yuv = degrade_rgb_to_yuv(img, jpeg_factor=85, chroma_subsampling=True)
compressed_img = degrade_yuv_to_rgb(yuv)
```

## Advanced Topics

### Custom Network Architecture

To create a custom architecture, modify the model definition in `Train_Model.ipynb`. Key considerations:

1. **4-channel alignment**: Output channels must be multiples of 4 for WebGPU
2. **Kernel size**: 3x3 kernels are most efficient in WebGPU
3. **Activation functions**: ReLU is directly supported in the export pipeline
4. **Residual connections**: Supported via the "lastresid" naming convention

### Debugging Your Model

Export debug activations to compare TensorFlow vs WebGPU outputs:

```python
from direct_webgpu_export import export_debug_activations

# Use a test image
test_input = np.expand_dims(test_image, axis=0)

# Export intermediate activations
export_debug_activations(model, test_input, 'debug-activations.json')
```

Compare these activations with WebGPU outputs to verify correct weight conversion.
