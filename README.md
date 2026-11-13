# WebSR

A free and open-source Javascript library / SDK for running practical, real-time AI Upscaling of videos and images on the web.

<img width="640" title="Demo clip from Hero by the Blender Foundation, licensed under Creative Commons 4.0 Attribution licensed" alt="Demo clip from Hero by the Blender Foundation, licensed under Creative Commons 4.0 Attribution licensed" src="https://github.com/sb2702/websr/assets/5678502/82e6e764-89f4-4c8d-b8f6-dcd43d43c0f3">


[Check out the SDK in action in Free AI Video Upscaler!](https://free.upscaler.video/)


**Warning**
This project is very new and still in development, and shouldn't be used (yet) in production until stable. Expect API changes between versions and bugs

## Features

- ✅ Real-time AI upscaling using WebGPU compute shaders
- ✅ Multiple network sizes (XS to L) for performance/quality tradeoffs, plus Real-ESRGAN x4v3 and UNets distilled from it
- ✅ Pre-trained weights for Animation, Real Life, and 3D content
- ✅ Automatic tiling of large images/frames to stay within GPU memory limits, with per-tile progress
- ✅ Input downscaling (`inputScale`) and arbitrary output sizes (`outputResolution`)
- ✅ f16 compute on devices with `shader-f16`
- ✅ Works without external textures (e.g. Firefox's WebGPU) by falling back to image copies
- ✅ Compact binary weight formats
- ✅ OffscreenCanvas support for worker thread processing
- ✅ Custom training pipeline - train your own models
- ✅ Manual render control for advanced use cases
- ✅ Dynamic network switching at runtime


### Installation

**npm**

```javascript
npm install @websr/websr
```


**yarn**

```javascript
yarn add @websr/websr
```


### Quick Start

```javascript
import WebSR from '@websr/websr';

const video = document.getElementById('video'); // HTML Video Element
const canvas = document.getElementById('canvas'); // Output canvas

const gpu = await WebSR.initWebGPU();

if(!gpu) return console.log("Browser/device doesn't support WebGPU");

const websr = new WebSR({
    network_name: "anime4k/cnn-2x-s",
    weights: await (await fetch('./cnn-2x-s.json')).json(), //found in weights/anime4k folder
    gpu,
    canvas
});

// Render loop using requestVideoFrameCallback
function renderFrame() {
    websr.render(video).then(() => {
        video.requestVideoFrameCallback(renderFrame);
    });
}

video.requestVideoFrameCallback(renderFrame);
```

`render()` accepts an `HTMLVideoElement`, `VideoFrame`, `HTMLImageElement` or `ImageBitmap`.

**Note:** Resolution is automatically detected from the source on first render. The canvas will be automatically resized to match the upscaled output (input resolution × network scale factor). If the source resolution changes, WebSR will automatically reinitialize with the new dimensions.


### OffscreenCanvas / Worker Thread

WebSR is compatible with OffscreenCanvas, allowing you to run the upscaling in a worker thread to keep your main thread responsive:

```javascript
// main.js
const canvas = document.getElementById('output');
const offscreen = canvas.transferControlToOffscreen();
const worker = new Worker('websr-worker.js');
const video = document.getElementById('video');

// Send canvas and video frames to worker
worker.postMessage({ canvas: offscreen }, [offscreen]);

// Use requestVideoFrameCallback to send frames to worker
function sendFrame() {
    createImageBitmap(video).then(bitmap => {
        worker.postMessage({ frame: bitmap }, [bitmap]);
        video.requestVideoFrameCallback(sendFrame);
    });
}
video.requestVideoFrameCallback(sendFrame);
```

```javascript
// websr-worker.js
importScripts('./websr.js');

let websr;

self.onmessage = async function(e) {
    if(e.data.canvas) {
        // Initialize WebSR in worker
        const gpu = await WebSR.initWebGPU();
        websr = new WebSR({
            network_name: "anime4k/cnn-2x-s",
            weights: await (await fetch('./cnn-2x-s.json')).json(),
            gpu,
            canvas: e.data.canvas
        });
    } else if(e.data.frame && websr) {
        // Render frame
        await websr.render(e.data.frame);
    }
};
```


### Networks

`network_name` picks the architecture; the weights are passed separately (see [WEIGHTS.md](./WEIGHTS.md) for every file, its format and where to get it).

| Network | Scale | What it is | Weights |
|---|---|---|---|
| `websr/cnn-xs` | 2× | Tiny depthwise-separable CNN (~1.4k params), 16-channel CReLU trunk + bilinear residual. Fastest | `weights/websr/cnn-xs.bin` |
| `websr/cnn-s` | 2× | Anime4K CNN-2x-L architecture (8 channels), WebSR-trained weights | `weights/websr/cnn-s.bin` |
| `websr/cnn-m` | 2× | Same design widened to 16 channels | `weights/websr/cnn-m.bin` |
| `websr/cnn-l` | 2× | Same design widened to 28 channels | `weights/websr/cnn-l.bin` |
| `websr/gan-s` | 2× | UNet distilled from Real-ESRGAN x4v3 (24-48-96 channels). Real-life / detail content | `weights/websr/gan-s.bin` (17 MB) |
| `websr/gan-m` | 2× | UNet distilled from Real-ESRGAN x4v3 (32-64-128 channels) | `weights/websr/gan-m.bin` (16 MB) |
| `realesrgan/x4v3` | 4× | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesr-general-x4v3` (SRVGGCompact, 34 convs, 64 channels). Heaviest | `weights/realesrgan/x4v3.bin` (4.9 MB) |
| `anime4k/cnn-2x-s` / `-m` / `-l` | 2× | [Anime4K](https://github.com/bloc97/Anime4K) upscalers, with Anime4K's original weights (`*-an.json`) or retrained Real Life (`*-rl`) / 3D (`*-3d`) weights | `weights/anime4k/*.json` |
| `anime4k/cnn-restore-s` / `-m` / `-l` | 1× | Anime4K restore (denoise/sharpen, no upscaling) networks | `weights/anime4k/cnn-restore-*.json` |

`websr/cnn-s` and `anime4k/cnn-2x-l` are the same architecture: use `websr/cnn-s` with the WebSR-trained weights and `anime4k/cnn-2x-l` with Anime4K's original weights. The older names `sr2/cnn-2x-xs`, `anime4k/cnn-2x-16` and `anime4k/cnn-2x-28` still work as aliases of `websr/cnn-xs`, `websr/cnn-m` and `websr/cnn-l`.

`websr/gan-*` and `realesrgan/x4v3` are larger networks aimed at real-life footage and need a reasonably capable GPU; the CNNs are cheap enough for real-time video on integrated graphics.

**Loading weights.** The weights are in this repo (not in the npm package). Fetch them from jsDelivr's GitHub CDN, pinned to a release tag, e.g. `https://cdn.jsdelivr.net/gh/sb2702/websr@v0.1.0/weights/websr/cnn-m.bin` (mirror: `https://huggingface.co/katana-video/websr/resolve/main/weights/websr/cnn-m.bin`). [WEIGHTS.md](./WEIGHTS.md) lists every file with its sha256. `WebSR.parseBinaryWeights()` reads any of WebSR's binary formats (it detects the format from the file):

```javascript
const url = 'https://cdn.jsdelivr.net/gh/sb2702/websr@v0.1.0/weights/websr/cnn-m.bin';
const buffer = await (await fetch(url)).arrayBuffer();

const websr = new WebSR({
    network_name: "websr/cnn-m",
    weights: WebSR.parseBinaryWeights(buffer),
    gpu,
    canvas
});
```

JSON weights (the `weights/anime4k` files) are passed as parsed JSON, as in the Quick Start. `scripts/json_to_bin.py` converts JSON weights to the binary format.

**Per-video calibration (`websr/gan-*`).** The UNets have conditional bottleneck convolutions: several kernel "banks" per layer, blended once per video by a small gate network. Call `calibrate()` with a few representative frames; everything runs on the GPU and each later frame costs the same as a plain network:

```javascript
const weights = WebSR.parseBinaryWeights(await (await fetch('https://cdn.jsdelivr.net/gh/sb2702/websr@v0.1.0/weights/websr/gan-s.bin')).arrayBuffer());
const websr = new WebSR({ network_name: 'websr/gan-s', weights, gpu, canvas });

await websr.calibrate(frames);   // optional, once per video: VideoFrame[] or ImageBitmap[]
await websr.render(source);
```

Calibration is optional (without it the UNets use their training-average blend). The thumbnails are kept, so a later `switchNetwork()` to the other UNet re-applies it; you can close the `VideoFrame`s once `calibrate()` resolves. For other networks it just stores the thumbnails and returns `null`.


### Options and methods

#### `new WebSR(params)`

| Option | Type | Required | Description |
|---|---|---|---|
| `canvas` | `HTMLCanvasElement \| OffscreenCanvas` | Yes | Output canvas. It is resized to the output resolution |
| `gpu` | `GPUDevice` | Yes | From `WebSR.initWebGPU()` |
| `network_name` | `string` | Yes | See [Networks](#networks) |
| `weights` | `object` | Yes | Parsed JSON, or the result of `WebSR.parseBinaryWeights()` |
| `resolution` | `{width, height}` | No | Source resolution, to set up in the constructor instead of on the first `render()` |
| `outputResolution` | `{width, height}` | No | Canvas size. Defaults to input × network scale; other sizes are reached with a GPU bicubic resample of the network output |
| `inputScale` | `number` | No | Resample the source by `1/inputScale` on the GPU (antialiased bicubic) before the network, e.g. `2` runs a 1080p source at 960×540. The output defaults to the resampled size × scale; pass `outputResolution` to keep the original output size |
| `tier` | `{tier, isMobile}` | No | GPU capability tier (e.g. from [detect-gpu](https://github.com/pmndrs/detect-gpu)). Sizes the tile memory budget: tier 0 ≈ 50 MB, 1 ≈ 100 MB (default), 2 ≈ 250 MB, 3 ≈ 1 GB of network buffers per tile. Mobile devices drop one tier |
| `precision` | `'auto' \| 'f32' \| 'f16'` | No | Compute precision for `realesrgan/x4v3` and `websr/gan-*`. `'auto'` (default) uses f16 when the device has `shader-f16` (which `WebSR.initWebGPU()` requests) |
| `debug` | `boolean` | No | Debug mode (readable buffers) |

#### Instance methods

| Method | Description |
|---|---|
| `render(source, onTileProgress?)` | Upscale one frame/image to the canvas. If the source resolution changes, WebSR re-plans automatically. `onTileProgress(done, total)` fires as each tile finishes on the GPU (useful for large images) |
| `switchNetwork(name, weights)` | Switch network at runtime (also before the first render) |
| `setInputScale(k?)` | Change `inputScale`; applied on the next `render()` |
| `setOutputResolution(res?)` | Change the output size; `undefined` restores the default |
| `calibrate(frames)` | Per-video calibration for `websr/gan-*` (see above). Returns the blend weights, or `null` for other networks |
| `destroy()` | Destroy the GPU device and release all resources |

#### Static methods

| Method | Description |
|---|---|
| `WebSR.initWebGPU()` | Returns a `GPUDevice` (with `shader-f16` when available), or `false` if WebGPU isn't supported |
| `WebSR.parseBinaryWeights(buffer)` | Parse any binary weight file (A4K, X4V or UNT, detected from the file). `parseAnime4KWeights`, `parseX4VWeights` and `parseUNetWeights` parse one format each |

#### Tiling

Large inputs are split into tiles automatically so each network's buffers fit the GPU memory budget (see `tier`). Each tile is padded so that seams don't show, and the grid is chosen to minimise the number of tiles. You don't need to configure anything; pass `onTileProgress` to `render()` to show progress on big images.

#### Browsers without external textures

Where `GPUDevice.importExternalTexture` isn't available (e.g. Firefox's WebGPU at the time of writing), video sources are copied in as `ImageBitmap`s instead. This is detected once per device and needs no configuration.


### Content detection

For convenience, I've also included a [content detection style network](https://github.com/sb2702/websr/blob/main/weights/tflite/content_detection_mobilenet_v3.tflite) implemented in TFLite. You can use it as follows:

        import '@tensorflow/tfjs-backend-cpu';
        import * as tf from '@tensorflow/tfjs-core';
        import * as tflite from '@tensorflow/tfjs-tflite';

        const contentDetectionCanvas = document.createElement('canvas');
        contentDetectionCanvas.width = 224;
        contentDetectionCanvas.height = 224;
        const contentDetectionCtx = contentDetectionCanvas.getContext('2d', {willReadFrequently: true});

        const tfliteModel = await tflite.loadTFLiteModel('./content_detection_mobilenet_v3.tflite',  {} );

        const source = //video, image, imageBitmap, etc... 

        contentDetectionCtx.drawImage(source, source.width/2-112, source.height/2-112, 224, 224, 0, 0, 224, 224 ); // Take the center 224x224 crop of the source

        
        const input = tf.div(tf.expandDims(img), 255);

        const outputTensor = tfliteModel.predict(input);

        const values = outputTensor.dataSync();

        if(values[1] > values[0]) //animation
        else: //real life

        // I tried building a 3-network classifier, but ran into some accracy issues









### Why WebSR

[Super resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging) is a technique which uses Neural Networks to "reconstruct" a high-resolution image from a low resolution image. Practically speaking, it provides better visual results than traditional upscaling algorithms like Bicubic, which is what devices and browsers usually use by default, at the cost of significantly more computation.

A general rule of thumb is that bigger super-resolution networks (which use more computation) generally provide better results.

<img src="https://github.com/sb2702/websr/assets/5678502/6e040e84-cf6d-4cf0-a45c-638b41d011ec" width="400" />

This is why commercial AI upscaling software like [Topaz Labs](https://www.topazlabs.com/) can produce excellent quality results, though it can take 10 seconds or more on a dedicated GPU to upscale a single image. 
If the goal is to run super resolution on a video at 30 frames per second on mid-tier integrated graphcs cards in the browser, you need networks that run thousands of times faster. Computation is a huge challenge and a big reason that super resolution hasn't been more popular in web video. 

WebSR is an attempt to take on this challenge, and make super resolution practical in the browser. By taking advantage of the fact that super-resolution has proved relatively better at improving the video quality of Anime, Cartoons and Screen-sharing content, as well the faster graphics computation enabled by [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API), WebSR should be just-about practical enough for production video and WebRTC applicaitons in cases where bandwidth or video quality is an issue.

See [this video](https://www.youtube.com/watch?v=R6Ly0GYlXa4) for more info on super resolution on the web.

### How it works

WebSR uses hand-written WebGPU shaders to implement Convolutional Neural Networks (like [this one](https://github.com/sb2702/websr/blob/main/src/networks/anime4k/cnn-2x-s.ts) from [Anime4K](https://github.com/bloc97/Anime4K)).

Each call to `websr.render(source)` uploads the frame (or splits it into tiles), runs it through the Neural Network and paints the result to a `<canvas>`.

The first networks were ports of [Anime4K](https://github.com/bloc97/Anime4K), chosen because 

1.  It was designed for real time video
2.  Already has GPU shader written in the similar `glsl` language
3.  Already has wide community adoption.

This repo provides the [default production weights](https://github.com/sb2702/websr/blob/main/weights/anime4k/cnn-2x-s.json) asociated with the `cnn-2x-s` network directly from Anime4K.

WebSR has since added its own networks (`websr/cnn-*`, `websr/gan-*`) and a port of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)'s `realesr-general-x4v3`.

### Custom Training

You can train your own custom super-resolution models tailored to your specific content type or quality requirements. The `custom-training/` folder includes:

- **Complete training pipeline** - Jupyter notebooks for training TensorFlow models
- **Weight export tools** - Convert trained models to WebGPU format
- **Training utilities** - Data augmentation, degradation simulation, and visualization
- **Pre-configured architectures** - Small, Medium, and Large network options

**Quick Start:**

1. Prepare a dataset of high-quality images (500-1000+ images recommended)
2. Train your model using `custom-training/Train_Model.ipynb`
3. Export weights using `custom-training/Export_Weights.ipynb`
4. Use your custom weights in WebSR (see Quick Start example above)

For detailed instructions, see the [Custom Training README](./custom_training/README.md).

**Use Cases for Custom Training:**
- Content-specific optimization (e.g., medical imaging, satellite imagery)
- Fine-tuning for specific art styles or game graphics
- Balancing quality vs. performance for your target hardware
- Training on proprietary content that differs from the provided weights

### Acknowledgements

1. [Anime4K](https://github.com/bloc97/Anime4K) for an excellent project, first version of networks were direct ports of Anime4K, though the project no longer uses it 
2. [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al. (BSD-3-Clause): `realesrgan/x4v3` is a WebGPU port of `realesr-general-x4v3`, and the `websr/gan-*` UNets are distilled from it
3. [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) for enabling compute-intensive projects like this
4. [Blender foundation](https://studio.blender.org/films/hero/) for use of their hero movie in WebSR demos

### Roadmap


- [x] Add more upscaling networks (especially for other types of content - like real life, or screen-content)
- [x] Dynamically switch between networks based on type of content
- [x] Provide lower level controls (e.g. control over the render loop)
- [x] OffscreenCanvas / worker thread support
- [x] Custom training pipeline and documentation
- [x] Tiling, input/output resolution control, f16 compute
- [x] Real-ESRGAN x4v3 port and distilled UNets
- [ ] Write Mobile SDKs with similar functionality
