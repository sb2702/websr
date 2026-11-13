# Weights

`network_name` picks the architecture; the weights are passed separately. All weights live in this
repo under `weights/` (they are not part of the npm package). Load the binary files with
`WebSR.parseBinaryWeights(arrayBuffer)`, which reads the format from the file header. The JSON
files are passed to `new WebSR({weights})` as parsed JSON.

## Loading weights

```javascript
import WebSR from '@websr/websr';

const url = 'https://cdn.jsdelivr.net/gh/sb2702/websr@v0.1.0/weights/websr/cnn-m.bin';
const weights = WebSR.parseBinaryWeights(await (await fetch(url)).arrayBuffer());

const gpu = await WebSR.initWebGPU();
const websr = new WebSR({ network_name: 'websr/cnn-m', weights, gpu, canvas });
await websr.render(video);
```

## Files

Canonical URL: `https://cdn.jsdelivr.net/gh/sb2702/websr@v0.1.0/<path>` (jsDelivr's GitHub CDN,
pinned to the `v0.1.0` tag; jsDelivr serves GitHub files up to 20 MB each).
Mirror: `https://huggingface.co/katana-video/websr/resolve/main/<path>`.

| Network name | Path | Format | Size | sha256 |
|---|---|---|---|---|
| `websr/cnn-xs` (alias `sr2/cnn-2x-xs`) | `weights/websr/cnn-xs.bin` | A4K | 5.8 KB | `90eaa49bf7b4f41376f9830775d7ba5846e03d9ae21f7d2c5ac180a84de12ff0` |
| `websr/cnn-s` | `weights/websr/cnn-s.bin` | A4K | 35 KB | `3292fbd7ab62c862bd44e59b3aa3e69bed51f6d90441f5d8c904c400c7842b00` |
| `websr/cnn-m` (alias `anime4k/cnn-2x-16`) | `weights/websr/cnn-m.bin` | A4K | 125 KB | `f5d1c334dd486640c562fc6390cb89a6258f1c057d58fd8d56c36d18ac1c3cf1` |
| `websr/cnn-l` (alias `anime4k/cnn-2x-28`) | `weights/websr/cnn-l.bin` | A4K | 364 KB | `a809e0b639a7a3ede91c288d9fbf608b73c979802eae7b144ae9305551b5cd07` |
| `websr/gan-s` | `weights/websr/gan-s.bin` | UNT | 17.2 MB | `94982de53a6e958fa85922ff68e63c47e57a40d9ebf383abaeb1740d4a7090f1` |
| `websr/gan-m` | `weights/websr/gan-m.bin` | UNT | 16.4 MB | `5cf3548391bac234df20a5fe130d27295c48c982c5b910a12eba2a0d42ffb865` |
| `realesrgan/x4v3` | `weights/realesrgan/x4v3.bin` | X4V | 4.9 MB | `adf05e65b2988f92d16fe988a05b450fab098261dae62e9f50fb3da6e35969cf` |
| `anime4k/cnn-2x-s/m/l`, `anime4k/cnn-restore-s/m/l` | `weights/anime4k/*.json` | JSON | 13–115 KB | Anime4K's original weights (`*-an`) plus Real Life (`*-rl`) and 3D (`*-3d`) retrains |

`websr/cnn-s` uses the same architecture as `anime4k/cnn-2x-l`, with WebSR-trained weights;
`anime4k/cnn-2x-l` with `weights/anime4k/cnn-2x-l-an.json` is Anime4K's original.

`weights/realesrgan/x4v3.bin` is Real-ESRGAN's `realesr-general-x4v3.pth` (BSD-3-Clause, xinntao)
repacked with `scripts/pth_to_bin.py`; you can rebuild it from the
[Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases). The `websr/gan-*` UNets
are distilled from that model.

## Formats

- **A4K**: named layers, each a float32 `weights` and `bias` array. `scripts/json_to_bin.py`
  converts the JSON weights to this format.
- **X4V**: ordered conv / conv+PReLU layers for `realesrgan/x4v3`. Written by `scripts/pth_to_bin.py`.
- **UNT**: a JSON execution graph (buffers, convs, gate weights, default alpha) followed by a
  float32 payload, for the `websr/gan-*` UNets.

The byte layouts are documented in `src/weights.ts` and the two scripts.
