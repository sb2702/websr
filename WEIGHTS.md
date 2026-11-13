# Weights

`network_name` picks the architecture; the weights are passed separately. Every file below can be
loaded with `WebSR.parseBinaryWeights(arrayBuffer)` (it reads the format from the file's magic),
except the JSON files, which are passed to `new WebSR({weights})` as parsed JSON.

## In this repo

| Network name | File | Format | Size | Notes |
|---|---|---|---|---|
| `websr/cnn-xs` (alias `sr2/cnn-2x-xs`) | `weights/websr/cnn-xs.bin` | A4K | 5.7 KB | |
| `websr/cnn-s` | `weights/websr/cnn-s.bin` | A4K | 35 KB | Anime4K CNN-2x-L architecture, WebSR-trained weights |
| `websr/cnn-m` | `weights/websr/cnn-m.bin` | A4K | 125 KB | 16-channel variant (`anime4k/cnn-2x-16`) |
| `websr/cnn-l` | `weights/websr/cnn-l.bin` | A4K | 364 KB | 28-channel variant (`anime4k/cnn-2x-28`) |
| `anime4k/cnn-2x-s/m/l`, `anime4k/cnn-restore-s/m/l` | `weights/anime4k/*.json` | JSON | 13–115 KB | Anime4K's original weights (`*-an`) plus Real Life (`*-rl`) and 3D (`*-3d`) retrains |

## Hosted separately (too large for git)

<!-- TODO: upload these and replace the placeholder URLs. -->

| Network name | File | Format | Size | sha256 | URL |
|---|---|---|---|---|---|
| `websr/gan-s` | `gan-s.bin` | UNT | 17.2 MB | `94982de53a6e958fa85922ff68e63c47e57a40d9ebf383abaeb1740d4a7090f1` | TODO: `https://huggingface.co/websr/...` |
| `websr/gan-m` | `gan-m.bin` | UNT | 16.4 MB | `5cf3548391bac234df20a5fe130d27295c48c982c5b910a12eba2a0d42ffb865` | TODO: `https://huggingface.co/websr/...` |
| `realesrgan/x4v3` | `realesr-x4v3.bin` | X4V | 4.9 MB | `adf05e65b2988f92d16fe988a05b450fab098261dae62e9f50fb3da6e35969cf` | TODO: `https://huggingface.co/websr/...` |

For local testing, put these files in `weights-local/` (gitignored).

`realesr-x4v3.bin` is Real-ESRGAN's `realesr-general-x4v3.pth` (BSD-3-Clause, xinntao) repacked
with `scripts/pth_to_bin.py`, so you can also build it yourself from the
[Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases).

## Formats

- **A4K**: named layers, each a float32 `weights` and `bias` array. `scripts/json_to_bin.py`
  converts the JSON weights to this format.
- **X4V**: ordered conv / conv+PReLU layers for `realesrgan/x4v3`. Written by `scripts/pth_to_bin.py`.
- **UNT**: a JSON execution graph (buffers, convs, gate weights, default alpha) followed by a
  float32 payload, for the `websr/gan-*` UNets.

The byte layouts are documented in `src/weights.ts` and the two scripts.
