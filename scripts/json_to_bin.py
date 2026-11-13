#!/usr/bin/env python3
"""Pack a WebSR weights JSON into the A4K binary weight format.

Used for both Anime4K and SR2-XS networks — they share the same on-the-wire shape:
each layer is identified by name and carries a `weights` Float32 array and a `bias`
Float32 array. Layers without `weights`/`bias` (e.g. the SR2-XS `display` sentinel)
are skipped — the runtime only needs the actual weight-bearing layers.

Binary layout (all integers little-endian uint32):
  Header (12 bytes):
    magic       : 4 bytes  = b"A4K\\0"
    version     : u32      = 1
    num_layers  : u32      (count of weight-bearing layers, not all entries in JSON)

  Per layer:
    name_len    : u32
    name        : UTF-8 (name_len bytes)
    num_weights : u32
    num_bias    : u32
    weights     : float32  (num_weights × 4 bytes)
    bias        : float32  (num_bias × 4 bytes)

Decoded by `WebSR.parseBinaryWeights` / `parseAnime4KWeights` (src/weights.ts).

Usage:
  python3 scripts/json_to_bin.py --json weights/anime4k/cnn-2x-s-an.json \\
                                 --out  cnn-2x-s-an.bin
"""

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"A4K\0"
VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--out",  required=True, type=Path)
    args = ap.parse_args()

    with args.json.open() as f:
        data = json.load(f)

    layers = data["layers"]

    # Accept either dict (name → layer) or list (layer with embedded name).
    if isinstance(layers, dict):
        items = layers.items()
    elif isinstance(layers, list):
        items = ((lyr["name"], lyr) for lyr in layers)
    else:
        raise ValueError(f"unexpected 'layers' type: {type(layers).__name__}")

    weight_bearing = [
        (name, lyr) for name, lyr in items
        if isinstance(lyr, dict) and "weights" in lyr and "bias" in lyr
    ]

    with args.out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(weight_bearing)))

        for name, lyr in weight_bearing:
            name_bytes = name.encode("utf-8")
            weights = lyr["weights"]
            bias    = lyr["bias"]

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<II", len(weights), len(bias)))
            f.write(struct.pack(f"<{len(weights)}f", *weights))
            f.write(struct.pack(f"<{len(bias)}f",    *bias))

            print(f"  {name:<20}  weights={len(weights):>5}  bias={len(bias):>4}")

    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out}  ({size_kb:.1f} KB, {len(weight_bearing)} layers)")


if __name__ == "__main__":
    main()
