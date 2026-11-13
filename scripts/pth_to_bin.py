#!/usr/bin/env python3
"""Pack realesr-general-x4v3.pth into the WebSR x4v3 binary weight format.

Binary layout (all integers little-endian uint32):
  Header (16 bytes):
    magic       : 4 bytes  = b"X4V\\0"
    version     : u32      = 1
    num_layers  : u32
    reserved    : u32      = 0

  Per layer:
    name_len           : u32
    name               : UTF-8 (name_len bytes)
    type               : u32     0 = conv, 1 = conv_prelu
    in_channels        : u32
    out_channels       : u32
    num_matrix_floats  : u32     = 9 * ceil(Cin/4) * ceil(Cout/4) * 16
    num_bias_floats    : u32     = ceil(Cout/4) * 4
    num_alpha_floats   : u32     = ceil(Cout/4) * 4 (only if type == conv_prelu, else 0)
    matrices           : float32 (num_matrix_floats × 4 bytes)
    bias               : float32 (num_bias_floats × 4 bytes)
    alpha              : float32 (num_alpha_floats × 4 bytes, may be 0-length)

Matrices follow the same column-major packing as the JSON converter so that the
WGSL kernel reads `mat4x4f` directly. Decoded by `WebSR.parseBinaryWeights`
(src/weights.ts).

Get realesr-general-x4v3.pth from the Real-ESRGAN releases
(https://github.com/xinntao/Real-ESRGAN, BSD-3-Clause).

Usage:
  python3 scripts/pth_to_bin.py --pth /path/to/realesr-general-x4v3.pth \\
                                --out realesr-x4v3.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"X4V\0"
VERSION = 1
TYPE_CONV = 0
TYPE_CONV_PRELU = 1

NUM_FEAT = 64
NUM_CONV = 32


def pack_conv_matrices(weight: torch.Tensor) -> list[float]:
    """Pack (Cout, Cin, 3, 3) into column-major mat4x4f layout used by WGSL.
    Order: 9 spatial × in_vec4s × out_vec4s × 16 floats.
      flat[col*4 + row] = W[out=ov*4+row, in=iv*4+col, ky, kx]   (zero-pad oc/ic)
    """
    Cout, Cin, kH, kW = weight.shape
    assert kH == 3 and kW == 3
    in_vec4s = (Cin + 3) // 4
    out_vec4s = (Cout + 3) // 4
    w = weight.detach().cpu().numpy()
    flat: list[float] = []
    for ky in range(3):
        for kx in range(3):
            for iv in range(in_vec4s):
                for ov in range(out_vec4s):
                    for col in range(4):
                        ic = iv * 4 + col
                        for row in range(4):
                            oc = ov * 4 + row
                            v = float(w[oc, ic, ky, kx]) if (oc < Cout and ic < Cin) else 0.0
                            flat.append(v)
    return flat


def pad_to_vec4(arr_1d, target_len) -> list[float]:
    out = [0.0] * target_len
    for i, v in enumerate(arr_1d):
        out[i] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ckpt = torch.load(str(args.pth), map_location="cpu", weights_only=True)
    state = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt

    last_idx = max(int(k.split(".")[1]) for k in state if k.startswith("body."))
    assert last_idx == 2 * (NUM_CONV + 1), f"unexpected body span: 0..{last_idx}"

    fused_pairs = [(2 * i, 2 * i + 1) for i in range(NUM_CONV + 1)]   # (conv, prelu) pairs
    num_layers = len(fused_pairs) + 1                                  # +1 final plain conv

    with args.out.open("wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, num_layers, 0))

        # Fused conv+PReLU layers
        for conv_idx, prelu_idx in fused_pairs:
            w = state[f"body.{conv_idx}.weight"]
            b = state[f"body.{conv_idx}.bias"]
            alpha = state[f"body.{prelu_idx}.weight"]
            Cout, Cin, _, _ = w.shape
            out_vec4s = (Cout + 3) // 4

            name = f"body.{conv_idx}+{prelu_idx}".encode("utf-8")
            matrices = pack_conv_matrices(w)
            bias = pad_to_vec4(b.detach().cpu().numpy().tolist(), out_vec4s * 4)
            alpha_padded = pad_to_vec4(alpha.detach().cpu().numpy().tolist(), out_vec4s * 4)

            f.write(struct.pack("<I", len(name)))
            f.write(name)
            f.write(struct.pack("<IIIIII",
                                TYPE_CONV_PRELU,
                                int(Cin), int(Cout),
                                len(matrices), len(bias), len(alpha_padded)))
            f.write(struct.pack(f"<{len(matrices)}f", *matrices))
            f.write(struct.pack(f"<{len(bias)}f", *bias))
            f.write(struct.pack(f"<{len(alpha_padded)}f", *alpha_padded))
            print(f"  {name.decode():<14} conv_prelu  {Cin:>3}→{Cout:>3}ch  "
                  f"matrices={len(matrices)} bias={len(bias)} alpha={len(alpha_padded)}")

        # Final plain conv (body.66 — no activation, 64→48)
        w = state[f"body.{last_idx}.weight"]
        b = state[f"body.{last_idx}.bias"]
        Cout, Cin, _, _ = w.shape
        out_vec4s = (Cout + 3) // 4
        name = f"body.{last_idx}".encode("utf-8")
        matrices = pack_conv_matrices(w)
        bias = pad_to_vec4(b.detach().cpu().numpy().tolist(), out_vec4s * 4)

        f.write(struct.pack("<I", len(name)))
        f.write(name)
        f.write(struct.pack("<IIIIII",
                            TYPE_CONV,
                            int(Cin), int(Cout),
                            len(matrices), len(bias), 0))
        f.write(struct.pack(f"<{len(matrices)}f", *matrices))
        f.write(struct.pack(f"<{len(bias)}f", *bias))
        print(f"  {name.decode():<14} conv        {Cin:>3}→{Cout:>3}ch  "
              f"matrices={len(matrices)} bias={len(bias)} alpha=0")

    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size_mb:.2f} MB, {num_layers} layers)")


if __name__ == "__main__":
    main()
