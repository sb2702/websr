// Parsers for WebSR's binary weight formats. All integers are little-endian uint32.
//
//   "A4K\0"  Anime4K-style networks (anime4k/*, websr/cnn-*, sr2/cnn-2x-xs): named layers,
//            each a float32 `weights` + `bias` array. Written by scripts/json_to_bin.py.
//   "X4V\0"  realesrgan/x4v3: ordered conv / conv+PReLU layers. Written by scripts/pth_to_bin.py.
//   "UNT\0"  UNets (websr/gan-*): JSON graph + float32 payload. See parseUNetWeights.
//
// parseBinaryWeights() reads the magic and dispatches, so callers can pass any of them.

// ─── A4K ─────────────────────────────────────────────────────────────────────
//   Header (12 bytes): magic "A4K\0" | version (1) | num_layers
//   Per layer: name_len | name (UTF-8) | num_weights | num_bias | weights (f32) | bias (f32)

export interface Anime4KWeights {
    name: string;
    layers: Record<string, { weights: Float32Array; bias: Float32Array }>;
}

const MAGIC_A4K = 0x004B3441;  // "A4K\0"

export function parseAnime4KWeights(buffer: ArrayBuffer): Anime4KWeights {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    if (magic !== MAGIC_A4K) {
        throw new Error(`parseAnime4KWeights: unknown magic 0x${magic.toString(16)} (expected A4K\\0)`);
    }
    const version = view.getUint32(4, true);
    if (version !== 1) throw new Error(`parseAnime4KWeights: unsupported version ${version}`);
    const numLayers = view.getUint32(8, true);

    const decoder = new TextDecoder();
    const layers: Anime4KWeights["layers"] = {};
    let off = 12;

    for (let i = 0; i < numLayers; i++) {
        const nameLen = view.getUint32(off, true); off += 4;
        const name = decoder.decode(new Uint8Array(buffer, off, nameLen)); off += nameLen;
        const numWeights = view.getUint32(off, true); off += 4;
        const numBias = view.getUint32(off, true); off += 4;
        // .slice() gives an aligned 0-offset ArrayBuffer (the name can leave off unaligned).
        const weights = new Float32Array(buffer.slice(off, off + numWeights * 4)); off += numWeights * 4;
        const bias = new Float32Array(buffer.slice(off, off + numBias * 4)); off += numBias * 4;
        layers[name] = { weights, bias };
    }

    return { name: "", layers };
}

// ─── X4V ─────────────────────────────────────────────────────────────────────
// See scripts/pth_to_bin.py for the layout spec.

export interface BinaryConvLayer {
    name: string;
    type: "conv" | "conv_prelu";
    in_channels: number;
    out_channels: number;
    in_vec4s: number;
    out_vec4s: number;
    matrices: Float32Array;
    bias: Float32Array;
    alpha?: Float32Array;
}

export interface BinaryWeights {
    version: number;
    layers: BinaryConvLayer[];
}

const MAGIC_X4V = 0x00563458;  // "X4V\0" little-endian: 'X'=0x58, '4'=0x34, 'V'=0x56, '\0'=0x00

// Parse any WebSR binary weight file (A4K, X4V or UNT), dispatching on its magic.
export function parseBinaryWeights(buffer: ArrayBuffer): BinaryWeights | Anime4KWeights | UNetWeights {
    if (buffer.byteLength < 4) throw new Error("parseBinaryWeights: file too short");
    const magic = new DataView(buffer).getUint32(0, true);
    if (magic === MAGIC_A4K) return parseAnime4KWeights(buffer);
    if (magic === MAGIC_UNT) return parseUNetWeights(buffer);
    if (magic === MAGIC_X4V) return parseX4VWeights(buffer);
    throw new Error(`parseBinaryWeights: unknown magic 0x${magic.toString(16)} (expected A4K, X4V or UNT)`);
}

export function parseX4VWeights(buffer: ArrayBuffer): BinaryWeights {
    const view = new DataView(buffer);
    let off = 0;

    const magic = view.getUint32(off, true); off += 4;
    if (magic !== MAGIC_X4V) {
        throw new Error(`parseX4VWeights: unknown magic 0x${magic.toString(16)} (expected X4V\\0)`);
    }
    const version = view.getUint32(off, true); off += 4;
    if (version !== 1) {
        throw new Error(`parseX4VWeights: unsupported version ${version}`);
    }
    const numLayers = view.getUint32(off, true); off += 4;
    off += 4;  // reserved

    const decoder = new TextDecoder();
    const layers: BinaryConvLayer[] = [];

    for (let i = 0; i < numLayers; i++) {
        const nameLen = view.getUint32(off, true); off += 4;
        const name = decoder.decode(new Uint8Array(buffer, off, nameLen)); off += nameLen;

        const typeTag = view.getUint32(off, true); off += 4;
        const inCh    = view.getUint32(off, true); off += 4;
        const outCh   = view.getUint32(off, true); off += 4;
        const nMat    = view.getUint32(off, true); off += 4;
        const nBias   = view.getUint32(off, true); off += 4;
        const nAlpha  = view.getUint32(off, true); off += 4;

        // .slice() guarantees an aligned 0-offset ArrayBuffer for the typed-array view,
        // sidestepping any alignment issues from the variable-length name field.
        const matrices = new Float32Array(buffer.slice(off, off + nMat * 4));   off += nMat * 4;
        const bias     = new Float32Array(buffer.slice(off, off + nBias * 4));  off += nBias * 4;
        const alpha    = nAlpha > 0
            ? new Float32Array(buffer.slice(off, off + nAlpha * 4))
            : undefined;
        off += nAlpha * 4;

        layers.push({
            name,
            type: typeTag === 1 ? "conv_prelu" : "conv",
            in_channels: inCh,
            out_channels: outCh,
            in_vec4s: Math.ceil(inCh / 4),
            out_vec4s: Math.ceil(outCh / 4),
            matrices,
            bias,
            alpha,
        });
    }

    return { version, layers };
}


// ─── UNet ("UNT\0") ────────────────────────────────────────────────────────────
// Written by the UNet export step of the training pipeline. Container:
//   u32 magic, u32 version (1), u32 json_len, json (utf-8, space-padded to 4), float32 payload.
// The JSON holds the whole execution graph (buffers + ordered convs + slices), the gate
// weights and the default alpha; every tensor is {offset, count} in floats into the payload.

export interface TensorRef { offset: number; count: number; }

export interface UNetLayerSpec {
    name: string;
    mode: "same" | "s2" | "up";
    cin: number; cout: number;
    in_vec4s: number; out_vec4s: number;
    crelu: boolean;
    banked: boolean;
    in_buf: string; in_offset: number;     // offsets in vec4s
    out_buf: string; out_offset: number;
    level: number; in_level: number;
    weights: TensorRef; bias: TensorRef;
}

export interface UNetMeta {
    name: string;
    source: string;
    chans: number[];
    mid: number;
    scale: number;
    banks: number;
    residual: "catmull_rom";
    default_alpha: number[];
    gate: {
        conv1_w: TensorRef; conv1_b: TensorRef;
        conv2_w: TensorRef; conv2_b: TensorRef;
        fc_w: TensorRef; fc_b: TensorRef;
        feat: number; thumb: number; temperature: number; slope: number;
    };
    buffers: Record<string, { channels: number; vec4s: number; level: number }>;
    layers: UNetLayerSpec[];
}

export interface UNetWeights {
    format: "unet";
    meta: UNetMeta;
    tensor(ref: TensorRef): Float32Array;
}

const MAGIC_UNT = 0x00544E55;  // "UNT\0"

export function isUNetWeights(w: any): w is UNetWeights {
    return !!w && w.format === "unet";
}

export function parseUNetWeights(buffer: ArrayBuffer): UNetWeights {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    if (magic !== MAGIC_UNT) {
        throw new Error(`parseUNetWeights: unknown magic 0x${magic.toString(16)} (expected UNT\\0)`);
    }
    const version = view.getUint32(4, true);
    if (version !== 1) throw new Error(`parseUNetWeights: unsupported version ${version}`);
    const jsonLen = view.getUint32(8, true);
    const meta: UNetMeta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 12, jsonLen)));
    // 12 + jsonLen is a multiple of 4 (the exporter pads), so this view is aligned.
    const payload = new Float32Array(buffer, 12 + jsonLen);
    return {
        format: "unet",
        meta,
        tensor: (ref: TensorRef) => payload.subarray(ref.offset, ref.offset + ref.count),
    };
}
