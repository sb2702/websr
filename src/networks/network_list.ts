import Anime4KCNN2XS from "./anime4k/cnn-2x-s";
import Anime4KCNN2XM from "./anime4k/cnn-2x-m";
import Anime4KCNN2XL from "./anime4k/cnn-2x-l";
import Anime4KCNN2X16 from "./anime4k/cnn-2x-16";
import Anime4KCNN2X28 from "./anime4k/cnn-2x-28";
import Anime4KCNNRL from "./anime4k/cnn-restore-l";
import Anime4KCNNRM from "./anime4k/cnn-restore-m";
import Anime4KCNNRS from "./anime4k/cnn-restore-s";
import RealESRGANx4v3 from "./realesrgan/x4v3";
import SR2XSCNN from "./sr2-xs/cnn-2x-xs";
import UNetCNN from "./unet/unet";
import PoCNetwork from "./poc_network";
import NeuralNetwork from "./base_network";



type NeuralNetworkConstructor = new(...args: any[]) => NeuralNetwork


// ─── Network names ───────────────────────────────────────────────────────────
// Every name accepted by `network_name` / switchNetwork(), mapped to the architecture it
// runs. Weights are passed separately, so several names can share one architecture.
// To rename a model, edit its key here — nothing else refers to public names.
export const NetworkNames = {
    // WebSR models
    "websr/cnn-xs":     "sr2/cnn-2x-xs",       // depthwise-separable 2x, ~1.4k params
    "websr/cnn-s":      "anime4k/cnn-2x-l",    // Anime4K CNN-2x-L architecture, WebSR-trained weights
    "websr/cnn-m":      "anime4k/cnn-2x-16",   // 16-channel variant
    "websr/cnn-l":      "anime4k/cnn-2x-28",   // 28-channel variant
    "websr/gan-s":      "unet/u24",            // UNet distilled from Real-ESRGAN x4v3
    "websr/gan-m":      "unet/u32",            // UNet distilled from Real-ESRGAN x4v3

    // Third-party models
    "realesrgan/x4v3":  "realesrgan/x4v3",     // Real-ESRGAN realesr-general-x4v3, 4x

    // Original names, kept as aliases / for Anime4K weights
    "sr2/cnn-2x-xs":         "sr2/cnn-2x-xs",
    "anime4k/cnn-2x-s":      "anime4k/cnn-2x-s",
    "anime4k/cnn-2x-m":      "anime4k/cnn-2x-m",
    "anime4k/cnn-2x-l":      "anime4k/cnn-2x-l",
    "anime4k/cnn-2x-16":     "anime4k/cnn-2x-16",
    "anime4k/cnn-2x-28":     "anime4k/cnn-2x-28",
    "anime4k/cnn-restore-s": "anime4k/cnn-restore-s",
    "anime4k/cnn-restore-m": "anime4k/cnn-restore-m",
    "anime4k/cnn-restore-l": "anime4k/cnn-restore-l",
    "sb2702/blur-poc":       "sb2702/blur-poc",
} as const;

export type NetworkName = keyof typeof NetworkNames;

// Internal architecture ids (the values above). Not accepted as network names themselves.
export type NetworkArchitecture = typeof NetworkNames[NetworkName];

export function networkArchitecture(name: NetworkName): NetworkArchitecture {
    const arch = Object.prototype.hasOwnProperty.call(NetworkNames, name) ? NetworkNames[name] : undefined;
    if (!arch) throw Error(`Network ${name} is not defined or implemented`);
    return arch;
}

export type DisplayScale = 1 | 2 | 4;

export const NetworkList: Record<NetworkArchitecture, NeuralNetworkConstructor> = {
    "anime4k/cnn-2x-s": Anime4KCNN2XS,
    "anime4k/cnn-2x-m": Anime4KCNN2XM,
    "anime4k/cnn-2x-l": Anime4KCNN2XL,
    "anime4k/cnn-2x-16": Anime4KCNN2X16,
    "anime4k/cnn-2x-28": Anime4KCNN2X28,
    "anime4k/cnn-restore-s": Anime4KCNNRS,
    "anime4k/cnn-restore-m": Anime4KCNNRM,
    "anime4k/cnn-restore-l": Anime4KCNNRL,
    "realesrgan/x4v3":   RealESRGANx4v3,
    "sr2/cnn-2x-xs":     SR2XSCNN,
    // Conditional CReLU UNets (weights: WebSR.parseUNetWeights). The graph is in the weight
    // file, so both share one class. Per-video routing via WebSR.calibrate().
    "unet/u24":          UNetCNN,
    "unet/u32":          UNetCNN,
    "sb2702/blur-poc": PoCNetwork
}

export const NetworkScales: Record<NetworkArchitecture, DisplayScale> = {
    "anime4k/cnn-2x-s": 2,
    "anime4k/cnn-2x-m": 2,
    "anime4k/cnn-2x-l": 2,
    "anime4k/cnn-2x-16": 2,
    "anime4k/cnn-2x-28": 2,
    "anime4k/cnn-restore-s": 1,
    "anime4k/cnn-restore-m": 1,
    "anime4k/cnn-restore-l": 1,
    "realesrgan/x4v3":  4,
    "sr2/cnn-2x-xs":    2,
    "unet/u24":         2,
    "unet/u32":         2,
    "sb2702/blur-poc": 1
}

// ─── Tile sizing ─────────────────────────────────────────────────────────────
// Large sources are split into tiles so a network's buffers fit the GPU memory budget
// (see WebSR.computeMaxTilePixels, which uses the device tier).

// Count of simultaneously-held GPU buffers per network (activation buffers, with a little
// headroom for the input / output textures).
// sr2/cnn-2x-xs: stem + (dw + pw) × num_blocks (+ optional projection); sized for 2 blocks.
export const NetworkBufferCounts: Record<NetworkArchitecture, number> = {
    "anime4k/cnn-2x-s": 6,
    "anime4k/cnn-2x-m": 12,
    "anime4k/cnn-2x-l": 23,
    "anime4k/cnn-2x-16": 43,
    "anime4k/cnn-2x-28": 73,
    "anime4k/cnn-restore-s": 6,
    "anime4k/cnn-restore-m": 10,
    "anime4k/cnn-restore-l": 11,
    "realesrgan/x4v3":  8,
    "sr2/cnn-2x-xs":    5,
    // UNets: buffers differ in size per level, so this is 1 and NetworkBytesPerPixel holds the
    // TOTAL bytes per full-res pixel (f32 worst case; f16 halves it).
    "unet/u24":         1,
    "unet/u32":         1,
    "sb2702/blur-poc": 4,
}

// Bytes per pixel of the largest single buffer.
// anime4k: every buffer is 4 channels × 4 bytes/float = 16.
// realesrgan/x4v3: the 64-channel ping-pong buffers.
// sr2/cnn-2x-xs: every buffer is 16 channels × 4 bytes/float = 64.
export const NetworkBytesPerPixel: Record<NetworkArchitecture, number> = {
    "anime4k/cnn-2x-s": 16,
    "anime4k/cnn-2x-m": 16,
    "anime4k/cnn-2x-l": 16,
    "anime4k/cnn-2x-16": 16,
    "anime4k/cnn-2x-28": 16,
    "anime4k/cnn-restore-s": 16,
    "anime4k/cnn-restore-m": 16,
    "anime4k/cnn-restore-l": 16,
    "realesrgan/x4v3":  512,
    "sr2/cnn-2x-xs":    64,
    // in 16 + cat0 (2*c0) + f0 (c0) + out (12) at full res, cat1 + f1 at 1/4 the pixels,
    // the two bottleneck buffers at 1/16 — all x4 bytes.
    //   u24 (24-48-96):  16 + 192 + 96 + 48 + (384+192)/4 + 2*384/16 = 544
    //   u32 (32-64-128): 16 + 256 + 128 + 48 + (512+256)/4 + 2*512/16 = 704
    "unet/u24":         544,
    "unet/u32":         704,
    "sb2702/blur-poc": 16,
}

// Largest single buffer per full-res pixel (f32). When set, tile size also respects the
// device's maxStorageBufferBindingSize (the UNet skip buffer cat0 is the big one).
export const NetworkMaxBufferBytesPerPixel: Partial<Record<NetworkArchitecture, number>> = {
    "unet/u24":  192,
    "unet/u32":  256,
}

// Tile origin/size alignment. A UNet with 2 stride-2 levels groups pixels in 4x4 blocks at its
// coarsest level; tiles must start on that grid (and padding must be a multiple of it) for a
// tile to compute the same function as the whole frame.
export const NetworkTileAlign: Partial<Record<NetworkArchitecture, number>> = {
    "unet/u24":  4,
    "unet/u32":  4,
}

// Tile padding (per side) needed to hide tile-seam contamination.
// Each 3x3 conv contaminates 1 pixel of padding per side, so the safe value
// is roughly the conv layer count. anime4k networks are shallow (~7 layers);
// realesrgan/x4v3 has 34 conv layers; sr2/cnn-2x-xs has only 2 spatial convs.
export const NetworkTilePadding: Record<NetworkArchitecture, number> = {
    "anime4k/cnn-2x-s": 6,
    "anime4k/cnn-2x-m": 12,
    "anime4k/cnn-2x-l": 12,
    "anime4k/cnn-2x-16": 12,
    "anime4k/cnn-2x-28": 12,
    "anime4k/cnn-restore-s": 6,
    "anime4k/cnn-restore-m": 12,
    "anime4k/cnn-restore-l": 12,
    "realesrgan/x4v3":  34,
    "sr2/cnn-2x-xs":    4,
    // ~half the receptive field (RF 71 LR px), rounded up to the tile alignment
    "unet/u24":         36,
    "unet/u32":         36,
    "sb2702/blur-poc": 2,
}
