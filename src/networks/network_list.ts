import Anime4KCNN2XS from "./anime4k/cnn-2x-s";
import Anime4KCNN2XM from "./anime4k/cnn-2x-m";
import Anime4KCNN2XL from "./anime4k/cnn-2x-l";
import Anime4KCNNRL from "./anime4k/cnn-restore-l";
import Anime4KCNNRM from "./anime4k/cnn-restore-m";
import Anime4KCNNRS from "./anime4k/cnn-restore-s";
import PoCNetwork from "./poc_network";
import NeuralNetwork from "./base_network";



type NeuralNetworkConstructor = new(...args: any[]) => NeuralNetwork


export type NetworkName = "anime4k/cnn-2x-l"|  "anime4k/cnn-2x-m"|  "anime4k/cnn-2x-s"|'sb2702/blur-poc' | "anime4k/cnn-restore-s" | "anime4k/cnn-restore-m" | "anime4k/cnn-restore-l";

export type DisplayScale = 1 | 2 | 4;

export const NetworkList: Record<NetworkName, NeuralNetworkConstructor> = {
    "anime4k/cnn-2x-s": Anime4KCNN2XS,
    "anime4k/cnn-2x-m": Anime4KCNN2XM,
    "anime4k/cnn-2x-l": Anime4KCNN2XL,
    "anime4k/cnn-restore-s": Anime4KCNNRS,
    "anime4k/cnn-restore-m": Anime4KCNNRM,
    "anime4k/cnn-restore-l": Anime4KCNNRL,
    "sb2702/blur-poc": PoCNetwork
}

export const NetworkScales: Record<NetworkName, DisplayScale> = {
    "anime4k/cnn-2x-s": 2,
    "anime4k/cnn-2x-m": 2,
    "anime4k/cnn-2x-l": 2,
    "anime4k/cnn-restore-s": 1,
    "anime4k/cnn-restore-m": 1,
    "anime4k/cnn-restore-l": 1,
    "sb2702/blur-poc": 1
}

// ─── Tile sizing ─────────────────────────────────────────────────────────────
// Large sources are split into tiles so a network's buffers fit the GPU memory budget
// (see WebSR.computeMaxTilePixels, which uses the device tier).

// Count of simultaneously-held GPU buffers per network (activation buffers, with a little
// headroom for the input / output textures).
export const NetworkBufferCounts: Record<NetworkName, number> = {
    "anime4k/cnn-2x-s": 6,
    "anime4k/cnn-2x-m": 12,
    "anime4k/cnn-2x-l": 23,
    "anime4k/cnn-restore-s": 6,
    "anime4k/cnn-restore-m": 10,
    "anime4k/cnn-restore-l": 11,
    "sb2702/blur-poc": 4,
}

// Bytes per pixel of the largest single buffer. anime4k: 4 channels × 4 bytes/float = 16.
export const NetworkBytesPerPixel: Record<NetworkName, number> = {
    "anime4k/cnn-2x-s": 16,
    "anime4k/cnn-2x-m": 16,
    "anime4k/cnn-2x-l": 16,
    "anime4k/cnn-restore-s": 16,
    "anime4k/cnn-restore-m": 16,
    "anime4k/cnn-restore-l": 16,
    "sb2702/blur-poc": 16,
}

// Largest single buffer per full-res pixel (f32). When set, tile size also respects the
// device's maxStorageBufferBindingSize.
export const NetworkMaxBufferBytesPerPixel: Partial<Record<NetworkName, number>> = {
}

// Tile origin/size alignment, for networks whose layers downsample (tiles must start on
// the coarsest grid to compute the same function as the whole frame).
export const NetworkTileAlign: Partial<Record<NetworkName, number>> = {
}

// Tile padding (per side) needed to hide tile-seam contamination. Each 3x3 conv
// contaminates 1 pixel of padding per side, so the safe value is roughly the conv
// layer count.
export const NetworkTilePadding: Record<NetworkName, number> = {
    "anime4k/cnn-2x-s": 6,
    "anime4k/cnn-2x-m": 12,
    "anime4k/cnn-2x-l": 12,
    "anime4k/cnn-restore-s": 6,
    "anime4k/cnn-restore-m": 12,
    "anime4k/cnn-restore-l": 12,
    "sb2702/blur-poc": 2,
}
