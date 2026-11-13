export interface Anime4KWeights {
    name: string;
    layers: Record<string, {
        weights: Float32Array;
        bias: Float32Array;
    }>;
}
export declare function parseAnime4KWeights(buffer: ArrayBuffer): Anime4KWeights;
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
export declare function parseBinaryWeights(buffer: ArrayBuffer): BinaryWeights | Anime4KWeights | UNetWeights;
export declare function parseX4VWeights(buffer: ArrayBuffer): BinaryWeights;
export interface TensorRef {
    offset: number;
    count: number;
}
export interface UNetLayerSpec {
    name: string;
    mode: "same" | "s2" | "up";
    cin: number;
    cout: number;
    in_vec4s: number;
    out_vec4s: number;
    crelu: boolean;
    banked: boolean;
    in_buf: string;
    in_offset: number;
    out_buf: string;
    out_offset: number;
    level: number;
    in_level: number;
    weights: TensorRef;
    bias: TensorRef;
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
        conv1_w: TensorRef;
        conv1_b: TensorRef;
        conv2_w: TensorRef;
        conv2_b: TensorRef;
        fc_w: TensorRef;
        fc_b: TensorRef;
        feat: number;
        thumb: number;
        temperature: number;
        slope: number;
    };
    buffers: Record<string, {
        channels: number;
        vec4s: number;
        level: number;
    }>;
    layers: UNetLayerSpec[];
}
export interface UNetWeights {
    format: "unet";
    meta: UNetMeta;
    tensor(ref: TensorRef): Float32Array;
}
export declare function isUNetWeights(w: any): w is UNetWeights;
export declare function parseUNetWeights(buffer: ArrayBuffer): UNetWeights;
