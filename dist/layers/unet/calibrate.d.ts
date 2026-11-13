import { UNetWeights } from "../../weights";
export declare const THUMB = 32;
export declare const MAX_THUMBNAILS = 64;
export declare function thumbShader(external: boolean): string;
export declare function computeThumbnails(device: GPUDevice, frames: (VideoFrame | ImageBitmap)[]): Promise<Float32Array>;
export declare function runGate(device: GPUDevice, w: UNetWeights, thumbs: Float32Array): Promise<Float32Array>;
export declare function foldShader(f16: boolean): string;
