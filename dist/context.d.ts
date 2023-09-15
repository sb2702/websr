/// <reference types="@webgpu/types" />
import { Resolution } from "./utils";
interface TextureOptions {
    width?: number;
    height?: number;
    format?: GPUTextureFormat;
}
export default class WebGPUContext {
    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    textures: Record<string, GPUTexture>;
    resolution: Resolution;
    input: GPUExternalTexture;
    destroyed: boolean;
    debug: boolean;
    usage: number;
    constructor(device: GPUDevice, resolution: Resolution, canvas: HTMLCanvasElement);
    readTexture(textureName: string): Promise<Uint8ClampedArray | Float32Array>;
    destroy(): void;
    texture(key: string, options?: TextureOptions): GPUTexture;
}
export {};
