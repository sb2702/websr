/// <reference types="@webgpu/types" />
import { Resolution } from "./utils";
interface TextureOptions {
    width?: number;
    height?: number;
    format?: GPUTextureFormat;
}
interface BufferOptions {
    channels?: number;
    bitdepth?: number;
    width?: number;
    height?: number;
}
export default class WebGPUContext {
    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    textures: Record<string, GPUTexture>;
    buffers: Record<string, GPUBuffer>;
    resolution: Resolution;
    input: GPUTexture | GPUExternalTexture;
    destroyed: boolean;
    debug?: boolean;
    textureUsage: number;
    bufferUsage: number;
    constructor(device: GPUDevice, resolution: Resolution, canvas: HTMLCanvasElement, debug?: boolean);
    readBuffer(bufferName: string): Promise<Uint8ClampedArray | Float32Array>;
    readTexture(textureName: string): Promise<Uint8ClampedArray | Float32Array>;
    destroy(): void;
    buffer(key: string, options?: BufferOptions): GPUBuffer;
    texture(key: string, options?: TextureOptions): GPUTexture;
}
export {};
