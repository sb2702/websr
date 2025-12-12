import { Resolution } from "./utils";
import { DisplayScale } from "./networks/network_list";
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
    scale: DisplayScale;
    constructor(device: GPUDevice, resolution: Resolution, canvas: HTMLCanvasElement, scale: DisplayScale, debug?: boolean);
    readBuffer(bufferName: string): Promise<Uint8ClampedArray | Float32Array>;
    readTexture(textureName: string): Promise<Uint8ClampedArray | Float32Array>;
    destroy(): void;
    buffer(key: string, options?: BufferOptions): GPUBuffer;
    texture(key: string, options?: TextureOptions): GPUTexture;
}
export {};
