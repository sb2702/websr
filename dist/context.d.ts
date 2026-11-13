import { Resolution } from "./utils";
import { DisplayScale } from "./networks/network_list";
export type Precision = 'auto' | 'f32' | 'f16';
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
    processResolution: Resolution;
    canvasResolution: Resolution;
    outputResolution: Resolution;
    input: GPUTexture | GPUExternalTexture;
    destroyed: boolean;
    debug?: boolean;
    textureUsage: number;
    bufferUsage: number;
    scale: DisplayScale;
    precision: Precision;
    get resolution(): Resolution;
    constructor(device: GPUDevice, processResolution: Resolution, canvasResolution: Resolution, canvas: HTMLCanvasElement, scale: DisplayScale, debug?: boolean, outputResolution?: Resolution, precision?: Precision);
    readBuffer(bufferName: string): Promise<Uint8ClampedArray | Float32Array>;
    readTexture(textureName: string): Promise<Uint8ClampedArray | Float32Array>;
    cleanup(): void;
    destroy(): void;
    useF16(): boolean;
    buffer(key: string, options?: BufferOptions): GPUBuffer;
    texture(key: string, options?: TextureOptions): GPUTexture;
}
export {};
