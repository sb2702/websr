/// <reference types="@webgpu/types" />
import WebGPUContext from "../context";
import { Resolution } from "../utils";
interface Uniform {
    name: string;
    type: string;
}
declare class Layer {
    device: GPUDevice;
    shader: GPUShaderModule;
    pipeline: GPURenderPipeline | GPUComputePipeline;
    sampler: GPUSampler;
    renderPassDescriptor: GPURenderPassDescriptor;
    bindGroup: GPUBindGroup | null;
    label: string;
    inputs: (GPUTexture | GPUExternalTexture | GPUBuffer)[];
    output: GPUTexture | GPUBuffer;
    uniforms: Uniform[];
    weights: any;
    buffers: Record<string, GPUBuffer>;
    context: WebGPUContext;
    resolution: Resolution;
    constructor(inputs: (GPUTexture | GPUExternalTexture | GPUBuffer)[], output: GPUTexture | GPUBuffer, weights?: any);
    createUniform(name: string, type: string): void;
    setUniform(name: string, value: Float32Array): void;
    defaultBindGroup(): GPUBindGroup;
    hasExternalTexture(): boolean;
    lazyLoadSetup(): void;
    run(): void;
}
export default Layer;
