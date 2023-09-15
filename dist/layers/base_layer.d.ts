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
    pipeline: GPURenderPipeline;
    sampler: GPUSampler;
    renderPassDescriptor: GPURenderPassDescriptor;
    bindGroup: GPUBindGroup | null;
    label: string;
    inputTextures: (GPUTexture | GPUExternalTexture)[];
    outputTexture: GPUTexture;
    uniforms: Uniform[];
    weights: any;
    buffers: Record<string, GPUBuffer>;
    context: WebGPUContext;
    resolution: Resolution;
    vertexScale: Resolution;
    constructor(inputTextures: (GPUTexture | GPUExternalTexture)[], outputTexture: GPUTexture, weights?: any);
    defaultVertexShader(): string;
    createUniform(name: string, type: string): void;
    defaultPipelineConfig(): GPURenderPipelineDescriptor;
    defaultSetup(): void;
    defaultRenderPassDescriptor(): GPURenderPassDescriptor;
    createStandardShader(fragmentShader: string): GPUShaderModule;
    fragmentShaderInputs(): string;
    setUniform(name: string, value: Float32Array): void;
    defaultBindGroup(): GPUBindGroup;
    hasExternalTexture(): boolean;
    run(): void;
}
export default Layer;
