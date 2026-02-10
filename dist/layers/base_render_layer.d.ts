import { Resolution } from "../utils";
import Layer from "./base_layer";
declare class RenderLayer extends Layer {
    vertexScale: Resolution;
    output: GPUTexture;
    pipeline: GPURenderPipeline;
    constructor(inputs: (GPUTexture | GPUExternalTexture | GPUBuffer)[], output: GPUTexture, weights?: any);
    defaultVertexShader(): string;
    defaultPipelineConfig(): GPURenderPipelineDescriptor;
    defaultSetup(): void;
    defaultRenderPassDescriptor(): GPURenderPassDescriptor;
    createStandardShader(fragmentShader: string): GPUShaderModule;
    fragmentShaderInputs(): string;
    run(): void;
    setOutput(outputTexture: GPUTexture): void;
}
export default RenderLayer;
