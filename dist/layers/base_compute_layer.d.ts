import Layer from "./base_layer";
declare class ComputeLayer extends Layer {
    pipeline: GPUComputePipeline;
    num_work_groups: number;
    constructor(inputTextures: (GPUTexture | GPUExternalTexture | GPUBuffer)[], outputBuffer: GPUBuffer, weights?: any);
    createStandardShader(computeShader: string): GPUShaderModule;
    computeShaderInputs(): string;
    defaultPipelineConfig(): GPUComputePipelineDescriptor;
    defaultSetup(): void;
    lazyLoadSetup(): void;
    run(): void;
}
export default ComputeLayer;
