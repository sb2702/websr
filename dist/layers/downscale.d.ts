export default class DownscaleLayer {
    private device;
    private context;
    private pipeline;
    private sampler;
    private bindGroup;
    private renderPassDescriptor;
    constructor(input: GPUTexture, output: GPUTexture);
    setOutput(output: GPUTexture): void;
    run(): void;
}
