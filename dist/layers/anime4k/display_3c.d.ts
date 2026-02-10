import RenderLayer from "../base_render_layer";
declare class DisplayLayer3C extends RenderLayer {
    label: string;
    constructor(inputs: (GPUTexture | GPUExternalTexture | GPUBuffer)[], output: GPUTexture);
    lazyLoadSetup(): void;
    defaultPipelineConfig(): GPURenderPipelineDescriptor;
    defaultBindGroup(): GPUBindGroup;
}
export default DisplayLayer3C;
