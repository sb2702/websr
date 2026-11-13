import RenderLayer from "../base_render_layer";
declare class SR2XSDisplay extends RenderLayer {
    label: string;
    constructor(inputs: (GPUTexture | GPUExternalTexture | GPUBuffer)[], output: GPUTexture);
    lazyLoadSetup(): void;
    defaultBindGroup(): GPUBindGroup;
}
export default SR2XSDisplay;
