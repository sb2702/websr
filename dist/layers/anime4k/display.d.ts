/// <reference types="@webgpu/types" />
import RenderLayer from "../base_render_layer";
declare class DisplayLayer extends RenderLayer {
    label: string;
    constructor(inputs: (GPUTexture | GPUExternalTexture | GPUBuffer)[], output: GPUTexture);
    lazyLoadSetup(): void;
    defaultBindGroup(): GPUBindGroup;
}
export default DisplayLayer;
