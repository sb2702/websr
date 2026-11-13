import RenderLayer from "../base_render_layer";
declare class RealESRGANOutputX4 extends RenderLayer {
    label: string;
    private useF16;
    constructor(inputs: (GPUBuffer | GPUTexture | GPUExternalTexture)[], output: GPUTexture, useF16?: boolean);
    private _builtForExternal;
    lazyLoadSetup(): void;
    defaultBindGroup(): GPUBindGroup;
}
export default RealESRGANOutputX4;
