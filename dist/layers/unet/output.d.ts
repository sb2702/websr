import RenderLayer from "../base_render_layer";
declare class UNetOutput extends RenderLayer {
    label: string;
    private useF16;
    private _builtForExternal;
    constructor(inputs: (GPUBuffer | GPUTexture | GPUExternalTexture)[], output: GPUTexture, useF16?: boolean);
    lazyLoadSetup(): void;
    defaultBindGroup(): GPUBindGroup;
}
export default UNetOutput;
