import RenderLayer from "../base_render_layer";
declare class RGB2YUV extends RenderLayer {
    label: string;
    private _builtForExternal;
    constructor(inputTextures: (GPUTexture | GPUExternalTexture)[], outputTexture: GPUTexture);
    lazyLoadSetup(): void;
}
export default RGB2YUV;
