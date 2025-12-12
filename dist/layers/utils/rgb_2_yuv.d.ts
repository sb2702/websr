import RenderLayer from "../base_render_layer";
declare class RGB2YUV extends RenderLayer {
    label: "RGB2YUV";
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture);
}
export default RGB2YUV;
