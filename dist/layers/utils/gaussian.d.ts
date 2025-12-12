import RenderLayer from "../base_render_layer";
declare class GuassianLayer extends RenderLayer {
    label: string;
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture);
}
export default GuassianLayer;
