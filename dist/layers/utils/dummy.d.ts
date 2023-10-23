/// <reference types="@webgpu/types" />
import RenderLayer from "../base_render_layer";
declare class DummyLayer extends RenderLayer {
    label: string;
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture);
}
export default DummyLayer;
