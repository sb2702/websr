/// <reference types="@webgpu/types" />
import Layer from "../base_layer";
declare class GuassianLayer extends Layer {
    label: string;
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture);
}
export default GuassianLayer;
