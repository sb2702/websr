/// <reference types="@webgpu/types" />
import Layer from "../base_layer";
declare class Anime4KConv8x4 extends Layer {
    label: string;
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture, weights: any);
}
export default Anime4KConv8x4;
