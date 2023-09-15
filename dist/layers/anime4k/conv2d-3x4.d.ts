/// <reference types="@webgpu/types" />
import Layer from "../base_layer";
declare class Anime4KConv3x4 extends Layer {
    label: string;
    constructor(inputTextures: GPUExternalTexture[], outputTexture: GPUTexture, weights: any);
    run(): void;
}
export default Anime4KConv3x4;
