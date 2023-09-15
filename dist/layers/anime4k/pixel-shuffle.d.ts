/// <reference types="@webgpu/types" />
import Layer from "../base_layer";
declare class PixelShuffle2X extends Layer {
    label: string;
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture);
}
export default PixelShuffle2X;
