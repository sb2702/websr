/// <reference types="@webgpu/types" />
import Layer from "../base_layer";
declare class RGB2YUV extends Layer {
    label: "RGB2YUV";
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture);
}
export default RGB2YUV;
