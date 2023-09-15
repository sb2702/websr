/// <reference types="@webgpu/types" />
import Layer from "../base_layer";
declare class DummyLayer extends Layer {
    label: string;
    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture);
}
export default DummyLayer;
