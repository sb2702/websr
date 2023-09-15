/// <reference types="@webgpu/types" />
import Layer from "../base_layer";
declare class DisplayLayer extends Layer {
    label: string;
    constructor(inputTextures: (GPUTexture | GPUExternalTexture)[], outputTexture: GPUTexture);
    defaultBindGroup(): GPUBindGroup;
    setOutput(outputTexture: GPUTexture): void;
}
export default DisplayLayer;
