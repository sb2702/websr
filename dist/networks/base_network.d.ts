import WebGPUContext from "../context";
import Layer from "../layers/base_layer";
declare class NeuralNetwork {
    context: WebGPUContext;
    layers: Layer[];
    weights: any;
    constructor(weights?: any);
    model(): Layer[];
    lastLayer(): Layer;
    protected bindInput(input: GPUTexture | GPUExternalTexture, output: GPUTexture, slots: [number, number][]): void;
    feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture): Promise<void>;
}
export default NeuralNetwork;
