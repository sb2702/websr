import WebGPUContext from "../context";
import Layer from "../layers/base_layer";
declare class NeuralNetwork {
    context: WebGPUContext;
    layers: Layer[];
    weights: any;
    constructor(weights?: any);
    model(): Layer[];
    lastLayer(): Layer;
    feedForward(source?: HTMLVideoElement | HTMLImageElement | ImageBitmap): Promise<void>;
}
export default NeuralNetwork;
