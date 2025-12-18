import WebGPUContext from "../context";
import Layer from "../layers/base_layer";
import { MediaSource } from "../utils";
declare class NeuralNetwork {
    context: WebGPUContext;
    layers: Layer[];
    weights: any;
    constructor(weights?: any);
    model(): Layer[];
    lastLayer(): Layer;
    feedForward(source?: MediaSource): Promise<void>;
}
export default NeuralNetwork;
