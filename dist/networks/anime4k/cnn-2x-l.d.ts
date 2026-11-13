import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
export default class Anime4KCNN2XL extends NeuralNetwork {
    constructor(weights: any);
    model(): Layer[];
    feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture): Promise<void>;
}
