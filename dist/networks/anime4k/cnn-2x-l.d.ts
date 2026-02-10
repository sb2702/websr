import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import { MediaSource } from "../../utils";
export default class Anime4KCNN2XL extends NeuralNetwork {
    constructor(weights: any);
    model(): Layer[];
    feedForward(source?: MediaSource): Promise<void>;
}
