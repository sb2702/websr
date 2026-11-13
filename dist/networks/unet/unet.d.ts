import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import { UNetWeights } from "../../weights";
import { Resolution } from "../../utils";
export declare function levelResolution(r: Resolution, level: number): Resolution;
export default class UNetCNN extends NeuralNetwork {
    private folds;
    private alphaBuf;
    private foldPipeline;
    weightsUNet: UNetWeights;
    alpha: Float32Array;
    constructor(weights: UNetWeights);
    model(): Layer[];
    private bankedTensor;
    get banks(): number;
    applyAlpha(alpha: Float32Array): void;
    calibrate(thumbs: Float32Array): Promise<Float32Array>;
    feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture): Promise<void>;
}
