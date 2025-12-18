import NeuralNetwork from "./networks/base_network";
import { MediaSource } from "./utils";
export default class WebSRRenderer {
    private context;
    private network;
    source?: MediaSource;
    active: boolean;
    vfc: number;
    constructor(network: NeuralNetwork, source?: MediaSource);
    switchNetwork(network: NeuralNetwork): void;
    start(): Promise<void>;
    stop(): Promise<void>;
    renderStep(): Promise<void>;
    render(source?: ImageBitmap): Promise<void>;
}
