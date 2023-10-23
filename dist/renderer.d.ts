import NeuralNetwork from "./networks/base_network";
export default class WebSRRenderer {
    private context;
    private network;
    source: HTMLVideoElement | HTMLImageElement;
    active: boolean;
    vfc: number;
    constructor(network: NeuralNetwork, source: HTMLVideoElement | HTMLImageElement);
    start(): Promise<void>;
    stop(): Promise<void>;
    renderStep(): Promise<void>;
    render(): Promise<void>;
}
