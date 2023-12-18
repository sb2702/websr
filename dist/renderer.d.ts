import NeuralNetwork from "./networks/base_network";
export default class WebSRRenderer {
    private context;
    private network;
    source?: HTMLVideoElement | HTMLImageElement | ImageBitmap;
    active: boolean;
    vfc: number;
    constructor(network: NeuralNetwork, source?: HTMLVideoElement | HTMLImageElement | ImageBitmap);
    start(): Promise<void>;
    stop(): Promise<void>;
    renderStep(): Promise<void>;
    render(source?: ImageBitmap): Promise<void>;
}
