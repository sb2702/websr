import WebGPUContext from './context';
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import { NetworkName, DisplayScale } from "./networks/network_list";
import { Resolution, MediaSource } from "./utils";
interface WebSRParams {
    canvas: HTMLCanvasElement;
    weights: any;
    debug?: boolean;
    resolution?: Resolution;
    network_name: NetworkName;
    gpu: GPUDevice;
}
declare global {
    var context: WebGPUContext;
}
export default class WebSR {
    canvas: HTMLCanvasElement;
    context?: WebGPUContext;
    network?: NeuralNetwork;
    renderer?: WebSRRenderer;
    resolution?: Resolution;
    debug?: boolean;
    scale: DisplayScale;
    private params;
    private initialized;
    constructor(params: WebSRParams);
    private initialize;
    private updateResolution;
    switchNetwork(network: NetworkName, weights: any): void;
    static initWebGPU(): Promise<GPUDevice | false>;
    render(source: MediaSource): Promise<void>;
    destroy(): Promise<void>;
}
export {};
