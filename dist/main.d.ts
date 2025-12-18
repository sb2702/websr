import WebGPUContext from './context';
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import { NetworkName, DisplayScale } from "./networks/network_list";
import { Resolution, MediaSource } from "./utils";
interface WebSRParams {
    source?: MediaSource;
    canvas?: HTMLCanvasElement;
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
    context: WebGPUContext;
    network: NeuralNetwork;
    renderer: WebSRRenderer;
    resolution: Resolution;
    debug?: boolean;
    source: MediaSource;
    scale: DisplayScale;
    constructor(params: WebSRParams);
    switchNetwork(network: NetworkName, weights: any): void;
    static initWebGPU(): Promise<GPUDevice | false>;
    start(): Promise<void>;
    stop(): Promise<void>;
    render(source?: ImageBitmap): Promise<void>;
    destroy(): Promise<void>;
}
export {};
