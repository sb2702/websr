/// <reference types="@webgpu/types" />
import WebGPUContext from './context';
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import { NetworkName } from "./networks/network_list";
import { Resolution } from "./utils";
interface WebSRParams {
    source: HTMLVideoElement | HTMLImageElement;
    canvas?: HTMLCanvasElement;
    weights: any;
    debug?: boolean;
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
    source: HTMLVideoElement | HTMLImageElement;
    constructor(params: WebSRParams);
    static initWebGPU(): Promise<GPUDevice | false>;
    start(): Promise<void>;
    stop(): Promise<void>;
    render(source?: ImageBitmap): Promise<void>;
    destroy(): Promise<void>;
}
export {};
