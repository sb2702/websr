/// <reference types="@webgpu/types" />
import WebGPUContext from './context';
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import { NetworkName } from "./networks/network_list";
import { Resolution } from "./utils";
interface WebSRParams {
    video: HTMLVideoElement;
    canvas?: HTMLCanvasElement;
    weights: any;
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
    video: HTMLVideoElement;
    constructor(params: WebSRParams);
    static initWebGPU(): Promise<GPUDevice | false>;
    start(): Promise<void>;
    stop(): Promise<void>;
    destroy(): Promise<void>;
}
export {};
