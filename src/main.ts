
import WebGPUContext from './context'
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import {NetworkList, NetworkName} from "./networks/network_list";

export interface Resolution {
    width: number,
    height: number
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
    video: HTMLVideoElement


    constructor(video: HTMLVideoElement, network_name: NetworkName, weights: any, device: GPUDevice, canvas: HTMLCanvasElement) {

        this.canvas = canvas;
        this.video = video;

        // We should make this dynamic
        this.resolution = {
            width: video.videoWidth,
            height: video.videoHeight
        };

        this.context = new WebGPUContext(device, this.resolution,  canvas);

        globalThis.context = this.context;

        if(!NetworkList[network_name]) throw Error(`Network ${network_name} is not defined or implemented`);

        this.network = new NetworkList[network_name](weights);

        this.renderer = new WebSRRenderer(this.network, this.video);

    }

    static async initWebGPU(): Promise<GPUDevice | false>{

        if(!navigator.gpu) return false;

        const adapter = await navigator.gpu.requestAdapter();

        if(!adapter) return false;

        const device = await adapter.requestDevice();

        if(!device) return false;

        return device;
    }


    start (){
        this.renderer.start();
    }
    


}
