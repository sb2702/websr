import WebGPUContext from './context'
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import {NetworkList, NetworkName} from "./networks/network_list";
import {Resolution} from "./utils";


interface WebSRParams {
    video: HTMLVideoElement,
    canvas?: HTMLCanvasElement,
    weights: any,
    network_name: NetworkName,
    gpu: GPUDevice
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


    constructor(params: WebSRParams) {

        this.video = params.video;

        // We should make this dynamic
        this.resolution = {
            width: this.video.videoWidth,
            height: this.video.videoHeight
        };

        if(params.canvas) this.canvas = params.canvas;
        else  {
            this.canvas = new HTMLCanvasElement();
            this.canvas.width = this.resolution.width*2;
            this.canvas.height = this.resolution.height*2;

        }

        this.context = new WebGPUContext(params.gpu, this.resolution,  this.canvas);

        globalThis.context = this.context;

        if(!NetworkList[params.network_name]) throw Error(`Network ${params.network_name} is not defined or implemented`);

        this.network = new NetworkList[params.network_name](params.weights);

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


    async start (){
        await this.renderer.start();
    }

    async stop(){
        await this.renderer.stop();
    }

    async destroy(){
        await this.renderer.stop();
        this.context.destroy();
    }



}
