import WebGPUContext from './context'
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import {NetworkList, NetworkName} from "./networks/network_list";
import {Resolution} from "./utils";


interface WebSRParams {
    source: HTMLVideoElement | HTMLImageElement,
    canvas?: HTMLCanvasElement,
    weights: any,
    debug?: boolean;
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
    debug?: boolean;
    source: HTMLVideoElement | HTMLImageElement


    constructor(params: WebSRParams) {

        this.source = params.source;

        const source = this.source;

        // We should make this dynamic
        this.resolution = {
            width: (source instanceof HTMLVideoElement) ? source.videoWidth : source.naturalWidth,
            height: (source instanceof HTMLVideoElement) ? source.videoHeight : source.naturalHeight,
        };

        if(params.canvas) this.canvas = params.canvas;
        else  {
            this.canvas = new HTMLCanvasElement();
            this.canvas.width = this.resolution.width*2;
            this.canvas.height = this.resolution.height*2;

        }

        console.log("Context");
        console.log(params.gpu.features.has('shader-f16'))

        this.context = new WebGPUContext(params.gpu, this.resolution,  this.canvas, this.debug);

        globalThis.context = this.context;

        if(!NetworkList[params.network_name]) throw Error(`Network ${params.network_name} is not defined or implemented`);

        this.network = new NetworkList[params.network_name](params.weights);

        this.renderer = new WebSRRenderer(this.network, this.source);

    }

    static async initWebGPU(): Promise<GPUDevice | false>{

        if(!navigator.gpu) return false;

        const adapter = await navigator.gpu.requestAdapter();

        if(!adapter) return false;


        let device;

        if(adapter.features.has("shader-f16")){
            device = await adapter.requestDevice({
                requiredFeatures:  ['shader-f16']
            });
        } else {
            device = await adapter.requestDevice();
        }


        if(!device) return false;

        return device;
    }


    async start (){
        await this.renderer.start();
    }

    async stop(){
        await this.renderer.stop();
    }

    async render(source?: ImageBitmap){
        await this.renderer.render(source);
    }

    async destroy(){
        await this.renderer.stop();
        this.context.destroy();
    }



}
