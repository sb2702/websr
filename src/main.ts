import WebGPUContext from './context'
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import {NetworkList, NetworkName, NetworkScales, DisplayScale} from "./networks/network_list";
import {Resolution} from "./utils";


interface WebSRParams {
    source?: HTMLVideoElement | HTMLImageElement | ImageBitmap,
    canvas?: HTMLCanvasElement,
    weights: any,
    debug?: boolean;
    resolution?: Resolution,
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
    source: HTMLVideoElement | HTMLImageElement | ImageBitmap
    scale: DisplayScale;


    constructor(params: WebSRParams) {



        if(!NetworkList[params.network_name]) throw Error(`Network ${params.network_name} is not defined or implemented`);


        this.source = params.source;

        const source = this.source;

        this.resolution = params.resolution? params.resolution : {
            width: (source instanceof HTMLVideoElement) ? source.videoWidth :  (source instanceof HTMLImageElement) ?  source.naturalWidth : source.width,
            height: (source instanceof HTMLVideoElement) ? source.videoHeight : (source instanceof HTMLImageElement) ?  source.naturalHeight: source.height
        }

        const scale = NetworkScales[params.network_name];

        if(params.canvas) this.canvas = params.canvas;
        else  {
            this.canvas = new HTMLCanvasElement();
            this.canvas.width = this.resolution.width*scale;
            this.canvas.height = this.resolution.height*scale;

        }

        this.scale = scale;

        this.context = new WebGPUContext(params.gpu, this.resolution,  this.canvas, this.scale, this.debug);

        globalThis.context = this.context;

        this.network = new NetworkList[params.network_name](params.weights);

        this.renderer = new WebSRRenderer(this.network, this.source);

    }

    switchNetwork(network: NetworkName, weights: any){
        if(!NetworkList[network]) throw Error(`Network ${network} is not defined or implemented`);
        this.network = new NetworkList[network](weights);
        this.renderer.switchNetwork(this.network);

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

    async render(source?: ImageBitmap){
        await this.renderer.render(source);
    }

    async destroy(){
        await this.renderer.stop();
        this.context.destroy();
    }



}
