import WebGPUContext from './context'
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import {NetworkList, NetworkName, NetworkScales, DisplayScale} from "./networks/network_list";
import {Resolution, MediaSource, getSourceWidth, getSourceHeight} from "./utils";


interface WebSRParams {
    canvas: HTMLCanvasElement,
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
    context?: WebGPUContext;
    network?: NeuralNetwork;
    renderer?: WebSRRenderer;
    resolution?: Resolution;
    debug?: boolean;
    scale: DisplayScale;
    private params: WebSRParams;
    private initialized: boolean = false;


    constructor(params: WebSRParams) {

        if(!NetworkList[params.network_name]) throw Error(`Network ${params.network_name} is not defined or implemented`);

        this.params = params;
        this.canvas = params.canvas;
        this.scale = NetworkScales[params.network_name];
        this.debug = params.debug;

        // If resolution is provided, initialize immediately
        if(params.resolution) {
            this.resolution = params.resolution;
            this.initialize();
        }
    }

    private initialize() {
        if(!this.resolution) {
            throw new Error("Cannot initialize without resolution");
        }

        // Resize canvas to match output resolution
        this.canvas.width = this.resolution.width * this.scale;
        this.canvas.height = this.resolution.height * this.scale;

        // Create context
        this.context = new WebGPUContext(this.params.gpu, this.resolution, this.canvas, this.scale, this.debug);
        globalThis.context = this.context;

        // Create network
        this.network = new NetworkList[this.params.network_name](this.params.weights);

        // Create renderer
        this.renderer = new WebSRRenderer(this.network);

        this.initialized = true;
    }

    private updateResolution(newResolution: Resolution) {
        // Check if resolution actually changed
        if(this.resolution &&
           this.resolution.width === newResolution.width &&
           this.resolution.height === newResolution.height) {
            return; // No change needed
        }

        console.log(`Resolution changed from ${this.resolution?.width}x${this.resolution?.height} to ${newResolution.width}x${newResolution.height}`);

        this.resolution = newResolution;

        // Clean up old resources if they exist
        if(this.context) {
            this.context.destroy();
        }

        // Re-initialize with new resolution
        this.initialize();
    }

    switchNetwork(network: NetworkName, weights: any){
        if(!NetworkList[network]) throw Error(`Network ${network} is not defined or implemented`);
        this.network = new NetworkList[network](weights);
        if(this.renderer) {
            this.renderer.switchNetwork(this.network);
        }
    }

    static async initWebGPU(): Promise<GPUDevice | false>{

        if(!navigator.gpu) return false;

        const adapter = await navigator.gpu.requestAdapter();

        if(!adapter) return false;

        const device = await adapter.requestDevice();

        if(!device) return false;

        return device;
    }


    async render(source: MediaSource){
        if(!source) {
            throw new Error("render() requires a source parameter");
        }

        // Detect resolution from source
        const sourceResolution: Resolution = {
            width: getSourceWidth(source),
            height: getSourceHeight(source)
        };

        // Initialize if not yet initialized, or update if resolution changed
        if(!this.initialized) {
            this.resolution = sourceResolution;
            this.initialize();
        } else if(sourceResolution.width !== this.resolution!.width ||
                  sourceResolution.height !== this.resolution!.height) {
            this.updateResolution(sourceResolution);
        }

        await this.renderer!.render(source);
    }

    async destroy(){
        if(this.context) {
            this.context.destroy();
        }
    }



}
