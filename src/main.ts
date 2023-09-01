
import WebGPUContext from './context'
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import PoCNetwork from "./networks/poc_network";


export default class WebSR {
    private canvas: HTMLCanvasElement;
    private context: WebGPUContext;
    private network: NeuralNetwork;
    private renderer: WebSRRenderer;

    constructor(device: GPUDevice, canvas: HTMLCanvasElement) {

        this.canvas = canvas;
        this.context = new WebGPUContext(device, canvas);
        this.network = new PoCNetwork(this.context);
        this.renderer = new WebSRRenderer(this.context, this.network);

    }

    static async initWebGPU(): Promise<GPUDevice | false>{

        if(!navigator.gpu) return false;

        const adapter = await navigator.gpu.requestAdapter();

        if(!adapter) return false;

        const device = await adapter.requestDevice();

        if(!device) return false;

        return device;
    }

    async loadImage(image: ImageBitmap){
        await this.renderer.loadImage(image);
    }

    async render(){
        await this.renderer.render()
    }


}
