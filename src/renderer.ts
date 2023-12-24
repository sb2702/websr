import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";
import DisplayLayer from "./layers/anime4k/display";
import DisplayLayer3C from "./layers/anime4k/display_3c";

export default class WebSRRenderer{
    private context: WebGPUContext;
    private network: NeuralNetwork;

    source?: HTMLVideoElement | HTMLImageElement | ImageBitmap;
    active: boolean;
    vfc: number;



    constructor(network: NeuralNetwork, source?: HTMLVideoElement | HTMLImageElement | ImageBitmap) {

        this.context = globalThis.context;
        this.network = network;

        this.source = source;

        this.active = false;

    }


    switchNetwork(network: NeuralNetwork){
        this.network = network;
    }

    async start(){

        if(this.context.destroyed) {
            throw new Error("WebSR instance was destroyed");
        }

        this.active = true;
        await this.renderStep();

    }

    async stop(){
        this.active = false;
        if(this.vfc && this.source && this.source instanceof HTMLVideoElement) this.source.cancelVideoFrameCallback(this.vfc);
    }

    async renderStep(){


        const lastLayer = this.network.lastLayer();

        if(lastLayer instanceof DisplayLayer) lastLayer.setOutput(this.context.context.getCurrentTexture());

        await this.render();

        if(this.active && this.source && this.source instanceof HTMLVideoElement) {
            this.vfc = this.source.requestVideoFrameCallback(this.renderStep.bind(this));
        }

    }




    async render(source?: ImageBitmap){

        if(source){
            const lastLayer = this.network.lastLayer();
            if(lastLayer instanceof DisplayLayer || lastLayer instanceof DisplayLayer3C) lastLayer.setOutput(this.context.context.getCurrentTexture());
        }

        await this.network.feedForward(source? source: this.source);

        if(source) await this.context.device.queue.onSubmittedWorkDone();
    }



}