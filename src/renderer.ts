import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";
import DisplayLayer from "./layers/anime4k/display";

export default class WebSRRenderer{
    private context: WebGPUContext;
    private network: NeuralNetwork;

    source: HTMLVideoElement | HTMLImageElement;
    active: boolean;
    vfc: number;



    constructor(network: NeuralNetwork, source: HTMLVideoElement | HTMLImageElement) {

        this.context = globalThis.context;
        this.network = network;

        this.source = source;

        this.active = false;

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
        if(this.vfc && this.source instanceof HTMLVideoElement) this.source.cancelVideoFrameCallback(this.vfc);
    }

    async renderStep(){


        const lastLayer = this.network.lastLayer();

        if(lastLayer instanceof DisplayLayer) lastLayer.setOutput(this.context.context.getCurrentTexture());

        await this.render();

        if(this.active && this.source instanceof HTMLVideoElement) {
            this.vfc = this.source.requestVideoFrameCallback(this.renderStep.bind(this));
        }

    }




    async render(){
        await this.network.feedForward(this.source);
    }



}