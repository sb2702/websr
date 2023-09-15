import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";
import DisplayLayer from "./layers/anime4k/display";

export default class WebSRRenderer{
    private context: WebGPUContext;
    private network: NeuralNetwork;

    video: HTMLVideoElement;
    active: boolean;
    vfc: number;



    constructor(network: NeuralNetwork, video: HTMLVideoElement) {

        this.context = globalThis.context;
        this.network = network;

        this.video = video;

        this.active = false;

    }



    async start(){
        this.active = true;
        await this.renderStep();

    }

    async stop(){
        this.active = false;
        if(this.vfc) this.video.cancelVideoFrameCallback(this.vfc);
    }

    async renderStep(){


        const lastLayer = this.network.lastLayer();

        if(lastLayer instanceof DisplayLayer) lastLayer.setOutput(this.context.context.getCurrentTexture());

        await this.render();

        if(this.active) {
            this.vfc = this.video.requestVideoFrameCallback(this.renderStep.bind(this));
        }

    }




    async render(){
        await this.network.feedForward(this.video);
    }



}