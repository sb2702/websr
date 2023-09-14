import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";

export default class WebSRRenderer{
    private context: WebGPUContext;
    private network: NeuralNetwork;

    video: HTMLVideoElement;



    constructor(network: NeuralNetwork) {

        this.context = globalThis.context;
        this.network = network;

    }


    async loadVideo(video: HTMLVideoElement){


        this.video = video;

        let image = await createImageBitmap(this.video);
        this.context.device.queue.copyExternalImageToTexture({source: image}, {texture:this.context.texture('input')}, [image.width, image.height]);

    }


    loadImage(image: ImageBitmap){

        const device  = this.context.device;

        device.queue.copyExternalImageToTexture({source: image}, {texture:this.context.texture('input')}, [image.width, image.height]);

    }


    start(){

        this.renderStep();
    }

    async renderStep(){


        this.context.textures['output'] = this.context.context.getCurrentTexture();

        this.network.layers[this.network.layers.length-1].outputTexture = this.context.textures['output'];


        await this.render();

        this.video.requestVideoFrameCallback(this.renderStep.bind(this));

    }




    async render(){

        if(this.video){
            let image = await createImageBitmap(this.video);
            this.context.device.queue.copyExternalImageToTexture({source: image}, {texture:this.context.texture('input')}, [image.width, image.height]);
        }

        this.network.feedForward(this.video);
    }



}