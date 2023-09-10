import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";

export default class WebSRRenderer{
    private context: WebGPUContext;
    private network: NeuralNetwork;



    constructor(context: WebGPUContext, network: NeuralNetwork) {

        this.context = context;
        this.network = network;

    }

    loadImage(image: ImageBitmap){

        const device  = this.context.device;

        device.queue.copyExternalImageToTexture({source: image}, {texture:this.context.texture('input')}, [image.width, image.height]);

    }



    render(){
        this.network.feedForward();
    }



}