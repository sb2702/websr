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

        const texture = device.createTexture({
            label: 'Input Image',
            size: [image.width, image.height],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        device.queue.copyExternalImageToTexture({source: image}, {texture}, [image.width, image.height]);

    }



    render(){

    }



}