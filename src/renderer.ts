import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";

export default class WebSRRenderer{
    private device: GPUDevice;
    private network: NeuralNetwork;



    constructor(device: GPUDevice, network: NeuralNetwork) {

        this.device = device;
        this.network = network;

    }

    loadImage(image: ImageBitmap){

        const texture = this.device.createTexture({
            label: 'Input Image',
            size: [image.width, image.height],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture({source: image}, {texture}, [image.width, image.height]);
        
    }



    render(){

    }



}