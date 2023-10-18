import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import Anime4KConv8x4 from "../../layers/anime4k/conv2d-8x4";
import DisplayLayer from "../../layers/anime4k/display";


export default class Anime4KCNN2XS extends NeuralNetwork{

    constructor(weights: any) {
        super(weights);

    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const weights = this.weights.layers;

        const context = this.context;

        const conv2d_tf = new Anime4KConv3x4([context.input], context.buffer('conv2d_tf'), weights['conv2d_tf']);

        const conv2d_1_tf = new Anime4KConv8x4([context.buffer('conv2d_tf')], context.buffer('conv2d_1_tf'), weights['conv2d_1_tf']);

        const conv2d_2_tf = new Anime4KConv8x4([context.buffer('conv2d_1_tf')], context.buffer('conv2d_2_tf'), weights['conv2d_2_tf']);

        const conv2d_last_tf = new Anime4KConv8x4([context.buffer('conv2d_2_tf')], context.buffer('conv2d_last_tf'), weights['conv2d_last_tf']);

        const paint = new DisplayLayer([context.buffer('conv2d_last_tf'), context.input], context.texture('output'));

        layers.push(conv2d_tf, conv2d_1_tf, conv2d_2_tf, conv2d_last_tf,  paint);

        return layers;

    }

    async feedForward(source?: HTMLVideoElement | HTMLImageElement){


        if(source instanceof HTMLVideoElement){
            this.context.input = this.context.device.importExternalTexture({source});
        } else {

            const image = await createImageBitmap(source);
            this.context.device.queue.copyExternalImageToTexture({source: image}, {texture:this.context.texture('input', {format: "rgba8unorm"})}, [source.naturalWidth, source.naturalHeight]);
            this.context.input = this.context.texture('input');
        }

        this.layers[0].inputs[0] = this.context.input;
        this.layers[4].inputs[1] = this.context.input;


        this.layers.forEach(function (layer) {
           layer.run();
        });

    }



}

