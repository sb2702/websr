import WebGPUContext from "../../context";
import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import Anime4KConv8x4 from "../../layers/anime4k/conv2d-8x4";
import PixelShuffle2X from "../../layers/anime4k/pixel-shuffle";
import DisplayLayer from "../../layers/anime4k/display";


export default class Anime4KCNN2XS extends NeuralNetwork{

    constructor(weights: any) {
        super(weights);

    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const weights = this.weights.layers;

        const context = this.context;


        const conv2d_tf = new Anime4KConv3x4([context.input], context.texture('conv2d_tf'), weights['conv2d_tf']);

        const conv2d_1_tf = new Anime4KConv8x4([context.texture('conv2d_tf')], context.texture('conv2d_1_tf'), weights['conv2d_1_tf']);

        const conv2d_2_tf = new Anime4KConv8x4([context.texture('conv2d_1_tf')], context.texture('conv2d_2_tf'), weights['conv2d_2_tf']);

        const conv2d_last_tf = new Anime4KConv8x4([context.texture('conv2d_2_tf')], context.texture('conv2d_last_tf'), weights['conv2d_last_tf']);

        const pixel_shuffle = new PixelShuffle2X( [context.texture('conv2d_last_tf')], context.texture('pixel_shuffle', {width: context.resolution.width*2, height: context.resolution.height*2, format: "r32float"}));

       const paint = new DisplayLayer([context.texture('pixel_shuffle'), context.input], context.texture('output'));

        layers.push(conv2d_tf, conv2d_1_tf, conv2d_2_tf, conv2d_last_tf, pixel_shuffle, paint);

        return layers;

    }

    async feedForward(source?: HTMLVideoElement | HTMLImageElement){


        if(source instanceof HTMLVideoElement){
            this.context.input = this.context.device.importExternalTexture({source});
        } else {

            const image = await createImageBitmap(source);
            this.context.input = this.context.device.queue.copyExternalImageToTexture({source: image}, {texture:this.context.texture('input')}, [source.naturalWidth, source.naturalHeight]);
        }

        this.layers[0].inputTextures[0] = this.context.input;
        this.layers[5].inputTextures[1] = this.context.input;


        this.layers.forEach(function (layer) {
           layer.run();
        });

    }



}

