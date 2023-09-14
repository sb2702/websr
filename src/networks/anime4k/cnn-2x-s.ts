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


        const conv2d_tf = new Anime4KConv3x4([context.texture('input')], context.texture('conv2d_tf'), weights['conv2d_tf']);

        const conv2d_1_tf = new Anime4KConv8x4([context.texture('conv2d_tf')], context.texture('conv2d_1_tf'), weights['conv2d_1_tf']);

        const conv2d_2_tf = new Anime4KConv8x4([context.texture('conv2d_1_tf')], context.texture('conv2d_2_tf'), weights['conv2d_2_tf']);

        const conv2d_last_tf = new Anime4KConv8x4([context.texture('conv2d_2_tf')], context.texture('conv2d_last_tf'), weights['conv2d_last_tf']);

        const pixel_shuffle = new PixelShuffle2X( [context.texture('conv2d_last_tf')], context.texture('pixel_shuffle', {width: context.resolution.width*2, height: context.resolution.height*2, format: "r32float"}));

       const paint = new DisplayLayer([context.texture('pixel_shuffle'), context.texture('input')], context.texture('output'));

        layers.push(conv2d_tf, conv2d_1_tf, conv2d_2_tf, conv2d_last_tf, pixel_shuffle, paint);

        return layers;

    }

    async feedForward(video?: HTMLVideoElement){




        this.layers.forEach(function (layer) {
           layer.run();
        });


        //

    }



}

