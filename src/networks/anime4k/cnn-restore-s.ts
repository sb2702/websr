import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import Anime4KConv8x4 from "../../layers/anime4k/conv2d-8x4";
import DisplayLayer from "../../layers/anime4k/display_1x";


export default class Anime4KCNNRS extends NeuralNetwork{

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

        const conv2d_last_tf = new Anime4KConv8x4([context.buffer('conv2d_2_tf')], context.buffer('conv2d_out_tf'), weights['conv2d_out_tf']);

        const paint = new DisplayLayer([context.buffer('conv2d_out_tf'), context.input], context.texture('output'));

        layers.push(conv2d_tf, conv2d_1_tf, conv2d_2_tf, conv2d_last_tf,  paint);

        return layers;

    }

    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture){

        this.bindInput(input, output, [[0, 0], [-1, 1]]);

        this.layers.forEach(function (layer) {
           layer.run();
        });

    }
}

