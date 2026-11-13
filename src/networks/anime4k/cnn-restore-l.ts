import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import Anime4KConv16x4 from "../../layers/anime4k/conv2d-16x4";
import DisplayLayer from "../../layers/anime4k/display_1x";


export default class Anime4KCNNRL extends NeuralNetwork{

    constructor(weights: any) {
        super(weights);

    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const weights = this.weights.layers;

        const context = this.context;

       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf'), weights['conv2d_tf']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf1'), weights['conv2d_tf1']));


       for (let i=1; i < 4; i++){
           let source = (i==1) ? `conv2d_tf` : `conv2d_${i-1}_tf`;
           layers.push(new Anime4KConv16x4([context.buffer(source), context.buffer(source + "1")], context.buffer(`conv2d_${i}_tf`), weights[`conv2d_${i}_tf`]));
           layers.push(new Anime4KConv16x4([context.buffer(source), context.buffer(source + "1")], context.buffer(`conv2d_${i}_tf1`), weights[`conv2d_${i}_tf1`]));
       }


        layers.push(new Anime4KConv16x4([context.buffer('conv2d_3_tf'), context.buffer('conv2d_3_tf1')], context.buffer(`conv2d_out_tf`), weights[`conv2d_out_tf`]));


        const paint = new DisplayLayer([context.buffer('conv2d_out_tf'), context.input], context.texture('output'));

        layers.push(paint);

        return layers;

    }

    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture){

        this.bindInput(input, output, [[0, 0], [1, 0], [-1, 1]]);

        this.layers.forEach(function (layer) {
           layer.run();
        });

    }

}

