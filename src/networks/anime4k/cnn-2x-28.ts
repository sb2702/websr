import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import DisplayLayer3C from "../../layers/anime4k/display_3c";
import Anime4KConv56x4 from "../../layers/anime4k/conv2d-56x4-2";
import Anime4KConv392x4 from "../../layers/anime4k/conv2d-392x4";
import Anime4KConcat7 from "../../layers/anime4k/conv2d-concat7";


export default class Anime4KCNN2X28 extends NeuralNetwork{

    constructor(weights: any) {
        super(weights);

    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const weights = this.weights.layers;

        const context = this.context;



       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf'), weights['conv2d_tf']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf1'), weights['conv2d_tf1']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf2'), weights['conv2d_tf2']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf3'), weights['conv2d_tf3']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf4'), weights['conv2d_tf4']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf5'), weights['conv2d_tf5']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf6'), weights['conv2d_tf6']));


       for (let i=1; i < 7; i++){
           let source = (i==1) ? `conv2d_tf` : `conv2d_${i-1}_tf`;

           const sources = [];
           for(let j =0; j < 7; j++){
            if(j==0){
                sources.push(context.buffer(source));
            } else {
                sources.push(context.buffer(source + `${j}`));
            }
           }

           for(let j =0; j < 7; j++){

            if(j==0){
                layers.push(new Anime4KConv56x4(sources, context.buffer(`conv2d_${i}_tf`), weights[`conv2d_${i}_tf`]));
            } else {
                layers.push(new Anime4KConv56x4(sources, context.buffer(`conv2d_${i}_tf${j}`), weights[`conv2d_${i}_tf${j}`]));
            }

           }

       }


       for (let c =0; c < 3; c++){

           const sources_0 = [];
           const sources_1 = [];
           const sources_2 = [];
           const sources_3 = [];
           const sources_4 = [];
           const sources_5 = [];
           const sources_6 = [];
           for(let i =0; i < 7; i++){
               let source = (i==0) ? `conv2d_tf` : `conv2d_${i}_tf`;
               sources_0.push(context.buffer(source));
               sources_1.push(context.buffer(source + "1"));
               sources_2.push(context.buffer(source + "2"));
               sources_3.push(context.buffer(source + "3"));
               sources_4.push(context.buffer(source + "4"));
               sources_5.push(context.buffer(source + "5"));
               sources_6.push(context.buffer(source + "6"));
           }

           const dest = (c==0) ? `conv2d_last_tf` : `conv2d_last_tf${c}`;
           layers.push(new Anime4KConv392x4(sources_0, context.buffer(`conv2d_last_${c}_pt1`), weights[dest], 0));
           layers.push(new Anime4KConv392x4(sources_1, context.buffer(`conv2d_last_${c}_pt2`), weights[dest], 1));
           layers.push(new Anime4KConv392x4(sources_2, context.buffer(`conv2d_last_${c}_pt3`), weights[dest], 2));
           layers.push(new Anime4KConv392x4(sources_3, context.buffer(`conv2d_last_${c}_pt4`), weights[dest], 3));
           layers.push(new Anime4KConv392x4(sources_4, context.buffer(`conv2d_last_${c}_pt5`), weights[dest], 4));
           layers.push(new Anime4KConv392x4(sources_5, context.buffer(`conv2d_last_${c}_pt6`), weights[dest], 5));
           layers.push(new Anime4KConv392x4(sources_6, context.buffer(`conv2d_last_${c}_pt7`), weights[dest], 6));

           layers.push(new Anime4KConcat7([context.buffer(`conv2d_last_${c}_pt1`), context.buffer(`conv2d_last_${c}_pt2`), context.buffer(`conv2d_last_${c}_pt3`), context.buffer(`conv2d_last_${c}_pt4`), context.buffer(`conv2d_last_${c}_pt5`), context.buffer(`conv2d_last_${c}_pt6`), context.buffer(`conv2d_last_${c}_pt7`)], context.buffer(dest), weights[dest]))
       }



       const paint = new DisplayLayer3C([context.buffer('conv2d_last_tf'), context.buffer('conv2d_last_tf1'), context.buffer('conv2d_last_tf2'), context.input], context.texture('output'));

       layers.push(paint);

        return layers;

    }

    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture){

        this.bindInput(input, output, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [-1, 3]]);

        this.layers.forEach(function (layer) {
           layer.run();
        });

    }

}

