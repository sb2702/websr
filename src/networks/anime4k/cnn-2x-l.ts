import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import Anime4KConv8x4 from "../../layers/anime4k/conv2d-8x4";
import Anime4KConv56x4 from "../../layers/anime4k/conv2d-56x4";
import DisplayLayer3C from "../../layers/anime4k/display_3c";
import Anime4KConv16x4 from "../../layers/anime4k/conv2d-16x4";
import Anime4KConv112x4 from "../../layers/anime4k/conv2d-112x4";
import Anime4KConcat2 from "../../layers/anime4k/conv2d-concat2";
import {MediaSource, isHTMLVideoElement, isVideoFrame, isImageBitmap, getSourceWidth, getSourceHeight} from "../../utils";


export default class Anime4KCNN2XL extends NeuralNetwork{

    constructor(weights: any) {
        super(weights);

    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const weights = this.weights.layers;

        const context = this.context;

       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf'), weights['conv2d_tf']));
       layers.push(new Anime4KConv3x4([context.input], context.buffer('conv2d_tf1'), weights['conv2d_tf1']));


       for (let i=1; i < 7; i++){
           let source = (i==1) ? `conv2d_tf` : `conv2d_${i-1}_tf`;
           layers.push(new Anime4KConv16x4([context.buffer(source), context.buffer(source + "1")], context.buffer(`conv2d_${i}_tf`), weights[`conv2d_${i}_tf`]));
           layers.push(new Anime4KConv16x4([context.buffer(source), context.buffer(source + "1")], context.buffer(`conv2d_${i}_tf1`), weights[`conv2d_${i}_tf1`]));
       }


       for (let c =0; c < 3; c++){

           const sources_0 = [];
           const sources_1 = [];

           for(let i =0; i < 7; i++){
               let source = (i==0) ? `conv2d_tf` : `conv2d_${i}_tf`;
               sources_0.push(context.buffer(source));
               sources_1.push(context.buffer(source + "1"));

           }

           const dest = (c==0) ? `conv2d_last_tf` : `conv2d_last_tf${c}`;
           layers.push(new Anime4KConv112x4(sources_0, context.buffer(`conv2d_last_${c}_pt1`), weights[dest], true));
           layers.push(new Anime4KConv112x4(sources_1, context.buffer(`conv2d_last_${c}_pt2`), weights[dest], false));
           layers.push(new Anime4KConcat2([context.buffer(`conv2d_last_${c}_pt1`), context.buffer(`conv2d_last_${c}_pt2`)], context.buffer(dest), weights[dest]))
       }



       const paint = new DisplayLayer3C([context.buffer('conv2d_last_tf'), context.buffer('conv2d_last_tf1'), context.buffer('conv2d_last_tf2'), context.input], context.texture('output'));

       layers.push(paint);

        return layers;

    }

    async feedForward(source?: MediaSource){


        if(isHTMLVideoElement(source) || isVideoFrame(source)){

            this.context.input = this.context.device.importExternalTexture({source});
        } else {

            const bitmap = isImageBitmap(source) ? source : await createImageBitmap(source);
            const width = getSourceWidth(source);
            const height = getSourceHeight(source);
            this.context.device.queue.copyExternalImageToTexture({source: bitmap}, {texture:this.context.texture('input', {format: "rgba8unorm"})}, [width, height]);
            this.context.input = this.context.texture('input');
        }

        this.layers[0].inputs[0] = this.context.input;
        this.layers[1].inputs[0] = this.context.input;
        this.layers[this.layers.length-1].inputs[3] = this.context.input

        this.layers[0].lazyLoadSetup();
        this.layers[this.layers.length-1].lazyLoadSetup();


        this.layers.forEach(function (layer) {
           layer.run();
        });

    }

}

