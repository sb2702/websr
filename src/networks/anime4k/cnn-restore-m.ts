import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import Anime4KConv8x4 from "../../layers/anime4k/conv2d-8x4";
import Anime4KConv56x4 from "../../layers/anime4k/conv2d-56x4";
import DisplayLayer from "../../layers/anime4k/display_1x";
import {MediaSource, isHTMLVideoElement, isImageBitmap, getSourceWidth, getSourceHeight} from "../../utils";


export default class Anime4KCNNRM extends NeuralNetwork{

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

        const conv2d_3_tf = new Anime4KConv8x4([context.buffer('conv2d_2_tf')], context.buffer('conv2d_3_tf'), weights['conv2d_3_tf']);

        const conv2d_4_tf = new Anime4KConv8x4([context.buffer('conv2d_3_tf')], context.buffer('conv2d_4_tf'), weights['conv2d_4_tf']);

        const conv2d_5_tf = new Anime4KConv8x4([context.buffer('conv2d_4_tf')], context.buffer('conv2d_5_tf'), weights['conv2d_5_tf']);

        const conv2d_6_tf = new Anime4KConv8x4([context.buffer('conv2d_5_tf')], context.buffer('conv2d_6_tf'), weights['conv2d_6_tf']);

        const conv2d_out_tf = new Anime4KConv56x4([context.buffer('conv2d_tf'), context.buffer('conv2d_1_tf'), context.buffer('conv2d_2_tf'), context.buffer('conv2d_3_tf'), context.buffer('conv2d_4_tf'), context.buffer('conv2d_5_tf'), context.buffer('conv2d_6_tf')], context.buffer('conv2d_out_tf'), weights['conv2d_out_tf']);

        const paint = new DisplayLayer([context.buffer('conv2d_out_tf'), context.input], context.texture('output'));

        layers.push(conv2d_tf, conv2d_1_tf, conv2d_2_tf, conv2d_3_tf, conv2d_4_tf, conv2d_5_tf, conv2d_6_tf, conv2d_out_tf, paint);

        return layers;

    }

    async feedForward(source?: MediaSource){


        if(isHTMLVideoElement(source)){

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
        this.layers[this.layers.length - 1].inputs[1] = this.context.input;


        this.layers.forEach(function (layer) {
           layer.run();
        });

    }

}

