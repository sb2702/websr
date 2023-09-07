import WebGPUContext from "../../context";
import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import RGB2YUV from "../../layers/rgb_2_yuv";
import GuassianLayer from "../../layers/gaussian";
import Anime4KConv3x4 from "../../layers/anime4k/conv2d-3x4";
import Anime4KConv8x4 from "../../layers/anime4k/conv2d-8x4";
import DummyLayer from "../../layers/dummy";
import PixelShuffle2X from "../../layers/pixel-shuffle";


export default class Anime4KCNN2XS extends NeuralNetwork{

    constructor(context: WebGPUContext, weights: any) {
        super(context, weights);
        console.log("Loading Anime4K network");
        console.log("Network weights");
        console.log(weights);


    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const weights = this.weights.layers;

        const context = this.context;

        const conv2d_tf = new Anime4KConv3x4(context.device, context.input, context.texture('conv2d_tf'), weights['conv2d_tf']);

        const conv2d_1_tf = new Anime4KConv8x4(context.device, context.texture('conv2d_tf'), context.texture('conv2d_1_tf'), weights['conv2d_1_tf']);

        const conv2d_2_tf = new Anime4KConv8x4(context.device, context.texture('conv2d_1_tf'), context.texture('conv2d_2_tf'), weights['conv2d_2_tf']);

        const conv2d_last_tf = new Anime4KConv8x4(context.device, context.texture('conv2d_2_tf'), context.texture('conv2d_last_tf'), weights['conv2d_last_tf']);

        const dummy_layer = new DummyLayer(context.device, context.input, context.texture('dummy'));

        const pixel_shuffle = new PixelShuffle2X(context.device, context.texture('dummy'), this.context.output);


        layers.push(conv2d_tf, conv2d_1_tf, conv2d_2_tf, conv2d_last_tf, dummy_layer, pixel_shuffle);

        //layers.push(new RGB2YUV(context.device, context.input, context.texture('yuv')));

      //  layers.push(new GuassianLayer(this.context.device, context.texture('yuv'), this.context.output));

        return layers;


    }


}

