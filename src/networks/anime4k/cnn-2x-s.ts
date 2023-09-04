import WebGPUContext from "../../context";
import Layer from "../../layers/base_layer";
import NeuralNetwork from "../base_network";
import RGB2YUV from "../../layers/rgb_2_yuv";
import GuassianLayer from "../../layers/gaussian";


export default class Anime4KCNN2XS extends NeuralNetwork{

    constructor(context: WebGPUContext, weights: any) {
        super(context, weights);
        console.log("Loading Anime4K network");
        console.log("Network weights");
        console.log(weights);
    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const context = this.context;

        layers.push(new RGB2YUV(context.device, context.input, context.texture('yuv')));

        layers.push(new GuassianLayer(this.context.device, context.texture('yuv'), this.context.output));

        return layers;


    }


}

