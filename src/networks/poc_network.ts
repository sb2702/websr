import WebGPUContext from "../context";
import Layer from "../layers/base_layer";
import NeuralNetwork from "./base_network";
import RGB2YUV from "../layers/utils/rgb_2_yuv";
import GuassianLayer from "../layers/utils/gaussian";


export default class PoCNetwork extends NeuralNetwork{

    constructor() {
        super();
    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const context = this.context;

        layers.push(new RGB2YUV([context.texture('input')], context.texture('yuv')));

        layers.push(new GuassianLayer( [context.texture('yuv')], context.texture('output')));

        return layers;


    }


}

