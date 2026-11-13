import Layer from "../layers/base_layer";
import NeuralNetwork from "./base_network";
import RGB2YUV from "../layers/utils/rgb_2_yuv";
import GuassianLayer from "../layers/utils/gaussian";


// Proof-of-concept "network": RGB -> YUV, then a 3x3 blur on the luma channel.
export default class PoCNetwork extends NeuralNetwork{

    constructor(weights?: any) {
        super(weights);
    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const context = this.context;

        layers.push(new RGB2YUV([context.input], context.texture('yuv')));

        layers.push(new GuassianLayer( [context.texture('yuv')], context.texture('output')));

        return layers;


    }

    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture){
        this.bindInput(input, output, [[0, 0]]);
        this.layers.forEach(layer => layer.run());
    }

}
