import WebGPUContext from "../context";
import Layer from "../layers/base_layer";
import {MediaSource} from "../utils";


class NeuralNetwork {
    context: WebGPUContext;
    layers: Layer[];
    weights: any;

    constructor(weights?: any) {


        this.weights = weights;
        this.context = globalThis.context;
        this.layers = this.model();
    }


    model(): Layer[]{

        return [];
    }

    lastLayer(): Layer {
        return this.layers[this.layers.length-1];
    }

    async feedForward(source?: MediaSource){


        this.layers.forEach(layer => {
            layer.run();
        });


    }



}


export default NeuralNetwork