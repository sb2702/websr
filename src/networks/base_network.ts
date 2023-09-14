import WebGPUContext from "../context";
import Layer from "../layers/base_layer";


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

    async feedForward(video?: HTMLVideoElement){

        this.layers.forEach(layer => {
            layer.run();
        });


    }



}


export default NeuralNetwork