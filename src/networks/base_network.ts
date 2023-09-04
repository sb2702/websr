import WebGPUContext from "../context";
import Layer from "../layers/base_layer";


class NeuralNetwork {
    context: WebGPUContext;
    layers: Layer[];
    weights: any;

    constructor(context: WebGPUContext, weights?: any) {

        this.context = context;
        this.weights = weights;
        this.layers = this.model();
    }


    model(): Layer[]{



        return [];
    }

    feedForward(){

        this.layers.forEach(layer => {
            layer.run();
        });


    }



}


export default NeuralNetwork