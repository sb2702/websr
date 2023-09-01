import WebGPUContext from "../context";
import Layer from "../layers/base_layer";


class NeuralNetwork {
    private context: WebGPUContext;
    private layers: Layer[];

    constructor(context: WebGPUContext) {

        this.context = context;
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