import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";

export default class WebSRRenderer{
    private context: WebGPUContext;
    private network: NeuralNetwork;



    constructor(context: WebGPUContext, network: NeuralNetwork) {

        this.context = context;
        this.network = network;

    }



    render(){

    }



}