import Anime4KCNN2XS from "./anime4k/cnn-2x-s";
import PoCNetwork from "./poc_network";
import NeuralNetwork from "./base_network";



type NeuralNetworkConstructor = new(...args: any[]) => NeuralNetwork

export type NetworkName = "anime4k/cnn-2x-s"|'sb2702/blur-poc';

export const NetworkList: Record<NetworkName, NeuralNetworkConstructor> = {
    "anime4k/cnn-2x-s": Anime4KCNN2XS,
    "sb2702/blur-poc": PoCNetwork
}







