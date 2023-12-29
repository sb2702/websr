import Anime4KCNN2XS from "./anime4k/cnn-2x-s";
import Anime4KCNN2XM from "./anime4k/cnn-2x-m";
import Anime4KCNN2XL from "./anime4k/cnn-2x-l";
import Anime4KCNNRL from "./anime4k/cnn-restore-l";
import Anime4KCNNRM from "./anime4k/cnn-restore-m";
import Anime4KCNNRS from "./anime4k/cnn-restore-s";
import PoCNetwork from "./poc_network";
import NeuralNetwork from "./base_network";



type NeuralNetworkConstructor = new(...args: any[]) => NeuralNetwork


export type NetworkName = "anime4k/cnn-2x-l"|  "anime4k/cnn-2x-m"|  "anime4k/cnn-2x-s"|'sb2702/blur-poc' | "anime4k/cnn-restore-s" | "anime4k/cnn-restore-m" | "anime4k/cnn-restore-l";

export type DisplayScale = 1 | 2;

export const NetworkList: Record<NetworkName, NeuralNetworkConstructor> = {
    "anime4k/cnn-2x-s": Anime4KCNN2XS,
    "anime4k/cnn-2x-m": Anime4KCNN2XM,
    "anime4k/cnn-2x-l": Anime4KCNN2XL,
    "anime4k/cnn-restore-s": Anime4KCNNRS,
    "anime4k/cnn-restore-m": Anime4KCNNRM,
    "anime4k/cnn-restore-l": Anime4KCNNRL,
    "sb2702/blur-poc": PoCNetwork
}

export const NetworkScales: Record<NetworkName, DisplayScale> = {
    "anime4k/cnn-2x-s": 2,
    "anime4k/cnn-2x-m": 2,
    "anime4k/cnn-2x-l": 2,
    "anime4k/cnn-restore-s": 1,
    "anime4k/cnn-restore-m": 1,
    "anime4k/cnn-restore-l": 1,
    "sb2702/blur-poc": 1
}








