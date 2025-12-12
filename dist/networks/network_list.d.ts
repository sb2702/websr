import NeuralNetwork from "./base_network";
type NeuralNetworkConstructor = new (...args: any[]) => NeuralNetwork;
export type NetworkName = "anime4k/cnn-2x-l" | "anime4k/cnn-2x-m" | "anime4k/cnn-2x-s" | 'sb2702/blur-poc' | "anime4k/cnn-restore-s" | "anime4k/cnn-restore-m" | "anime4k/cnn-restore-l";
export type DisplayScale = 1 | 2;
export declare const NetworkList: Record<NetworkName, NeuralNetworkConstructor>;
export declare const NetworkScales: Record<NetworkName, DisplayScale>;
export {};
