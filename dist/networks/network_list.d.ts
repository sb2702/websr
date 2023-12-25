import NeuralNetwork from "./base_network";
type NeuralNetworkConstructor = new (...args: any[]) => NeuralNetwork;
export type NetworkName = "anime4k/cnn-2x-l" | "anime4k/cnn-2x-m" | "anime4k/cnn-2x-s" | 'sb2702/blur-poc';
export declare const NetworkList: Record<NetworkName, NeuralNetworkConstructor>;
export {};
