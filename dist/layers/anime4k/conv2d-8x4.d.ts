import ComputeLayer from "../base_compute_layer";
declare class Anime4KConv8x4 extends ComputeLayer {
    label: string;
    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any);
}
export default Anime4KConv8x4;
