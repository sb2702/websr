import ComputeLayer from "../base_compute_layer";
declare class Anime4KConcat2 extends ComputeLayer {
    label: string;
    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any);
}
export default Anime4KConcat2;
