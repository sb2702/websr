import ComputeLayer from "../base_compute_layer";
declare class SR2XSDepthwise extends ComputeLayer {
    label: string;
    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any);
    lazyLoadSetup(): void;
}
export default SR2XSDepthwise;
