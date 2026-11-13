import ComputeLayer from "../base_compute_layer";
declare class Anime4KConv224x4 extends ComputeLayer {
    label: string;
    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any, index: number);
    defaultPipelineConfig(): GPUComputePipelineDescriptor;
    defaultBindGroup(): GPUBindGroup;
}
export default Anime4KConv224x4;
