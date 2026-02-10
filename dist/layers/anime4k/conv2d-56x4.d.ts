import ComputeLayer from "../base_compute_layer";
declare class Anime4KConv56x4 extends ComputeLayer {
    label: string;
    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any);
    defaultPipelineConfig(): GPUComputePipelineDescriptor;
    defaultBindGroup(): GPUBindGroup;
}
export default Anime4KConv56x4;
