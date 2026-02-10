import ComputeLayer from "../base_compute_layer";
declare class Anime4KConv112x4 extends ComputeLayer {
    label: string;
    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any, first: boolean);
    defaultPipelineConfig(): GPUComputePipelineDescriptor;
    defaultBindGroup(): GPUBindGroup;
}
export default Anime4KConv112x4;
