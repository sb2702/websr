import ComputeLayer from "../base_compute_layer";
declare class RealESRGANConv2DPReLU extends ComputeLayer {
    label: string;
    num_work_groups: number;
    private kernelsBuf;
    private biasBuf;
    private alphaBuf;
    private paramsBuf;
    private inputBuf;
    private outBuf;
    private inVec4s;
    private outVec4s;
    private applyPReLU;
    constructor(inputBuf: GPUBuffer, outputBuf: GPUBuffer, layerWeights: any, useF16?: boolean);
}
export default RealESRGANConv2DPReLU;
