import ComputeLayer from "../base_compute_layer";
declare class RealESRGANIngest extends ComputeLayer {
    label: string;
    num_work_groups: number;
    private useF16;
    constructor(inputTexture: GPUTexture | GPUExternalTexture, outputBuffer: GPUBuffer, useF16?: boolean);
    private _builtForExternal;
    lazyLoadSetup(): void;
}
export default RealESRGANIngest;
