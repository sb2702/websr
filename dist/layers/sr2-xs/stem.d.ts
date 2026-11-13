import ComputeLayer from "../base_compute_layer";
declare class SR2XSStem extends ComputeLayer {
    label: string;
    constructor(inputTextures: (GPUTexture | GPUExternalTexture)[], outputBuffer: GPUBuffer, weights: any);
    lazyLoadSetup(): void;
}
export default SR2XSStem;
