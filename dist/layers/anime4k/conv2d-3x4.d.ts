import ComputeLayer from "../base_compute_layer";
declare class Anime4KConv3x4 extends ComputeLayer {
    label: string;
    constructor(inputTextures: (GPUTexture | GPUExternalTexture)[], outputBuffer: GPUBuffer, weights: any);
    lazyLoadSetup(): void;
}
export default Anime4KConv3x4;
