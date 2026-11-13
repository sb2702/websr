export default class InputResampleLayer {
    private context;
    private device;
    private pipelines;
    private params;
    constructor();
    private pipeline;
    /** Resample `source` (srcW × srcH) into an rgba8unorm texture of outW × outH and return it. */
    run(source: VideoFrame | HTMLVideoElement | ImageBitmap, srcW: number, srcH: number, outW: number, outH: number): GPUTexture;
}
