export interface Resolution {
    width: number,
    height: number
}

// Video source type - compatible with both main thread and worker contexts
export type VideoSource = HTMLVideoElement | VideoFrame;

// Image source type - compatible with both main thread and worker contexts
export type ImageSource = HTMLImageElement | ImageBitmap;

// All possible source types
export type MediaSource = VideoSource | ImageSource;

export function isHTMLVideoElement(source: any): source is HTMLVideoElement {
    return typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement;
}

export function isHTMLImageElement(source: any): source is HTMLImageElement {
    return typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement;
}

export function isImageBitmap(source: any): source is ImageBitmap {
    return typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap;
}

export function isVideoFrame(source: any): source is VideoFrame {
    return typeof VideoFrame !== 'undefined' && source instanceof VideoFrame;
}

/**
 * Check if we're running on the main thread (not in a worker)
 */
export function isMainThread(): boolean {
    return typeof HTMLVideoElement !== 'undefined';
}

/**
 * Check if a source is any type of video source
 */
export function isVideoSource(source: any): source is VideoSource {
    return isHTMLVideoElement(source) || isVideoFrame(source);
}

/**
 * Check if a source is any type of image source
 */
export function isImageSource(source: any): source is ImageSource {
    return isHTMLImageElement(source) || isImageBitmap(source);
}

export function getSourceWidth(source: MediaSource): number {
    if (isHTMLVideoElement(source)) return source.videoWidth;
    if (isHTMLImageElement(source)) return source.naturalWidth;
    if (isVideoFrame(source)) return source.displayWidth;
    if (isImageBitmap(source)) return source.width;
    return 0;
}

export interface GPUTier {
    tier: number,
    isMobile: boolean,
    type: string,
    fps: number,
    gpu: string
}


export function getGPUTier(tier?: GPUTier): number {

    if(!tier) return 1;

    let gpuTier = tier.tier;

    if(!gpuTier) return 1;

    if(tier.isMobile){
        gpuTier = Math.max(gpuTier -1, 1);
    }

    return gpuTier;

}

export function getSourceHeight(source: MediaSource): number {
    if (isHTMLVideoElement(source)) return source.videoHeight;
    if (isHTMLImageElement(source)) return source.naturalHeight;
    if (isVideoFrame(source)) return source.displayHeight;
    if (isImageBitmap(source)) return source.height;
    return 0;
}

// Whether device.importExternalTexture works. Firefox ships WebGPU with external
// textures behind a pref (NotSupportedError otherwise), so probe once per device,
// with a real source, and fall back to ImageBitmap copies when it doesn't.
const externalTextureSupport = new WeakMap<GPUDevice, boolean>();

export function externalTexturesSupported(device: GPUDevice, source: VideoSource): boolean {
    let ok = externalTextureSupport.get(device);
    if (ok === undefined) {
        try {
            device.importExternalTexture({ source });
            ok = true;
        } catch (e) {
            ok = false;
        }
        externalTextureSupport.set(device, ok);
    }
    return ok;
}
