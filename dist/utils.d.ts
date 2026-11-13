export interface Resolution {
    width: number;
    height: number;
}
export type VideoSource = HTMLVideoElement | VideoFrame;
export type ImageSource = HTMLImageElement | ImageBitmap;
export type MediaSource = VideoSource | ImageSource;
export declare function isHTMLVideoElement(source: any): source is HTMLVideoElement;
export declare function isHTMLImageElement(source: any): source is HTMLImageElement;
export declare function isImageBitmap(source: any): source is ImageBitmap;
export declare function isVideoFrame(source: any): source is VideoFrame;
/**
 * Check if we're running on the main thread (not in a worker)
 */
export declare function isMainThread(): boolean;
/**
 * Check if a source is any type of video source
 */
export declare function isVideoSource(source: any): source is VideoSource;
/**
 * Check if a source is any type of image source
 */
export declare function isImageSource(source: any): source is ImageSource;
export declare function getSourceWidth(source: MediaSource): number;
export interface GPUTier {
    tier: number;
    isMobile: boolean;
    type: string;
    fps: number;
    gpu: string;
}
export declare function getGPUTier(tier?: GPUTier): number;
export declare function getSourceHeight(source: MediaSource): number;
export declare function externalTexturesSupported(device: GPUDevice, source: VideoSource): boolean;
