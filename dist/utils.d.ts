export interface Resolution {
    width: number;
    height: number;
}
export type VideoSource = HTMLVideoElement | VideoFrame;
export type ImageSource = HTMLImageElement | ImageBitmap;
export type MediaSource = VideoSource | ImageSource;
/**
 * Check if we're running on the main thread (not in a worker)
 */
export declare function isMainThread(): boolean;
/**
 * Check if a source is a video element (main thread only)
 */
export declare function isHTMLVideoElement(source: any): source is HTMLVideoElement;
/**
 * Check if a source is an image element (main thread only)
 */
export declare function isHTMLImageElement(source: any): source is HTMLImageElement;
/**
 * Check if a source is an ImageBitmap (works in both contexts)
 */
export declare function isImageBitmap(source: any): source is ImageBitmap;
/**
 * Check if a source is a VideoFrame (worker context)
 */
export declare function isVideoFrame(source: any): source is VideoFrame;
/**
 * Check if a source is any type of video source
 */
export declare function isVideoSource(source: any): source is VideoSource;
/**
 * Check if a source is any type of image source
 */
export declare function isImageSource(source: any): source is ImageSource;
/**
 * Get the width of a media source
 */
export declare function getSourceWidth(source: MediaSource): number;
/**
 * Get the height of a media source
 */
export declare function getSourceHeight(source: MediaSource): number;
