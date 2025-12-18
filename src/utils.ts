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

/**
 * Check if we're running on the main thread (not in a worker)
 */
export function isMainThread(): boolean {
    return typeof HTMLVideoElement !== 'undefined';
}

/**
 * Check if a source is a video element (main thread only)
 */
export function isHTMLVideoElement(source: any): source is HTMLVideoElement {
    return typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement;
}

/**
 * Check if a source is an image element (main thread only)
 */
export function isHTMLImageElement(source: any): source is HTMLImageElement {
    return typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement;
}

/**
 * Check if a source is an ImageBitmap (works in both contexts)
 */
export function isImageBitmap(source: any): source is ImageBitmap {
    return typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap;
}

/**
 * Check if a source is a VideoFrame (worker context)
 */
export function isVideoFrame(source: any): source is VideoFrame {
    return typeof VideoFrame !== 'undefined' && source instanceof VideoFrame;
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

/**
 * Get the width of a media source
 */
export function getSourceWidth(source: MediaSource): number {
    if (isHTMLVideoElement(source)) return source.videoWidth;
    if (isHTMLImageElement(source)) return source.naturalWidth;
    if (isVideoFrame(source)) return source.displayWidth;
    if (isImageBitmap(source)) return source.width;
    return 0;
}

/**
 * Get the height of a media source
 */
export function getSourceHeight(source: MediaSource): number {
    if (isHTMLVideoElement(source)) return source.videoHeight;
    if (isHTMLImageElement(source)) return source.naturalHeight;
    if (isVideoFrame(source)) return source.displayHeight;
    if (isImageBitmap(source)) return source.height;
    return 0;
}