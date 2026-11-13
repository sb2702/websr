import WebGPUContext, { Precision } from './context';
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import { NetworkName, DisplayScale } from "./networks/network_list";
import { Resolution, MediaSource, GPUTier } from "./utils";
import { parseBinaryWeights, parseAnime4KWeights, parseX4VWeights, parseUNetWeights } from "./weights";
export type { Precision } from './context';
export type { Resolution, MediaSource, GPUTier } from './utils';
export type { NetworkName } from './networks/network_list';
export type { BinaryWeights, Anime4KWeights, UNetWeights } from './weights';
export interface WebSRParams {
    canvas: HTMLCanvasElement | OffscreenCanvas;
    weights: any;
    network_name: NetworkName;
    gpu: GPUDevice;
    debug?: boolean;
    resolution?: Resolution;
    outputResolution?: Resolution;
    inputScale?: number;
    tier?: GPUTier;
    precision?: Precision;
}
declare global {
    var context: WebGPUContext;
}
export default class WebSR {
    canvas: HTMLCanvasElement | OffscreenCanvas;
    context?: WebGPUContext;
    network?: NeuralNetwork;
    renderer?: WebSRRenderer;
    processResolution?: Resolution;
    canvasResolution?: Resolution;
    nominalTileResolution?: Resolution;
    debug?: boolean;
    scale: DisplayScale;
    tier: number;
    private params;
    private initialized;
    private thumbs?;
    private alphaCache;
    private pendingCalibration?;
    constructor(params: WebSRParams);
    get resolution(): Resolution | undefined;
    private networkInputResolution;
    private setup;
    private initialize;
    switchNetwork(network: NetworkName, weights: any): void;
    private planTiles;
    calibrate(frames: (VideoFrame | ImageBitmap)[]): Promise<Float32Array | null>;
    get calibrationThumbnails(): Float32Array | undefined;
    /** Change inputScale (see WebSRParams); the tile plan is redone on the next render(). */
    setInputScale(inputScale?: number): void;
    /** Change the output (canvas) resolution; undefined restores the default (input × scale). */
    setOutputResolution(outputResolution?: Resolution): void;
    static parseBinaryWeights: typeof parseBinaryWeights;
    static parseAnime4KWeights: typeof parseAnime4KWeights;
    static parseX4VWeights: typeof parseX4VWeights;
    static parseUNetWeights: typeof parseUNetWeights;
    static initWebGPU(): Promise<GPUDevice | false>;
    private static computeMaxTilePixels;
    private static computeOptimalGrid;
    render(source: MediaSource, onTileProgress?: (done: number, total: number) => void): Promise<void>;
    destroy(): Promise<void>;
}
