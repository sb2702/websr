import WebGPUContext, {Precision} from './context'
import WebSRRenderer from "./renderer";
import NeuralNetwork from "./networks/base_network";
import {NetworkList, NetworkName, NetworkScales, NetworkBufferCounts, NetworkBytesPerPixel, NetworkTilePadding, DisplayScale,
        NetworkMaxBufferBytesPerPixel, NetworkTileAlign} from "./networks/network_list";
import {Resolution, MediaSource, getSourceWidth, getSourceHeight, GPUTier, getGPUTier} from "./utils";

export type {Precision} from './context';
export type {Resolution, MediaSource, GPUTier} from './utils';
export type {NetworkName} from './networks/network_list';


export interface WebSRParams {
    canvas: HTMLCanvasElement | OffscreenCanvas,
    weights: any,
    network_name: NetworkName,
    gpu: GPUDevice,
    debug?: boolean;
    // Source resolution, to set everything up in the constructor instead of on the first
    // render(). Optional: render() detects it from the source either way.
    resolution?: Resolution,
    // Final canvas size. Defaults to the (resampled) source size × the network scale; any
    // other size is reached with a GPU bicubic resample of the network output.
    outputResolution?: Resolution,
    // Resample the source by 1/inputScale (on the GPU, antialiased bicubic) before the
    // network, e.g. 2 runs a 1080p source through the network at 960x540. The output
    // defaults to the resampled size × the network scale; pass outputResolution to keep
    // the nominal output size.
    inputScale?: number,
    // GPU capability tier, used to size the tile memory budget for large sources.
    // Defaults to tier 1 (~100 MB of network buffers per tile).
    tier?: GPUTier,
    // Compute precision for networks with an f16 path. 'auto' (default) uses f16 when the
    // device has shader-f16 (requested by WebSR.initWebGPU) and the runtime has Float16Array.
    precision?: Precision
}


declare global {
    var context: WebGPUContext;
}


export default class WebSR {
    canvas: HTMLCanvasElement | OffscreenCanvas;
    context?: WebGPUContext;
    network?: NeuralNetwork;
    renderer?: WebSRRenderer;
    // Resolution each network pass runs at (one padded tile when tiling).
    processResolution?: Resolution;
    // Network input resolution: the source resolution divided by inputScale.
    canvasResolution?: Resolution;
    nominalTileResolution?: Resolution;
    debug?: boolean;
    scale: DisplayScale;
    tier: number;
    private params: WebSRParams;
    private initialized: boolean = false;


    constructor(params: WebSRParams) {

        if(!NetworkList[params.network_name]) throw Error(`Network ${params.network_name} is not defined or implemented`);

        this.params = params;
        this.canvas = params.canvas;
        this.scale = NetworkScales[params.network_name];
        this.debug = params.debug;

        this.tier = getGPUTier(params.tier);

        // If resolution is provided, initialize immediately
        if(params.resolution) {
            this.setup(this.networkInputResolution(params.resolution.width, params.resolution.height));
        }
    }

    // Network input resolution (before tiling). Same as the source resolution unless inputScale is set.
    get resolution(): Resolution | undefined {
        return this.canvasResolution;
    }

    private networkInputResolution(width: number, height: number): Resolution {
        const inputScale = this.params.inputScale && this.params.inputScale > 0 ? this.params.inputScale : 1;
        return {
            width: Math.max(1, Math.round(width / inputScale)),
            height: Math.max(1, Math.round(height / inputScale))
        };
    }

    // (Re)build everything for a network input resolution: tile plan, context, network, renderer.
    private setup(canvasResolution: Resolution) {
        if(this.context) {
            this.context.cleanup();
        }
        this.canvasResolution = canvasResolution;
        this.planTiles(canvasResolution.width, canvasResolution.height);
        this.initialize();
    }

    private initialize() {
        if(!this.processResolution || !this.canvasResolution) {
            throw new Error("Cannot initialize without resolution");
        }

        const outputResolution = this.params.outputResolution ?? {
            width: this.canvasResolution.width * this.scale,
            height: this.canvasResolution.height * this.scale,
        };

        this.canvas.width = outputResolution.width;
        this.canvas.height = outputResolution.height;

        this.context = new WebGPUContext(this.params.gpu, this.processResolution, this.canvasResolution, this.canvas as HTMLCanvasElement, this.scale, this.debug, outputResolution, this.params.precision);
        globalThis.context = this.context;

        this.network = new NetworkList[this.params.network_name](this.params.weights);
        this.renderer = new WebSRRenderer(this.network, this.nominalTileResolution, NetworkTilePadding[this.params.network_name]);

        this.initialized = true;
    }


    // Switch network (and weights). Networks differ in scale and buffer count, so the tile
    // plan and context are rebuilt. Safe to call before the first render().
    switchNetwork(network: NetworkName, weights: any){
        if(!NetworkList[network]) throw Error(`Network ${network} is not defined or implemented`);

        this.params = { ...this.params, network_name: network, weights };
        this.scale = NetworkScales[network];

        if(this.canvasResolution) {
            this.setup(this.canvasResolution);
        }
    }

    // Tile grid for a W x H source: nominal tile (stitched region) + processResolution (tile with
    // padding, what the network runs at). Networks with NetworkTileAlign get tiles on that grid;
    // networks with NetworkMaxBufferBytesPerPixel get padded tiles under the per-buffer binding limit.
    private planTiles(W: number, H: number) {
        const name = this.params.network_name;
        const pad = NetworkTilePadding[name];
        const align = NetworkTileAlign[name] ?? 1;
        const maxBuf = NetworkMaxBufferBytesPerPixel[name];
        const bindLimit = this.params.gpu.limits.maxStorageBufferBindingSize;
        let budget = WebSR.computeMaxTilePixels(name, this.tier, bindLimit);
        for (let attempt = 0; ; attempt++) {
            let nominal = WebSR.computeOptimalGrid(W, H, budget);
            const needsTiling = nominal.width < W || nominal.height < H;
            if (needsTiling && align > 1) {
                nominal = {width: Math.ceil(nominal.width / align) * align, height: Math.ceil(nominal.height / align) * align};
            }
            const process = needsTiling ? {width: nominal.width + 2 * pad, height: nominal.height + 2 * pad} : nominal;
            const fits = !maxBuf || process.width * process.height * maxBuf <= bindLimit;
            if (fits || attempt >= 12) {
                this.nominalTileResolution = nominal;
                this.processResolution = process;
                return;
            }
            budget = Math.floor(budget * 0.85);
        }
    }

    /** Change inputScale (see WebSRParams); the tile plan is redone on the next render(). */
    setInputScale(inputScale?: number){
        this.params = { ...this.params, inputScale };
        this.initialized = false;
    }

    /** Change the output (canvas) resolution; undefined restores the default (input × scale). */
    setOutputResolution(outputResolution?: Resolution){
        this.params = { ...this.params, outputResolution };

        if(this.canvasResolution) {
            this.setup(this.canvasResolution);
        }
    }


    static async initWebGPU(): Promise<GPUDevice | false>{

        if(!navigator.gpu) return false;

        const adapter = await navigator.gpu.requestAdapter();

        if(!adapter) return false;

        // Request shader-f16 when available so networks with an f16 path (e.g. realesrgan/x4v3)
        // can use it. Harmless on devices that already have it; omitted otherwise.
        const requiredFeatures: GPUFeatureName[] = adapter.features.has('shader-f16') ? ['shader-f16'] : [];

        const device = await adapter.requestDevice({ requiredFeatures });

        if(!device) return false;

        return device;
    }

    // Compute max pixels a single tile can hold under two constraints:
    //   1) total memory budget (tier-dependent): sum of all buffers must fit
    //   2) per-buffer binding limit: no single buffer may exceed device.limits.maxStorageBufferBindingSize
    private static computeMaxTilePixels(networkName: NetworkName, tier = 1, deviceMaxStorageBufferBindingSize = 128 * 1024 * 1024): number {
        const bufferCount = NetworkBufferCounts[networkName];
        const bytesPerPixel = NetworkBytesPerPixel[networkName];

        let budgetBytes = 100e6;

        if(tier == 3){
            budgetBytes = 1000e6
        } else if (tier == 2){
            budgetBytes = 250e6
        } else if (tier == 1){
            budgetBytes = 100e6;
        } else{
            budgetBytes = 50e6;
        }

        let pixels = Math.floor(budgetBytes / (bufferCount * bytesPerPixel));
        const maxBuf = NetworkMaxBufferBytesPerPixel[networkName];
        if (maxBuf) pixels = Math.min(pixels, Math.floor(deviceMaxStorageBufferBindingSize / maxBuf));
        return pixels;
    }

    // Find tile dimensions that minimise the number of tiles (cols × rows) while keeping
    // each tile within maxTilePixels. Divides the source evenly so edge tiles are never
    // wasted — e.g. 1920×1080 with a 428k-pixel budget yields 5 tiles of 1920×216
    // rather than 12 tiles of 512×512 with a largely-empty bottom row.
    // Returns the full source dimensions unchanged when no tiling is needed.
    private static computeOptimalGrid(W: number, H: number, maxTilePixels: number): Resolution {
        if (W * H <= maxTilePixels) {
            return { width: W, height: H };
        }

        let bestCols = W, bestRows = H;

        for (let cols = 1; cols <= W; cols++) {
            const tileW = Math.ceil(W / cols);
            const maxTileH = Math.floor(maxTilePixels / tileW);
            if (maxTileH <= 0) break;
            const rows = Math.ceil(H / maxTileH);
            const tileH = Math.ceil(H / rows);

            if (tileW * tileH <= maxTilePixels && cols * rows < bestCols * bestRows) {
                bestCols = cols;
                bestRows = rows;
            }

            if (rows === 1) break;
        }

        return {
            width: Math.ceil(W / bestCols),
            height: Math.ceil(H / bestRows),
        };
    }


    // Upscale one frame/image to the canvas. Re-plans automatically when the source
    // resolution changes. onTileProgress (optional) fires as each tile completes on the GPU.
    async render(source: MediaSource, onTileProgress?: (done: number, total: number) => void){
        if(!source) {
            throw new Error("render() requires a source parameter");
        }

        const canvasResolution = this.networkInputResolution(getSourceWidth(source), getSourceHeight(source));

        if(!this.initialized || !this.canvasResolution ||
           canvasResolution.width !== this.canvasResolution.width ||
           canvasResolution.height !== this.canvasResolution.height) {
            if(this.initialized) {
                console.log(`Resolution changed from ${this.canvasResolution?.width}x${this.canvasResolution?.height} to ${canvasResolution.width}x${canvasResolution.height}`);
            }
            this.setup(canvasResolution);
        }

        await this.renderer!.render(source, onTileProgress);
    }

    async destroy(){
        if(this.context) {
            this.context.destroy();
        }
    }
}
