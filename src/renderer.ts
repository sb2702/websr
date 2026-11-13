import WebGPUContext from "./context";
import NeuralNetwork from "./networks/base_network";
import DownscaleLayer from "./layers/downscale";
import InputResampleLayer from "./layers/input_resample";
import {Resolution, MediaSource, isHTMLVideoElement, isVideoFrame, isImageBitmap, getSourceWidth, getSourceHeight, externalTexturesSupported} from "./utils";

interface TileDescriptor {
    // Padded extraction region from the full source (clamped to image bounds)
    srcX: number;
    srcY: number;
    srcW: number;
    srcH: number;
    // Offset within tileOutput where the nominal (non-padded) content starts
    cropX: number;
    cropY: number;
    // Destination in the stitch target (pre-multiplied by scale)
    dstX: number;
    dstY: number;
    dstW: number;
    dstH: number;
}

export default class WebSRRenderer{
    private context: WebGPUContext;
    private network: NeuralNetwork;
    private tiles: TileDescriptor[];
    private nominalTileResolution: Resolution;
    private tilePadding: number;
    private downscaleLayer?: DownscaleLayer;
    private resampleLayer?: InputResampleLayer;

    source?: MediaSource;
    active: boolean;
    vfc: number;


    constructor(network: NeuralNetwork, nominalTileResolution?: Resolution, tilePadding: number = 12, source?: MediaSource) {

        this.context = globalThis.context;
        this.network = network;
        this.source = source;
        this.active = false;
        this.nominalTileResolution = nominalTileResolution ?? this.context.processResolution;
        this.tilePadding = tilePadding;
        this.tiles = this.calculateTileGrid();


        if (this.needsDownscale()) {
            const scale = this.context.scale;
            const intermediateW = this.context.canvasResolution.width * scale;
            const intermediateH = this.context.canvasResolution.height * scale;
            const intermediate = this.context.texture('upscaled_intermediate', {
                width: intermediateW,
                height: intermediateH,
                format: navigator.gpu.getPreferredCanvasFormat()
            });
            const canvasTexture = this.context.context.getCurrentTexture();
            this.downscaleLayer = new DownscaleLayer(intermediate, canvasTexture);
        }

        console.log(`[WebGPU Renderer]: Configured with ${this.tiles.length} tiles`);

    }


    switchNetwork(network: NeuralNetwork){
        this.network = network;
    }

    // Render loop for an HTMLVideoElement source (kept from <= 0.0.16): renders every
    // new video frame via requestVideoFrameCallback until stop() is called.
    async start(){

        if(this.context.destroyed) {
            throw new Error("WebSR instance was destroyed");
        }

        this.active = true;
        await this.renderStep();
    }

    async stop(){
        this.active = false;
        if(this.vfc && this.source && isHTMLVideoElement(this.source)) this.source.cancelVideoFrameCallback(this.vfc);
    }

    async renderStep(){

        await this.render();

        if(this.active && this.source && isHTMLVideoElement(this.source)) {
            this.vfc = this.source.requestVideoFrameCallback(this.renderStep.bind(this));
        }
    }

    private needsDownscale(): boolean {
        const scale = this.context.scale;
        const outRes = this.context.outputResolution;
        return outRes.width !== this.context.canvasResolution.width * scale ||
               outRes.height !== this.context.canvasResolution.height * scale;
    }

    private isTiling(): boolean {
        return this.tiles.length > 1;
    }

    // Enumerate tile descriptors based on nominal tile size. When tiling is active each
    // tile's source region is expanded by TILE_PADDING on each side (clamped to image
    // bounds) so the network sees context beyond the seam. Only the nominal center is
    // stitched to the output, hiding the seam artifacts.
    private calculateTileGrid(): TileDescriptor[] {
        const tiles: TileDescriptor[] = [];
        const scale = this.context.scale;
        const nomTileW = this.nominalTileResolution.width;
        const nomTileH = this.nominalTileResolution.height;
        const canvasW = this.context.canvasResolution.width;
        const canvasH = this.context.canvasResolution.height;

        for (let y = 0; y < canvasH; y += nomTileH) {
            for (let x = 0; x < canvasW; x += nomTileW) {
                const nomW = Math.min(nomTileW, canvasW - x);
                const nomH = Math.min(nomTileH, canvasH - y);

                const padLeft   = Math.min(this.tilePadding, x);
                const padTop    = Math.min(this.tilePadding, y);
                const padRight  = Math.min(this.tilePadding, canvasW - (x + nomW));
                const padBottom = Math.min(this.tilePadding, canvasH - (y + nomH));

                tiles.push({
                    srcX: x - padLeft,
                    srcY: y - padTop,
                    srcW: nomW + padLeft + padRight,
                    srcH: nomH + padTop + padBottom,
                    cropX: padLeft * scale,
                    cropY: padTop * scale,
                    dstX: x * scale,
                    dstY: y * scale,
                    dstW: nomW * scale,
                    dstH: nomH * scale,
                });
            }
        }
        return tiles;
    }

    private async materializeInput(src: MediaSource): Promise<GPUTexture> {
        const w = getSourceWidth(src);
        const h = getSourceHeight(src);

        const targetKey = 'full_input';
        // VideoFrames / video elements go through importExternalTexture (the resample
        // pass at 1:1, an exact copy): Firefox's copyExternalImageToTexture rejects
        // VideoFrame sources.
        if (isVideoFrame(src) || isHTMLVideoElement(src)) {
            return this.resampler().run(src, w, h, w, h);
        }

        const texture = this.context.texture(targetKey, { width: w, height: h, format: 'rgba8unorm' });

        let source: GPUImageCopyExternalImageSource;
        if (isImageBitmap(src)) {
            source = src as GPUImageCopyExternalImageSource;
        } else {
            source = await createImageBitmap(src as ImageBitmapSource);
        }

        try {
            this.context.device.queue.copyExternalImageToTexture({ source }, { texture }, [w, h]);
        } catch(e) {
            console.log("[WebGPU Renderer]: Error copying texture");
            console.log(e);
        }


        return texture;
    }

    // Optional per-tile progress. Fires on real GPU completion of each tile (via
    // queue.onSubmittedWorkDone()), not on JS dispatch, so the count tracks the
    // GPU rather than racing ahead and stalling on the final flush. When no
    // callback is passed, onSubmittedWorkDone is never awaited and behaviour is
    // unchanged.
    async render(source?: MediaSource, onTileProgress?: (done: number, total: number) => void){

        let src = source ?? this.source;
        if(!src) throw new Error("render() requires a source");

        // Without external textures, video sources are copied in as ImageBitmaps.
        let bitmap: ImageBitmap | undefined;
        if ((isVideoFrame(src) || isHTMLVideoElement(src)) && !externalTexturesSupported(this.context.device, src)) {
            src = bitmap = await createImageBitmap(src);
        }

        try {
            // A source whose size differs from the planned input (inputScale) is
            // resampled on the GPU first, and the network reads that texture instead.
            const resampled = this.resampleIfNeeded(src);

            if (this.isTiling()) {
                await this.renderTiled(src, onTileProgress, resampled);
            } else {
                await this.renderSingle(src, onTileProgress, resampled);
            }

            if (this.downscaleLayer) {
                const canvasTexture = this.context.context.getCurrentTexture();
                this.downscaleLayer.setOutput(canvasTexture);
                this.downscaleLayer.run();
            }
        } finally {
            bitmap?.close();
        }

    }

    private resampleIfNeeded(src: MediaSource): GPUTexture | undefined {
        const w = getSourceWidth(src), h = getSourceHeight(src);
        const { width, height } = this.context.canvasResolution;
        if (w === width && h === height) return undefined;
        if (!(isHTMLVideoElement(src) || isVideoFrame(src) || isImageBitmap(src))) {
            throw new Error("inputScale needs a VideoFrame, HTMLVideoElement or ImageBitmap source");
        }
        return this.resampler().run(src as VideoFrame | HTMLVideoElement | ImageBitmap, w, h, width, height);
    }

    private resampler(): InputResampleLayer {
        if (!this.resampleLayer) this.resampleLayer = new InputResampleLayer();
        return this.resampleLayer;
    }

    private async renderSingle(src: MediaSource, onTileProgress?: (done: number, total: number) => void, resampled?: GPUTexture) {
        let input: GPUTexture | GPUExternalTexture;

        if (resampled) {
            input = resampled;
        } else if (isHTMLVideoElement(src) || isVideoFrame(src)) {
            input = this.context.device.importExternalTexture({source: src});
        } else {
            const bitmap = isImageBitmap(src) ? src : await createImageBitmap(src as ImageBitmapSource);
            const width = getSourceWidth(src);
            const height = getSourceHeight(src);
            this.context.device.queue.copyExternalImageToTexture({source: bitmap}, {texture: this.context.texture('input', {format: "rgba8unorm"})}, [width, height]);
            input = this.context.texture('input');
        }

        if (this.downscaleLayer) {
            const output = this.context.texture('upscaled_intermediate');
            await this.network.feedForward(input, output);
        } else {
            const output = this.context.context.getCurrentTexture();
            await this.network.feedForward(input, output);
        }

        if (onTileProgress) {
            await this.context.device.queue.onSubmittedWorkDone();
            onTileProgress(1, 1);
        }
    }



    private async renderTiled(src: MediaSource, onTileProgress?: (done: number, total: number) => void, resampled?: GPUTexture) {

        const fullInput = resampled ?? await this.materializeInput(src);

        const scale = this.context.scale;

        // Per-tile progress awaits GPU completion mid-loop, which crosses a task
        // boundary and expires the canvas's current texture. So when reporting
        // progress (and not already on the downscale path, which stitches to an
        // intermediate anyway), stitch to an offscreen texture and copy it to the
        // canvas once at the end, instead of stitching into the swapchain texture.
        const deferCanvasCopy = !this.downscaleLayer && !!onTileProgress;

        const stitchTarget = this.downscaleLayer
            ? this.context.texture('upscaled_intermediate')
            : deferCanvasCopy
                ? this.context.texture('stitched_output', {
                    width: this.context.canvasResolution.width * scale,
                    height: this.context.canvasResolution.height * scale,
                    format: navigator.gpu.getPreferredCanvasFormat(),
                })
                : this.context.context.getCurrentTexture();

        // tileInput/tileOutput are sized to processResolution, which includes the padding
        const tileInput = this.context.texture('input', { format: 'rgba8unorm' });
        const tileOutW = this.context.processResolution.width * scale;
        const tileOutH = this.context.processResolution.height * scale;
        const tileOutput = this.context.texture('tile_output', {
            width: tileOutW,
            height: tileOutH,
            format: navigator.gpu.getPreferredCanvasFormat()
        });

        for (let tileIndex = 0; tileIndex < this.tiles.length; tileIndex++) {
            const tile = this.tiles[tileIndex];
            // Clear tileInput to black then copy the padded region from the full source.
            // Clearing first ensures pixels outside the padded region (at image edges where
            // full TILE_PADDING isn't available) are zero rather than stale data.
            const prepEncoder = this.context.device.createCommandEncoder();
            const clearPass = prepEncoder.beginRenderPass({
                colorAttachments: [{
                    view: tileInput.createView(),
                    clearValue: [0, 0, 0, 0],
                    loadOp: 'clear' as const,
                    storeOp: 'store' as const,
                }]
            });
            clearPass.end();
            prepEncoder.copyTextureToTexture(
                { texture: fullInput, origin: { x: tile.srcX, y: tile.srcY } },
                { texture: tileInput },
                { width: tile.srcW, height: tile.srcH }
            );
            this.context.device.queue.submit([prepEncoder.finish()]);

            // Process padded tile through network
            await this.network.feedForward(tileInput, tileOutput);

            // Stitch only the nominal (non-padded) center of the output to the stitch target
            const stitchEncoder = this.context.device.createCommandEncoder();
            stitchEncoder.copyTextureToTexture(
                { texture: tileOutput, origin: { x: tile.cropX, y: tile.cropY } },
                { texture: stitchTarget, origin: { x: tile.dstX, y: tile.dstY } },
                { width: tile.dstW, height: tile.dstH }
            );
            this.context.device.queue.submit([stitchEncoder.finish()]);

            if (onTileProgress) {
                await this.context.device.queue.onSubmittedWorkDone();
                onTileProgress(tileIndex + 1, this.tiles.length);
            }
        }

        // Flush the fully stitched result to the canvas in one copy (see
        // deferCanvasCopy above): the swapchain texture is acquired now, after
        // all the mid-loop awaits, so it is current and valid.
        if (deferCanvasCopy) {
            const canvasTexture = this.context.context.getCurrentTexture();
            const copyEncoder = this.context.device.createCommandEncoder();
            copyEncoder.copyTextureToTexture(
                { texture: stitchTarget },
                { texture: canvasTexture },
                { width: this.context.canvasResolution.width * scale, height: this.context.canvasResolution.height * scale }
            );
            this.context.device.queue.submit([copyEncoder.finish()]);
        }

    }

    async destroy(){
        this.context.destroy();
    }
}
