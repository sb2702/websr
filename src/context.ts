import {Resolution} from "./utils";
import {DisplayScale} from "./networks/network_list";

// Compute precision for networks that support f16 (see WebGPUContext.useF16).
export type Precision = 'auto' | 'f32' | 'f16';


interface TextureOptions {
    width?: number,
    height?: number,
    format?: GPUTextureFormat
}


interface BufferOptions{
    channels?: number,
    bitdepth?: number,
    width?: number,
    height?: number
}


export default class WebGPUContext {

    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    textures: Record<string, GPUTexture>;
    buffers: Record<string, GPUBuffer>;
    processResolution: Resolution;
    canvasResolution: Resolution;
    outputResolution: Resolution;
    input: GPUTexture | GPUExternalTexture;
    destroyed: boolean;
    debug?: boolean;
    textureUsage: number;
    bufferUsage: number;
    scale: DisplayScale;
    precision: Precision;

    // Resolution each network pass runs at (one tile, including padding, when tiling).
    // Kept as `resolution` for backwards compatibility with code written against <= 0.0.16.
    get resolution(): Resolution {
        return this.processResolution;
    }


    constructor(device: GPUDevice, processResolution: Resolution, canvasResolution: Resolution, canvas: HTMLCanvasElement, scale: DisplayScale, debug?: boolean, outputResolution?: Resolution, precision?: Precision) {

        this.device = device;
        this.precision = precision ?? 'auto';
        this.canvas = canvas;
        this.processResolution = processResolution;
        this.canvasResolution = canvasResolution;
        this.outputResolution = outputResolution ?? { width: canvasResolution.width * scale, height: canvasResolution.height * scale };



        this.textures = {};
        this.buffers = {};
        this.destroyed = false;
        this.scale = scale;
        this.debug = debug;


        let context = this.canvas.getContext('webgpu');

        if(context instanceof GPUCanvasContext){
            this.context = context;
        } else {
            throw  new Error("Unable to load WebGPU context");
        }



        this.textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT |  GPUTextureUsage.COPY_SRC;
        this.bufferUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;


        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        });



        if(this.debug) {
            // Read output pixel value

            this.bufferUsage = this.bufferUsage | GPUBufferUsage.COPY_SRC
        }
        


        this.textures['output'] = this.context.getCurrentTexture();


    }



    async readBuffer(bufferName: string): Promise< Uint8ClampedArray | Float32Array> {

        if(!this.buffers[bufferName]) throw new Error(`No buffer with name ${bufferName}`);

        const readEncoder = this.device.createCommandEncoder({
            label: `Read ${bufferName} buffer encoder`,
        });

        const buffer = this.buffers[bufferName];

        const resultBuffer = this.device.createBuffer({
            label: 'result buffer',
            size:  buffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });


        readEncoder.copyBufferToBuffer(buffer, 0, resultBuffer, 0, resultBuffer.size);

        this.device.queue.submit([readEncoder.finish()]);

        await resultBuffer.mapAsync(GPUMapMode.READ);

        let range = resultBuffer.getMappedRange();

        return  new Float32Array(range);

    }



    async readTexture(textureName: string): Promise< Uint8ClampedArray | Float32Array> {


        if(!this.textures[textureName]) throw new Error(`No texture with name ${textureName}`);


        const readEncoder = this.device.createCommandEncoder({
            label: `Read ${textureName} texture encoder`,
        });


        const texture = this.textures[textureName];

        let bitsPerPixel = 16;

        if(texture.format === 'rgba8unorm') bitsPerPixel = 4;
        if(texture.format === 'bgra8unorm') bitsPerPixel = 4;
        if(texture.format === 'r32float') bitsPerPixel = 4;


        const resultBuffer = this.device.createBuffer({
            label: 'result buffer',
            size: texture.width*texture.height*bitsPerPixel,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });



        readEncoder.copyTextureToBuffer(
            {
                texture: this.textures[textureName],
            },
            {
                buffer: resultBuffer,
                bytesPerRow: texture.width*bitsPerPixel
            },
            {
                width: texture.width,
                height: texture.height,
                depthOrArrayLayers: 1,
            }
        );

        this.device.queue.submit([readEncoder.finish()]);

        await resultBuffer.mapAsync(GPUMapMode.READ);

        if(texture.format === 'r32float') return  new Float32Array(resultBuffer.getMappedRange());
        else if(texture.format === 'rgba32float') return  new Float32Array(resultBuffer.getMappedRange());
        else if(texture.format === 'rgba8unorm') return new Uint8ClampedArray(resultBuffer.getMappedRange());
        else if(texture.format === 'bgra8unorm') return new Uint8ClampedArray(resultBuffer.getMappedRange());

        return  new Float32Array(0);
    }

    cleanup(){
        for(const key of Object.keys(this.buffers)){
            this.buffers[key].destroy();
            delete this.buffers[key]
        }

        for(const key of Object.keys(this.textures)){
            this.textures[key].destroy();
            delete this.textures[key]
        }

        
    }

    destroy(){
        this.device.destroy();
        this.destroyed = true;
    }


    // Resolve whether compute should run in f16. 'auto' picks f16 when the device
    // exposes shader-f16 (requested in WebSR.initWebGPU) and the JS runtime has the
    // native Float16Array used to pack f16 weights. Forcing 'f16' without support throws.
    useF16(): boolean {
        const hasF16 = !!this.device.features?.has('shader-f16') && typeof Float16Array !== 'undefined';
        if (this.precision === 'f16') {
            if (!hasF16) throw new Error("precision 'f16' requested but device lacks shader-f16 (or runtime lacks Float16Array)");
            return true;
        }
        if (this.precision === 'f32') return false;
        return hasF16;
    }

    buffer(key: string, options?: BufferOptions): GPUBuffer{

        if(!this.buffers[key]){

            options = options || {};

            const width = options.width || this.processResolution.width;
            const height = options.height || this.processResolution.height;
            const channels = options.channels || 4;
            const bitdepth = options.bitdepth || 4;



            this.buffers[key] = this.device.createBuffer({
                label: key,
                size: width*height*channels*bitdepth,
                usage: this.bufferUsage
            });


        }

        return this.buffers[key];

    }

    texture(key:string, options?: TextureOptions): GPUTexture {

        if(!this.textures[key]){

            options = options || {};

            this.textures[key] = this.device.createTexture({
                label: key,
                size: [options.width || this.processResolution.width, options.height || this.processResolution.height],
                format: options.format || 'rgba32float',
                usage: this.textureUsage
            });
        }

        return this.textures[key];

    }



}