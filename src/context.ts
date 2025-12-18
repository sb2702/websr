import {Resolution} from "./utils";
import {DisplayScale} from "./networks/network_list";


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
    resolution: Resolution;
    input: GPUTexture | GPUExternalTexture;
    destroyed: boolean;
    debug?: boolean;
    textureUsage: number;
    bufferUsage: number;
    scale: DisplayScale;


    constructor(device: GPUDevice, resolution: Resolution, canvas: HTMLCanvasElement , scale: DisplayScale, debug?: boolean) {

        this.device = device;
        this.canvas = canvas;
        this.resolution = resolution;
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

        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat()
        });


        this.textureUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
        this.bufferUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

        if(this.debug) {
            // Read output pixel value
            this.textureUsage  = this.textureUsage |  GPUTextureUsage.COPY_SRC;
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

        return  new Float32Array(0);
    }

    destroy(){
        this.device.destroy();
        this.destroyed = true;
    }


    buffer(key: string, options?: BufferOptions): GPUBuffer{

        if(!this.buffers[key]){

            options = options || {};

            const width = options.width || this.resolution.width;
            const height = options.height || this.resolution.height;
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
                size: [options.width || this.resolution.width, options.height || this.resolution.height],
                format: options.format || 'rgba32float',
                usage: this.textureUsage
            });
        }

        return this.textures[key];

    }



}