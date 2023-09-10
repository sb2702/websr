

interface TextureOptions {
    width?: number,
    height?: number,
    format?: GPUTextureFormat
}

export default class WebGPUContext {

    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    textures: Record<string, GPUTexture>;
    destinationContext: GPUCanvasContext;

    private debug: boolean;
    private usage: number;


    constructor(device: GPUDevice, workingCanvas: HTMLCanvasElement, destinationCanvas: HTMLCanvasElement) {

        this.device = device;
        this.canvas = workingCanvas;
        this.textures = {};


        this.context = this.canvas.getContext('webgpu');

        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat()
        });


        this.destinationContext = destinationCanvas.getContext('webgpu');
        this.destinationContext.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat()
        })

        
        this.debug = true;

        this.usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
        if(this.debug) this.usage  = this.usage |  GPUTextureUsage.COPY_SRC;
        

        const inputTexture = device.createTexture({
            label: 'Input Image',
            size: [workingCanvas.width, workingCanvas.height],
            format: 'rgba8unorm',
            usage: this.usage
        });
        


        this.textures['input'] = inputTexture;

        this.textures['output'] = this.destinationContext.getCurrentTexture();


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


    texture(key:string, options?: TextureOptions): GPUTexture {

        if(!this.textures[key]){

            options = options || {};

            this.textures[key] = this.device.createTexture({
                label: key,
                size: [options.width || this.context.canvas.width, options.height || this.context.canvas.height],
                format: options.format || 'rgba32float',
                usage: this.usage
            });
        }

        return this.textures[key];

    }



}