
export default class WebGPUContext {

    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    textures: Record<string, GPUTexture>;
    destinationContext: GPUCanvasContext;

    static  bitsPerPixelbyFormat = {
        'rgba8unorm': 4,
        'rgba32float': 16
    }
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

        this.usage = this.debug? GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT 
        

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

        const bitsPerPixel = WebGPUContext.bitsPerPixelbyFormat[texture.format];


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

        if(bitsPerPixel == 4) return  new Uint8ClampedArray(resultBuffer.getMappedRange());
        else if(bitsPerPixel == 16) return new Float32Array(resultBuffer.getMappedRange());

        return  new Float32Array(0);
    }


    texture(key:string): GPUTexture {

        if(!this.textures[key]){

            this.textures[key] = this.device.createTexture({
                label: key,
                size: [this.context.canvas.width, this.context.canvas.height],
                format: 'rgba32float',
                usage: this.usage
            });
        }

        return this.textures[key];

    }



}