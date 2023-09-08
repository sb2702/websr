
export default class WebGPUContext {

    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    input: GPUTexture;
    output: GPUTexture;
    textures: Record<string, GPUTexture>;
    destinationContext: GPUCanvasContext;

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



        const inputTexture = device.createTexture({
            label: 'Input Image',
            size: [workingCanvas.width, workingCanvas.height],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        


        this.input = inputTexture;

        this.output = this.destinationContext.getCurrentTexture();
        
        

    }


    texture(key:string): GPUTexture {

        if(!this.textures[key]){

            this.textures[key] = this.device.createTexture({
                label: key,
                size: [this.context.canvas.width, this.context.canvas.height],
                format: 'rgba32float',
                usage:
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }

        return this.textures[key];

    }



}