
export default class WebGPUContext {

    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    input: GPUTexture;
    output: GPUTexture;
    textures: {}

    constructor(device: GPUDevice, canvas: HTMLCanvasElement) {

        this.device = device;
        this.canvas = canvas;
        this.textures = {};


        this.context = this.canvas.getContext('webgpu');

        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat()
        });



        const inputTexture = device.createTexture({
            label: 'Input Image',
            size: [canvas.width, canvas.height],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        


        this.input = inputTexture;

        this.output = this.context.getCurrentTexture();
        
        

    }


    texture(key): GPUTexture {

        if(!this.textures[key]){

            this.textures[key] = this.device.createTexture({
                label: key,
                size: [this.context.canvas.width, this.context.canvas.height],
                format: 'rgba8unorm',
                usage:
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }

        return this.textures[key];

    }



}