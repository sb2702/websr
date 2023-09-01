
export default class WebGPUContext {

    canvas: HTMLCanvasElement;
    device: GPUDevice;
    context: GPUCanvasContext;
    input: GPUTexture;

    constructor(device: GPUDevice, canvas: HTMLCanvasElement) {

        this.device = device;
        this.canvas = canvas;


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
        
        

    }




}