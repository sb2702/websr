
export default class WebGPUContext {

    private canvas: HTMLCanvasElement;
    public device: GPUDevice;
    private context: GPUCanvasContext;

    constructor(device: GPUDevice, canvas: HTMLCanvasElement) {

        this.device = device;
        this.canvas = canvas;


        this.context = this.canvas.getContext('webgpu');

        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat()
        });

    }




}