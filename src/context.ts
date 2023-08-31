
export default class WebGPUContext {

    private canvas: HTMLCanvasElement;
    private adapter: GPUAdapter | null;
    public device: GPUDevice;
    private context: GPUCanvasContext;

    constructor(canvas: HTMLCanvasElement) {

        this.canvas = canvas;

    }


    async load(): Promise<boolean> {

        if(!navigator.gpu) return false;

        this.adapter = await navigator.gpu.requestAdapter();

        if(!this.adapter) return false;

        this.device = await this.adapter.requestDevice();

        if(!this.device) return false;

        this.context = this.canvas.getContext('webgpu');

        this.context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat()
        });

        return true;

    }






}