
import WebGPUContext from './context'


export default class WebSR {
    private canvas: HTMLCanvasElement;
    private context: WebGPUContext;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.context = new WebGPUContext(canvas);

    }

    async initWebGPU(): Promise<boolean>{
        return await this.context.load();
    }


}
