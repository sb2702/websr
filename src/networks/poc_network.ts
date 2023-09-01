import WebGPUContext from "../context";
import Layer from "../layers/base_layer";
import NeuralNetwork from "./base_network";
import RGB2YUV from "../layers/rgb_2_yuv";
import GuassianLayer from "../layers/gaussian";


export default class PoCNetwork extends NeuralNetwork{

    constructor(context: WebGPUContext) {
        super(context);
    }


    model(): Layer[]{

        const layers:Layer[] = [];

        const texture1 = this.context.device.createTexture({
            label: 'Input Image',
            size: [this.context.canvas.width, this.context.canvas.height],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        layers.push(new RGB2YUV(this.context.device, this.context.input, texture1));

        layers.push(new GuassianLayer(this.context.device, texture1, this.context.output));

        return layers;


    }


}

