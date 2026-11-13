import WebGPUContext from "../context";
import Layer from "../layers/base_layer";
import RenderLayer from "../layers/base_render_layer";


class NeuralNetwork {
    context: WebGPUContext;
    layers: Layer[];
    weights: any;

    constructor(weights?: any) {
        this.weights = weights;
        this.context = globalThis.context;
        this.layers = this.model();
    }


    model(): Layer[]{
        return [];
    }

    lastLayer(): Layer {
        return this.layers[this.layers.length-1];
    }

    // Point the layers that read the source texture at `input` (each [layerIndex, inputIndex]
    // slot; negative layer indices count from the end), direct the final render layer at
    // `output`, and (re)build the pipelines that depend on the input type: a GPUExternalTexture
    // (video, zero-copy) needs a different shader than a GPUTexture (images, tiles).
    protected bindInput(input: GPUTexture | GPUExternalTexture, output: GPUTexture, slots: [number, number][]) {
        this.context.input = input;

        const resolve = (i: number) => i < 0 ? this.layers.length + i : i;

        for (const [layer, slot] of slots) this.layers[resolve(layer)].inputs[slot] = input;

        const last = this.lastLayer();
        if (last instanceof RenderLayer) last.setOutput(output);

        for (const layer of new Set(slots.map(([l]) => resolve(l)))) this.layers[layer].lazyLoadSetup();
    }

    // Run the network on `input` (the source, or one padded tile of it, at the context's
    // processResolution) and write the upscaled result to `output`. The renderer owns input
    // handling (external textures, image copies, tiling, inputScale resampling).
    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture){
        this.bindInput(input, output, []);
        this.layers.forEach(layer => {
            layer.run();
        });
    }
}


export default NeuralNetwork
