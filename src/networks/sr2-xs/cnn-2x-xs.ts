import Layer            from "../../layers/base_layer";
import RenderLayer      from "../../layers/base_render_layer";
import NeuralNetwork    from "../base_network";
import SR2XSStem        from "../../layers/sr2-xs/stem";
import SR2XSDepthwise   from "../../layers/sr2-xs/depthwise";
import SR2XSPointwise   from "../../layers/sr2-xs/pointwise";
import SR2XSDisplay     from "../../layers/sr2-xs/display";

// SR2-XS ("sr2/cnn-2x-xs") — 2× depthwise-separable upscaler with 16-channel
// CReLU-aware trunk and bilinear residual base. Default ~1.4k params at
// num_blocks=1.
//
//   RGBA texture  → SR2XSStem (3×3, 4→16, no CReLU input)
//                 → [block_dw (3×3 depthwise CReLU, 16→16)
//                    → block_pw (1×1 CReLU, 16→16)] × num_blocks
//                 → [projection] (1×1, optional, only if trunk width != 16)
//                 → SR2XSDisplay (depth-to-space + bilinear + add → canvas)
//
// num_blocks is read from the weights JSON dynamically.
// Intermediate buffers are 16-channel (4 vec4 per LR pixel).
export default class SR2XSCNN extends NeuralNetwork {

    constructor(weights: any) {
        super(weights);
    }

    model(): Layer[] {
        const layers: Layer[] = [];
        const w = this.weights.layers;
        const ctx = this.context;
        const buf = (name: string) => ctx.buffer(name, { channels: 16 });

        // Stem (4-ch texture → 16-ch buffer). inputs[0] is patched per-frame.
        layers.push(new SR2XSStem([ctx.input], buf('stem'), w['stem']));

        // CReLU blocks. Detect count by scanning weight keys.
        let cursor = 'stem';
        for (let i = 0; w[`block${i}_dw`]; i++) {
            const dw_name = `block${i}_dw`;
            const pw_name = `block${i}_pw`;
            layers.push(new SR2XSDepthwise([buf(cursor)],  buf(dw_name), w[dw_name]));
            layers.push(new SR2XSPointwise([buf(dw_name)], buf(pw_name), w[pw_name]));
            cursor = pw_name;
        }

        // Optional projection (only present when trunk width != 16).
        if (w['projection']) {
            layers.push(new SR2XSPointwise([buf(cursor)], buf('projection'), w['projection']));
            cursor = 'projection';
        }

        // Display (fragment shader fusing depth_to_space + bilinear + add → canvas).
        // inputs: [residual buffer, source texture]. Source texture is patched per frame.
        layers.push(new SR2XSDisplay(
            [buf(cursor), ctx.input],
            ctx.texture('output'),
        ));

        return layers;
    }

    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture) {
        this.context.input = input;

        // Patch the texture inputs for this frame.
        const stem = this.layers[0];
        const last = this.layers[this.layers.length - 1];
        stem.inputs[0] = input;
        last.inputs[1] = input;

        // External-texture-aware lazy setup for the layers that touch the input texture.
        stem.lazyLoadSetup();
        if (last instanceof RenderLayer) last.setOutput(output);
        last.lazyLoadSetup();

        this.layers.forEach((layer) => layer.run());
    }
}
