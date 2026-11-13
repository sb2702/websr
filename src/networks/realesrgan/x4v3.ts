import Layer from "../../layers/base_layer";
import RenderLayer from "../../layers/base_render_layer";
import NeuralNetwork from "../base_network";
import RealESRGANConv2DPReLU from "../../layers/realesrgan/conv2d_prelu";
import RealESRGANIngest from "../../layers/realesrgan/ingest";
import RealESRGANOutputX4 from "../../layers/realesrgan/output_x4";

// RealESRGAN x4v3 (SRVGGCompact) — 64ch, 32 inner Conv+PReLU pairs, ×4 PixelShuffle, nearest residual.
// Weights (X4V binary from scripts/pth_to_bin.py, parsed by WebSR.parseBinaryWeights) have 34 entries:
//   34 conv layers — first 33 are "conv_prelu" (fused conv+PReLU), the last is plain conv (no act).
// Ping-pong scheme:
//   ingest: input texture → input_buf (1 vec4/px)
//   conv 0:  input_buf → act_a
//   conv 1:  act_a → act_b
//   conv 2:  act_b → act_a
//   ...alternating...
//   conv 32: act_b → act_a            (body.64+65 — last conv_prelu, last fused layer)
//   conv 33: act_a → final_buf        (body.66 — plain conv, 48ch)
//   output:  final_buf + input texture → canvas (4× res)

export default class RealESRGANx4v3 extends NeuralNetwork {

    constructor(weights: any) {
        super(weights);
    }

    model(): Layer[] {
        const ctx = this.context;
        const layers: Layer[] = [];

        // f16 conv kernels read/write vec4<f16> buffers, so the whole ping-pong chain
        // (ingest → convs → output) switches precision together. bitdepth halves to 2.
        const useF16 = ctx.useF16();
        const bd = useF16 ? 2 : 4;

        const inputBuf = ctx.buffer("x4v3_input",  { channels: 4,  bitdepth: bd });   // 1 vec4/px (RGB+0)
        const actA     = ctx.buffer("x4v3_act_a",  { channels: 64, bitdepth: bd });   // 16 vec4/px
        const actB     = ctx.buffer("x4v3_act_b",  { channels: 64, bitdepth: bd });   // 16 vec4/px
        const finalBuf = ctx.buffer("x4v3_final",  { channels: 48, bitdepth: bd });   // 12 vec4/px (matches body.66 out_vec4s=12)

        const layerWeights: any[] = this.weights.layers;

        // ingest: input texture → input_buf. layers[0].inputs[0] gets patched per-frame.
        layers.push(new RealESRGANIngest(ctx.input, inputBuf, useF16));

        // Body: 33 fused conv_prelu, alternating between act_a and act_b.
        for (let i = 0; i < layerWeights.length - 1; i++) {
            const lw = layerWeights[i];
            const inBuf  = (i === 0) ? inputBuf
                         : (i % 2 === 1) ? actA
                         : actB;
            const outBuf = (i % 2 === 0) ? actA : actB;
            layers.push(new RealESRGANConv2DPReLU(inBuf, outBuf, lw, useF16));
        }

        // Final plain conv. The fused-layer index i=N-2 (= 32 for N=34) is even → its
        // output is actA. The final layer reads actA, writes final_buf.
        const finalIdx = layerWeights.length - 1;
        const finalLw = layerWeights[finalIdx];
        const finalIn = (finalIdx - 1) % 2 === 0 ? actA : actB;
        layers.push(new RealESRGANConv2DPReLU(finalIn, finalBuf, finalLw, useF16));

        // Output: PixelShuffle ×4 + nearest residual + add → canvas.
        // inputs: [final_buf, input_texture]. Input texture is patched per frame.
        layers.push(new RealESRGANOutputX4([finalBuf, ctx.input], ctx.texture("output"), useF16));

        return layers;
    }

    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture) {
        this.context.input = input;

        // Patch the texture inputs for this frame.
        const ingest = this.layers[0];
        const last = this.layers[this.layers.length - 1];
        ingest.inputs[0] = input;
        last.inputs[1] = input;

        // External-texture-aware lazy setup for the layers that touch the input texture.
        ingest.lazyLoadSetup();
        if (last instanceof RenderLayer) last.setOutput(output);
        last.lazyLoadSetup();

        this.layers.forEach((layer) => layer.run());
    }
}
