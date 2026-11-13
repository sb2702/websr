import Layer from "../../layers/base_layer";
import RenderLayer from "../../layers/base_render_layer";
import NeuralNetwork from "../base_network";
import RealESRGANIngest from "../../layers/realesrgan/ingest";
import UNetConv from "../../layers/unet/conv";
import UNetOutput from "../../layers/unet/output";
import {foldShader, runGate} from "../../layers/unet/calibrate";
import {UNetWeights, UNetLayerSpec, isUNetWeights} from "../../weights";
import {Resolution} from "../../utils";

// Conditional CReLU UNet (websr/gan-s, websr/gan-m: UNets distilled from Real-ESRGAN x4v3), 2x,
// built entirely from the graph in the weight file (see parseUNetWeights):
//
//   ingest (texture -> "in" buffer) -> convs in file order -> UNetOutput (PixelShuffle + Catmull-Rom)
//
// Buffers live at their UNet level: level l is ceil-halved l times from the process resolution
// (what a stride-2, pad-1 conv produces). Banked (bottleneck) convs keep all banks in an f32
// buffer and a live kernel/bias buffer the conv reads; applyAlpha() folds into the live buffers
// (one compute pass per tensor). Until calibrate() runs, the training-average alpha is folded.

// Edge-safe ingest: the base ComputeLayer dispatch rounds DOWN and would leave the last
// (width % 16) columns of the input buffer unwritten; the shader already bounds-checks.
class UNetIngest extends RealESRGANIngest {
    run() {
        if (!this.pipeline) this.lazyLoadSetup();
        const encoder = this.device.createCommandEncoder({label: this.label});
        const pass = encoder.beginComputePass({label: this.label});
        pass.setPipeline(this.pipeline);
        if (this.hasExternalTexture()) this.bindGroup = this.defaultBindGroup();
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.resolution.width / 16), Math.ceil(this.resolution.height / 16));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}

interface FoldOp {
    banks: GPUBuffer;        // f32, banks x len vec4s
    out: GPUBuffer;          // live tensor read by the conv (f32 or f16)
    params: GPUBuffer;
    bindGroup: GPUBindGroup;
    len: number;             // vec4s per bank
}

export function levelResolution(r: Resolution, level: number): Resolution {
    let {width, height} = r;
    for (let i = 0; i < level; i++) { width = Math.ceil(width / 2); height = Math.ceil(height / 2); }
    return {width, height};
}

export default class UNetCNN extends NeuralNetwork {
    // Assigned inside model(), which the base constructor calls — so no field initialisers here.
    private folds!: FoldOp[];
    private alphaBuf!: GPUBuffer;
    private foldPipeline!: GPUComputePipeline;
    weightsUNet!: UNetWeights;
    alpha!: Float32Array;

    constructor(weights: UNetWeights) {
        super(weights);
    }

    model(): Layer[] {
        if (!isUNetWeights(this.weights)) throw new Error("UNetCNN needs weights from WebSR.parseUNetWeights()");
        const w = this.weights as UNetWeights;
        const meta = w.meta;
        const ctx = this.context;
        const device = ctx.device;
        const useF16 = ctx.useF16();
        const bd = useF16 ? 2 : 4;
        this.weightsUNet = w;
        this.folds = [];

        // Activation buffers ("unet_<name>", so a debug page can readBuffer them).
        const bufs: Record<string, GPUBuffer> = {};
        for (const name of Object.keys(meta.buffers)) {
            const b = meta.buffers[name];
            const r = levelResolution(ctx.processResolution, b.level);
            bufs[name] = ctx.buffer(`unet_${name}`, {width: r.width, height: r.height, channels: b.vec4s * 4, bitdepth: bd});
        }

        // Weight buffers are registered in ctx.buffers too, so context.cleanup() frees them.
        const upload = (key: string, data: Float32Array, f16: boolean, extra = 0) => {
            const arr = f16 ? new Float16Array(data) : data;
            const buf = device.createBuffer({label: key, size: arr.byteLength,
                                             usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra});
            device.queue.writeBuffer(buf, 0, arr);
            ctx.buffers[key] = buf;
            return buf;
        };

        this.foldPipeline = device.createComputePipeline({
            label: "UNetFold", layout: "auto",
            compute: {module: device.createShaderModule({code: foldShader(useF16)}), entryPoint: "main"},
        });
        this.alphaBuf = device.createBuffer({label: "unet-alpha", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
        ctx.buffers["unetw_alpha"] = this.alphaBuf;

        const layers: Layer[] = [];
        layers.push(new UNetIngest(ctx.input, bufs["in"], useF16));

        meta.layers.forEach((spec: UNetLayerSpec) => {
            const K = w.tensor(spec.weights), B = w.tensor(spec.bias);
            let kern: GPUBuffer, bias: GPUBuffer;
            if (spec.banked) {
                kern = this.bankedTensor(`unetw_${spec.name}_k`, K, meta.banks, useF16, upload);
                bias = this.bankedTensor(`unetw_${spec.name}_b`, B, meta.banks, useF16, upload);
            } else {
                kern = upload(`unetw_${spec.name}_k`, K, useF16);
                bias = upload(`unetw_${spec.name}_b`, B, useF16);
            }
            const ib = meta.buffers[spec.in_buf], ob = meta.buffers[spec.out_buf];
            layers.push(new UNetConv(spec, bufs[spec.in_buf], bufs[spec.out_buf], kern, bias, {
                out: levelResolution(ctx.processResolution, spec.level),
                in: levelResolution(ctx.processResolution, spec.in_level),
                inStride: ib.vec4s, outStride: ob.vec4s,
            }, useF16));
        });

        layers.push(new UNetOutput([bufs["out"], ctx.input], ctx.texture("output"), useF16));

        this.applyAlpha(new Float32Array(meta.default_alpha));
        return layers;
    }

    // All banks in one f32 buffer + a live buffer (conv precision) the fold writes into.
    private bankedTensor(key: string, data: Float32Array, banks: number, useF16: boolean,
                         upload: (k: string, d: Float32Array, f16: boolean, extra?: number) => GPUBuffer): GPUBuffer {
        const device = this.context.device;
        const len = data.length / banks / 4;                 // vec4s per bank
        const banksBuf = upload(`${key}_banks`, data, false);
        const live = upload(key, data.subarray(0, len * 4), useF16);   // overwritten by the first fold
        const params = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
        device.queue.writeBuffer(params, 0, new Uint32Array([len, banks, 0, 0]));
        this.context.buffers[`${key}_params`] = params;
        this.folds.push({
            banks: banksBuf, out: live, params, len,
            bindGroup: device.createBindGroup({
                layout: this.foldPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: banksBuf}},
                    {binding: 1, resource: {buffer: live}},
                    {binding: 2, resource: {buffer: params}},
                    {binding: 3, resource: {buffer: this.alphaBuf}},
                ],
            }),
        });
        return live;
    }

    get banks(): number { return this.weightsUNet.meta.banks; }

    // Fold alpha into every banked conv. Queue-ordered: frames submitted afterwards use it.
    applyAlpha(alpha: Float32Array) {
        if (alpha.length !== this.banks) throw new Error(`applyAlpha: expected ${this.banks} values, got ${alpha.length}`);
        const device = this.context.device;
        const a = new Float32Array(8);
        a.set(alpha);
        device.queue.writeBuffer(this.alphaBuf, 0, a);
        const enc = device.createCommandEncoder({label: "UNetFold"});
        const pass = enc.beginComputePass();
        pass.setPipeline(this.foldPipeline);
        for (const f of this.folds) {
            pass.setBindGroup(0, f.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(f.len / 64));
        }
        pass.end();
        device.queue.submit([enc.finish()]);
        this.alpha = Float32Array.from(alpha);
    }

    // thumbnails (from calibrate.computeThumbnails) -> gate -> alpha -> fold. Returns alpha.
    async calibrate(thumbs: Float32Array): Promise<Float32Array> {
        const alpha = await runGate(this.context.device, this.weightsUNet, thumbs);
        this.applyAlpha(alpha);
        return alpha;
    }

    async feedForward(input: GPUTexture | GPUExternalTexture, output: GPUTexture) {
        this.context.input = input;
        const ingest = this.layers[0];
        const last = this.layers[this.layers.length - 1];
        ingest.inputs[0] = input;
        last.inputs[1] = input;
        ingest.lazyLoadSetup();
        if (last instanceof RenderLayer) last.setOutput(output);
        last.lazyLoadSetup();
        this.layers.forEach((layer) => layer.run());
    }
}
