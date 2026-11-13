import WebGPUContext from "../context";

// Resizes the source to the network's input size (WebSRParams.inputScale) on the
// GPU, before tiling and the network.
//
// Separable Catmull-Rom (bicubic, a = -0.5) with the kernel stretched by the
// downscale factor, i.e. antialiased: each output pixel averages the ~4×scale
// source pixels it covers per axis instead of point-sampling 4 (which aliases at
// 2× and beyond). Upscaling degrades to plain bicubic.
//
// Two render passes: horizontal (source → rgba16float, outW × srcH), then vertical
// (→ rgba8unorm, outW × outH). VideoFrames / video elements are read as external
// textures (zero-copy, same colour conversion as the renderer's own path);
// ImageBitmaps are copied in first. textureLoad throughout — no sampler, no flips.

function shader(external: boolean): string {
    return /* wgsl */`
        struct Params { scale: f32, srcLen: i32, axis: u32, _pad: u32 };

        @group(0) @binding(0) var src: ${external ? 'texture_external' : 'texture_2d<f32>'};
        @group(0) @binding(1) var<uniform> p: Params;

        @vertex
        fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            // Fullscreen triangle.
            let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
            return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
        }

        fn catmullRom(x: f32) -> f32 {
            let t = abs(x);
            if (t >= 2.0) { return 0.0; }
            let t2 = t * t;
            let t3 = t2 * t;
            if (t <= 1.0) { return 1.5 * t3 - 2.5 * t2 + 1.0; }
            return -0.5 * t3 + 2.5 * t2 - 4.0 * t + 2.0;
        }

        fn load(c: vec2i) -> vec3f {
            return textureLoad(src, c${external ? '' : ', 0'}).rgb;
        }

        @fragment
        fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
            let o = vec2i(floor(pos.xy));
            let stretch = max(p.scale, 1.0);
            let along = select(o.x, o.y, p.axis == 1u);
            let center = (f32(along) + 0.5) * p.scale;
            let lo = i32(floor(center - 2.0 * stretch));
            let hi = i32(ceil(center + 2.0 * stretch));

            var acc = vec3f(0.0);
            var wsum = 0.0;
            for (var i = lo; i <= hi; i = i + 1) {
                let w = catmullRom((f32(i) + 0.5 - center) / stretch);
                if (w == 0.0) { continue; }
                let c = clamp(i, 0, p.srcLen - 1);
                acc += load(select(vec2i(c, o.y), vec2i(o.x, c), p.axis == 1u)) * w;
                wsum += w;
            }
            return vec4f(acc / wsum, 1.0);
        }
    `;
}

export default class InputResampleLayer {
    private context: WebGPUContext;
    private device: GPUDevice;
    private pipelines: Record<string, GPURenderPipeline> = {};
    private params: GPUBuffer[];

    constructor() {
        this.context = globalThis.context;
        this.device = this.context.device;
        // One uniform buffer per pass, so both can be written before a single submit.
        this.params = [0, 1].map(() => this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }));
    }

    private pipeline(external: boolean, format: GPUTextureFormat): GPURenderPipeline {
        const key = `${external}-${format}`;
        if (!this.pipelines[key]) {
            const module = this.device.createShaderModule({ label: 'input-resample', code: shader(external) });
            this.pipelines[key] = this.device.createRenderPipeline({
                label: `input-resample-${key}`,
                layout: 'auto',
                vertex: { module, entryPoint: 'vertexMain' },
                fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
                primitive: { topology: 'triangle-list' },
            });
        }
        return this.pipelines[key];
    }

    /** Resample `source` (srcW × srcH) into an rgba8unorm texture of outW × outH and return it. */
    run(source: VideoFrame | HTMLVideoElement | ImageBitmap, srcW: number, srcH: number, outW: number, outH: number): GPUTexture {
        const ctx = this.context;
        const device = this.device;

        let input: GPUExternalTexture | GPUTexture;
        const external = !(source instanceof ImageBitmap);
        if (external) {
            input = device.importExternalTexture({ source: source as VideoFrame | HTMLVideoElement });
        } else {
            input = ctx.texture(`resample_src_${srcW}x${srcH}`, { width: srcW, height: srcH, format: 'rgba8unorm' });
            device.queue.copyExternalImageToTexture({ source: source as ImageBitmap }, { texture: input }, [srcW, srcH]);
        }

        // Keyed by size: context.texture() caches by key, and the source size isn't fixed.
        const tmp = ctx.texture(`resample_tmp_${outW}x${srcH}`, { width: outW, height: srcH, format: 'rgba16float' });
        const out = ctx.texture(`resample_out_${outW}x${outH}`, { width: outW, height: outH, format: 'rgba8unorm' });

        const passes: [GPUExternalTexture | GPUTexture, GPUTexture, boolean, number, number, GPUTextureFormat][] = [
            [input, tmp, external, srcW, srcW / outW, 'rgba16float'],
            [tmp, out, false, srcH, srcH / outH, 'rgba8unorm'],
        ];

        const encoder = device.createCommandEncoder({ label: 'input-resample' });
        passes.forEach(([from, to, ext, srcLen, scale, format], axis) => {
            const data = new ArrayBuffer(16);
            new Float32Array(data, 0, 1)[0] = scale;
            new Int32Array(data, 4, 1)[0] = srcLen;
            new Uint32Array(data, 8, 1)[0] = axis;
            device.queue.writeBuffer(this.params[axis], 0, data);

            const pipeline = this.pipeline(ext, format);
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: ext ? from as GPUExternalTexture : (from as GPUTexture).createView() },
                    { binding: 1, resource: { buffer: this.params[axis] } },
                ],
            });
            const pass = encoder.beginRenderPass({
                colorAttachments: [{ view: to.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(3);
            pass.end();
        });
        device.queue.submit([encoder.finish()]);

        return out;
    }
}
