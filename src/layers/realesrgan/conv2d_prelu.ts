import ComputeLayer from "../base_compute_layer";

// Fused Conv 3x3 + per-channel PReLU.
// Thick storage buffers: input/output are `array<vec4>` with N vec4 per pixel
// (N = (channels + 3) / 4). Weights/bias/PReLU alphas are also storage buffers
// because 64×64 conv weights blow the 64 KB uniform binding limit (~147 KB).
//
// Kernel: output-channel-blocked ("cblock4"). One pixel per thread (16×16 wg),
// no shared memory. The output-channel loop runs in BLOCKS OF 4: each input vec4
// is loaded ONCE per 3×3 tap and reused across 4 output-group accumulators (4× fewer
// input loads than a 1-acc kernel). Benchmarked ~2× faster than the prior shared-memory
// variant on target hardware — the barriers/LDS traffic there cost more than they saved.
// f32 and f16 variants are identical except for element types (f16 also halves buffer mem).

function cblock4Shader(f16: boolean): string {
    const v4   = f16 ? "vec4<f16>" : "vec4f";
    const m4   = f16 ? "mat4x4<f16>" : "mat4x4f";
    const zero = f16 ? "vec4<f16>(0.0)" : "vec4f(0.0)";
    const enable = f16 ? "enable f16;\n" : "";

    return `${enable}
struct Params {
    width: u32,
    height: u32,
    in_vec4s: u32,
    out_vec4s: u32,
    apply_prelu: u32,
}

@group(0) @binding(0) var<storage, read>       input: array<${v4}>;
@group(0) @binding(1) var<storage, read_write> output: array<${v4}>;
@group(0) @binding(2) var<storage, read>       kernels: array<${m4}>;
@group(0) @binding(3) var<storage, read>       bias: array<${v4}>;
@group(0) @binding(4) var<storage, read>       prelu_alpha: array<${v4}>;
@group(0) @binding(5) var<uniform>             params: Params;

fn act(v: ${v4}, ov: u32) -> ${v4} {
    var r = v + bias[ov];
    if (params.apply_prelu != 0u) {
        let s = prelu_alpha[ov];
        r = max(r, ${zero}) + s * min(r, ${zero});
    }
    return r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if (ox >= params.width || oy >= params.height) {
        return;
    }
    let IO = params.in_vec4s * params.out_vec4s;

    for (var ob = 0u; ob < params.out_vec4s; ob = ob + 4u) {
        let h1 = (ob + 1u) < params.out_vec4s;
        let h2 = (ob + 2u) < params.out_vec4s;
        let h3 = (ob + 3u) < params.out_vec4s;
        let o0 = ob;
        let o1 = select(ob, ob + 1u, h1);   // clamp tail (no OOB weight read)
        let o2 = select(ob, ob + 2u, h2);
        let o3 = select(ob, ob + 3u, h3);

        var a0 = ${zero};
        var a1 = ${zero};
        var a2 = ${zero};
        var a3 = ${zero};

        for (var in_vec = 0u; in_vec < params.in_vec4s; in_vec = in_vec + 1u) {
            for (var ky = 0u; ky < 3u; ky = ky + 1u) {
                for (var kx = 0u; kx < 3u; kx = kx + 1u) {
                    let gx = i32(ox) + i32(kx) - 1;
                    let gy = i32(oy) + i32(ky) - 1;
                    var v = ${zero};
                    if (gx >= 0 && gx < i32(params.width) &&
                        gy >= 0 && gy < i32(params.height)) {
                        v = input[(u32(gy) * params.width + u32(gx)) * params.in_vec4s + in_vec];
                    }
                    let base = (ky * 3u + kx) * IO + in_vec * params.out_vec4s;
                    a0 = a0 + kernels[base + o0] * v;   // v loaded once, reused ×4
                    a1 = a1 + kernels[base + o1] * v;
                    a2 = a2 + kernels[base + o2] * v;
                    a3 = a3 + kernels[base + o3] * v;
                }
            }
        }

        let rowBase = (oy * params.width + ox) * params.out_vec4s;
        output[rowBase + o0] = act(a0, o0);
        if (h1) { output[rowBase + o1] = act(a1, o1); }
        if (h2) { output[rowBase + o2] = act(a2, o2); }
        if (h3) { output[rowBase + o3] = act(a3, o3); }
    }
}
`;
}


function toFloat32Array(src: any): Float32Array {
    if (src instanceof Float32Array) return src;
    if (Array.isArray(src) && src.length > 0 && Array.isArray(src[0])) {
        return new Float32Array(([] as number[]).concat.apply([], src as number[][]));
    }
    return new Float32Array(src);
}

// Pack weights for upload. In f16 mode the values are narrowed to Float16Array so the
// storage buffer matches the shader's vec4<f16>/mat4x4<f16> bindings (half the bytes).
function packWeights(f32: Float32Array, useF16: boolean): Float32Array | Float16Array {
    return useF16 ? new Float16Array(f32) : f32;
}


class RealESRGANConv2DPReLU extends ComputeLayer {
    label = "RealESRGANConv2DPReLU";
    num_work_groups = 16;

    private kernelsBuf: GPUBuffer;
    private biasBuf: GPUBuffer;
    private alphaBuf: GPUBuffer;
    private paramsBuf: GPUBuffer;
    private inputBuf: GPUBuffer;
    private outBuf: GPUBuffer;
    private inVec4s: number;
    private outVec4s: number;
    private applyPReLU: boolean;

    constructor(inputBuf: GPUBuffer, outputBuf: GPUBuffer, layerWeights: any, useF16: boolean = false) {
        super([inputBuf], outputBuf, layerWeights);
        this.inputBuf = inputBuf;
        this.outBuf = outputBuf;
        this.inVec4s = layerWeights.in_vec4s;
        this.outVec4s = layerWeights.out_vec4s;
        this.applyPReLU = layerWeights.type === "conv_prelu";

        // matrices: array<mat4x4> = 9 * in_vec4s * out_vec4s, 16 floats each.
        // Accept either Float32Array (binary loader) or nested number[][] (JSON loader).
        // Narrowed to Float16Array in f16 mode so byteLength halves and matches the binding.
        const matricesData = packWeights(toFloat32Array(layerWeights.matrices), useF16);
        this.kernelsBuf = this.device.createBuffer({
            label: `${layerWeights.name}/kernels`,
            size: matricesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.kernelsBuf, 0, matricesData);

        // bias: vec4 per out_vec4 (already padded to out_vec4s * 4).
        const biasData = packWeights(toFloat32Array(layerWeights.bias), useF16);
        this.biasBuf = this.device.createBuffer({
            label: `${layerWeights.name}/bias`,
            size: biasData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.biasBuf, 0, biasData);

        // PReLU alpha (always allocated; zeros when apply_prelu=0 so the binding stays valid).
        const alphaData = packWeights(
            this.applyPReLU ? toFloat32Array(layerWeights.alpha) : new Float32Array(this.outVec4s * 4),
            useF16,
        );
        this.alphaBuf = this.device.createBuffer({
            label: `${layerWeights.name}/alpha`,
            size: alphaData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.alphaBuf, 0, alphaData);

        const params = new Uint32Array([
            this.resolution.width, this.resolution.height,
            this.inVec4s, this.outVec4s,
            this.applyPReLU ? 1 : 0,
        ]);
        this.paramsBuf = this.device.createBuffer({
            label: `${layerWeights.name}/params`,
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.paramsBuf, 0, params);

        this.shader = this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: cblock4Shader(useF16),
        });
        this.pipeline = this.device.createComputePipeline({
            label: `${this.label}-pipeline`,
            layout: "auto",
            compute: { module: this.shader, entryPoint: "main" },
        });

        this.bindGroup = this.device.createBindGroup({
            label: `${this.label}-bg`,
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.inputBuf } },
                { binding: 1, resource: { buffer: this.outBuf } },
                { binding: 2, resource: { buffer: this.kernelsBuf } },
                { binding: 3, resource: { buffer: this.biasBuf } },
                { binding: 4, resource: { buffer: this.alphaBuf } },
                { binding: 5, resource: { buffer: this.paramsBuf } },
            ],
        });
    }
}

export default RealESRGANConv2DPReLU;
