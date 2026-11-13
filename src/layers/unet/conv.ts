import ComputeLayer from "../base_compute_layer";
import {UNetLayerSpec} from "../../weights";
import {Resolution} from "../../utils";

// One 3x3 conv of the CReLU UNet. Covers every conv in the graph:
//   mode "same" : output (x,y) reads input (x+dx, y+dy)                 same resolution
//   mode "s2"   : output (x,y) reads input (2x+dx, 2y+dy)                stride 2 (down)
//   mode "up"   : nearest 2x upsample fused into the read: tap (x+dx) on the OUTPUT grid maps to
//                 input floor(u * inW / outW) — size-exact like PyTorch's interpolate(size=...),
//                 so odd sizes align. No upsampled buffer exists.
// Taps outside the image read zero (PyTorch zero padding; for "up" the bound is the upsampled grid).
//
// CReLU lives in the read (the websr/Anime4K convention): the conv sees cat(relu(v), relu(-v)),
// stored as 2x input vec4s — weight block iv = g for relu(v), iv = g + in_vec4s for relu(-v).
// No activation on the output: the next conv's read applies it.
//
// Buffers are vec4-strided: element (x, y, group g) lives at (y*W + x)*stride + offset + g, so a
// conv can read or write a slice of a wider buffer. That is the skip concat: the skip and the up
// conv write the two halves of one buffer and the fuse conv reads all of it — no copy pass.
//
// Weights: mat4x4 per (tap, iv, ov), column-major (scripts/pth_to_bin.py convention), indexed
// kernels[(tap*IV + iv)*out_vec4s + ov]. Kernel structure follows realesrgan/conv2d_prelu.ts:
// one pixel per thread, output vec4s in blocks of 4 so each input vec4 is loaded once per tap
// and reused by 4 accumulators. f32 and f16 variants differ only in element types.

export interface UNetConvGeometry {
    out: Resolution;
    in: Resolution;
    inStride: number;     // vec4s per pixel of the input buffer
    outStride: number;    // vec4s per pixel of the output buffer
}

const MODE = {same: 0, s2: 1, up: 2};

function convShader(f16: boolean): string {
    const v4 = f16 ? "vec4<f16>" : "vec4f";
    const m4 = f16 ? "mat4x4<f16>" : "mat4x4f";
    const zero = f16 ? "vec4<f16>(0.0h)" : "vec4f(0.0)";
    return `${f16 ? "enable f16;\n" : ""}
struct Params {
    out_w: u32, out_h: u32, in_w: u32, in_h: u32,
    in_stride: u32, in_offset: u32, in_vec4s: u32,
    out_stride: u32, out_offset: u32, out_vec4s: u32,
    crelu: u32, mode: u32,
}

@group(0) @binding(0) var<storage, read>       input: array<${v4}>;
@group(0) @binding(1) var<storage, read_write> output: array<${v4}>;
@group(0) @binding(2) var<storage, read>       kernels: array<${m4}>;
@group(0) @binding(3) var<storage, read>       bias: array<${v4}>;
@group(0) @binding(4) var<uniform>             p: Params;

// Input pixel index for tap (dx, dy) of output (ox, oy), or -1 when the tap reads padding.
fn src_index(ox: i32, oy: i32, dx: i32, dy: i32) -> i32 {
    var sx: i32; var sy: i32;
    if (p.mode == 1u) {                         // stride 2
        sx = ox * 2 + dx; sy = oy * 2 + dy;
        if (sx < 0 || sy < 0 || sx >= i32(p.in_w) || sy >= i32(p.in_h)) { return -1; }
    } else if (p.mode == 2u) {                  // fused nearest 2x upsample
        let ux = ox + dx; let uy = oy + dy;
        if (ux < 0 || uy < 0 || ux >= i32(p.out_w) || uy >= i32(p.out_h)) { return -1; }
        sx = i32((u32(ux) * p.in_w) / p.out_w);
        sy = i32((u32(uy) * p.in_h) / p.out_h);
    } else {
        sx = ox + dx; sy = oy + dy;
        if (sx < 0 || sy < 0 || sx >= i32(p.in_w) || sy >= i32(p.in_h)) { return -1; }
    }
    return sy * i32(p.in_w) + sx;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if (ox >= p.out_w || oy >= p.out_h) { return; }

    let IV = select(p.in_vec4s, p.in_vec4s * 2u, p.crelu != 0u);    // weight blocks per tap
    let OV = p.out_vec4s;

    for (var ob = 0u; ob < OV; ob = ob + 4u) {
        let h1 = (ob + 1u) < OV;
        let h2 = (ob + 2u) < OV;
        let h3 = (ob + 3u) < OV;
        let o0 = ob;
        let o1 = select(ob, ob + 1u, h1);       // clamp tail (no OOB weight read)
        let o2 = select(ob, ob + 2u, h2);
        let o3 = select(ob, ob + 3u, h3);
        var a0 = ${zero}; var a1 = ${zero}; var a2 = ${zero}; var a3 = ${zero};

        for (var ky = 0; ky < 3; ky = ky + 1) {
            for (var kx = 0; kx < 3; kx = kx + 1) {
                let s = src_index(i32(ox), i32(oy), kx - 1, ky - 1);
                if (s < 0) { continue; }
                let tap = u32(ky * 3 + kx);
                let px = u32(s) * p.in_stride + p.in_offset;
                for (var g = 0u; g < p.in_vec4s; g = g + 1u) {
                    let v = input[px + g];
                    if (p.crelu != 0u) {
                        let vp = max(v, ${zero});
                        let vn = max(-v, ${zero});
                        let bp = (tap * IV + g) * OV;
                        let bn = (tap * IV + g + p.in_vec4s) * OV;
                        a0 = a0 + kernels[bp + o0] * vp + kernels[bn + o0] * vn;
                        a1 = a1 + kernels[bp + o1] * vp + kernels[bn + o1] * vn;
                        a2 = a2 + kernels[bp + o2] * vp + kernels[bn + o2] * vn;
                        a3 = a3 + kernels[bp + o3] * vp + kernels[bn + o3] * vn;
                    } else {
                        let b = (tap * IV + g) * OV;
                        a0 = a0 + kernels[b + o0] * v;
                        a1 = a1 + kernels[b + o1] * v;
                        a2 = a2 + kernels[b + o2] * v;
                        a3 = a3 + kernels[b + o3] * v;
                    }
                }
            }
        }

        let base = (oy * p.out_w + ox) * p.out_stride + p.out_offset;
        output[base + o0] = a0 + bias[o0];
        if (h1) { output[base + o1] = a1 + bias[o1]; }
        if (h2) { output[base + o2] = a2 + bias[o2]; }
        if (h3) { output[base + o3] = a3 + bias[o3]; }
    }
}
`;
}

// One pipeline per precision, shared by every conv of every UNet on a device.
const pipelineCache = new WeakMap<GPUDevice, Map<boolean, GPUComputePipeline>>();

function getPipeline(device: GPUDevice, f16: boolean): GPUComputePipeline {
    let m = pipelineCache.get(device);
    if (!m) { m = new Map(); pipelineCache.set(device, m); }
    let pl = m.get(f16);
    if (!pl) {
        pl = device.createComputePipeline({
            label: `UNetConv-${f16 ? "f16" : "f32"}`,
            layout: "auto",
            compute: {module: device.createShaderModule({code: convShader(f16)}), entryPoint: "main"},
        });
        m.set(f16, pl);
    }
    return pl;
}

class UNetConv extends ComputeLayer {
    label = "UNetConv";
    private geom: UNetConvGeometry;

    // kernels / bias buffers are owned by the network (banked convs get them rewritten by the fold).
    constructor(spec: UNetLayerSpec, input: GPUBuffer, output: GPUBuffer,
                kernels: GPUBuffer, bias: GPUBuffer, geom: UNetConvGeometry, useF16: boolean) {
        super([input], output);
        this.label = `UNetConv-${spec.name}`;
        this.geom = geom;
        this.pipeline = getPipeline(this.device, useF16);

        const params = new Uint32Array([
            geom.out.width, geom.out.height, geom.in.width, geom.in.height,
            geom.inStride, spec.in_offset, spec.in_vec4s,
            geom.outStride, spec.out_offset, spec.out_vec4s,
            spec.crelu ? 1 : 0, MODE[spec.mode],
        ]);
        const paramsBuf = this.device.createBuffer({
            label: `${this.label}/params`, size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(paramsBuf, 0, params);

        this.bindGroup = this.device.createBindGroup({
            label: `${this.label}-bg`,
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: input}},
                {binding: 1, resource: {buffer: output}},
                {binding: 2, resource: {buffer: kernels}},
                {binding: 3, resource: {buffer: bias}},
                {binding: 4, resource: {buffer: paramsBuf}},
            ],
        });
    }

    run() {
        const encoder = this.device.createCommandEncoder({label: this.label});
        const pass = encoder.beginComputePass({label: this.label});
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        // Round UP: every output pixel must be computed (the shader bounds-checks). The base
        // ComputeLayer rounds down, which would leave edge pixels unwritten — and in a UNet those
        // feed the coarser levels.
        pass.dispatchWorkgroups(Math.ceil(this.geom.out.width / 16), Math.ceil(this.geom.out.height / 16));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}

export default UNetConv;
