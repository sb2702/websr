import {UNetWeights} from "../../weights";
import {externalTexturesSupported} from "../../utils";

// Per-video routing for the conditional UNets, all on WebGPU:
//
//   computeThumbnails  full frame -> 3x32x32 area-average thumbnail per frame (PyTorch
//                      adaptive_avg_pool2d bins: rows [floor(i*H/32), ceil((i+1)*H/32)) )
//   runGate            thumbnails -> gate (2 stride-2 3x3 convs + LeakyReLU 0.2, global mean,
//                      linear, softmax/temperature) -> alpha per thumbnail; averaged = the video's alpha
//   foldPipeline       alpha -> weight = sum_b alpha_b * bank_b, written straight into the live
//                      kernel/bias buffers of the banked convs (see networks/unet/unet.ts)
//
// Thumbnails are network-independent (both UNets' gates were trained on the same 32x32 pooling),
// so WebSR keeps them and re-runs the gate of whichever UNet becomes active.

export const THUMB = 32;
const THUMB_FLOATS = 3 * THUMB * THUMB;
export const MAX_THUMBNAILS = 64;

export function thumbShader(external: boolean): string {
    const texType = external ? "texture_external" : "texture_2d<f32>";
    const load = external ? "textureLoad(frame, vec2<u32>(x, y))" : "textureLoad(frame, vec2<u32>(x, y), 0)";
    return `
struct P { w: u32, h: u32, slot: u32, _pad: u32 }
@group(0) @binding(0) var frame: ${texType};
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read_write> thumbs: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;   // column bin
    let i = gid.y;   // row bin
    if (i >= ${THUMB}u || j >= ${THUMB}u) { return; }
    let y0 = (i * p.h) / ${THUMB}u;
    let y1 = ((i + 1u) * p.h + ${THUMB - 1}u) / ${THUMB}u;
    let x0 = (j * p.w) / ${THUMB}u;
    let x1 = ((j + 1u) * p.w + ${THUMB - 1}u) / ${THUMB}u;
    var acc = vec3f(0.0);
    for (var y = y0; y < y1; y = y + 1u) {
        for (var x = x0; x < x1; x = x + 1u) {
            acc = acc + ${load}.rgb;
        }
    }
    let m = acc / f32((y1 - y0) * (x1 - x0));
    let base = p.slot * ${THUMB_FLOATS}u + i * ${THUMB}u + j;      // CHW
    thumbs[base] = m.r;
    thumbs[base + ${THUMB * THUMB}u] = m.g;
    thumbs[base + ${2 * THUMB * THUMB}u] = m.b;
}`;
}

function frameSize(f: VideoFrame | ImageBitmap): [number, number] {
    if (typeof VideoFrame !== "undefined" && f instanceof VideoFrame) return [f.displayWidth, f.displayHeight];
    return [(f as ImageBitmap).width, (f as ImageBitmap).height];
}

async function readBack(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<Float32Array> {
    const dst = device.createBuffer({size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, dst, 0, bytes);
    device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(dst.getMappedRange().slice(0));
    dst.unmap();
    dst.destroy();
    return out;
}

// frames -> Float32Array(n * 3*32*32), CHW per frame. VideoFrames are read through
// importExternalTexture (the renderer's path, same colour conversion); ImageBitmaps are copied
// into an rgba8unorm texture (also the renderer's path).
export async function computeThumbnails(device: GPUDevice, frames: (VideoFrame | ImageBitmap)[]): Promise<Float32Array> {
    if (frames.length === 0) throw new Error("calibrate: no frames");
    if (frames.length > MAX_THUMBNAILS) frames = frames.slice(0, MAX_THUMBNAILS);
    const n = frames.length;
    const out = device.createBuffer({
        label: "unet-thumbs", size: n * THUMB_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const pipes: Record<string, GPUComputePipeline> = {};
    const pipe = (external: boolean) => pipes[String(external)] ??= device.createComputePipeline({
        label: `unet-thumb-${external ? "ext" : "tex"}`, layout: "auto",
        compute: {module: device.createShaderModule({code: thumbShader(external)}), entryPoint: "main"},
    });
    const temps: GPUBuffer[] = [];
    const tempTex: GPUTexture[] = [];
    for (let k = 0; k < n; k++) {
        const f = frames[k];
        const [w, h] = frameSize(f);
        let isVF = typeof VideoFrame !== "undefined" && f instanceof VideoFrame;
        // Without external textures (Firefox, by default) VideoFrames are copied in as bitmaps.
        const bitmap = isVF && !externalTexturesSupported(device, f as VideoFrame) ? await createImageBitmap(f) : null;
        if (bitmap) isVF = false;
        let resource: GPUExternalTexture | GPUTextureView;
        if (isVF) {
            resource = device.importExternalTexture({source: f as VideoFrame});
        } else {
            const tex = device.createTexture({
                size: [w, h], format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });
            device.queue.copyExternalImageToTexture({source: bitmap ?? f as ImageBitmap}, {texture: tex}, [w, h]);
            bitmap?.close();
            tempTex.push(tex);
            resource = tex.createView();
        }
        const u = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
        device.queue.writeBuffer(u, 0, new Uint32Array([w, h, k, 0]));
        temps.push(u);
        const pl = pipe(isVF);
        const enc = device.createCommandEncoder({label: "unet-thumb"});
        const pass = enc.beginComputePass();
        pass.setPipeline(pl);
        pass.setBindGroup(0, device.createBindGroup({
            layout: pl.getBindGroupLayout(0),
            entries: [{binding: 0, resource}, {binding: 1, resource: {buffer: u}}, {binding: 2, resource: {buffer: out}}],
        }));
        pass.dispatchWorkgroups(THUMB / 8, THUMB / 8);
        pass.end();
        // submit per frame: an imported external texture is only valid until the task ends
        device.queue.submit([enc.finish()]);
    }
    const data = await readBack(device, out, n * THUMB_FLOATS * 4);
    out.destroy();
    temps.forEach(b => b.destroy());
    tempTex.forEach(t => t.destroy());
    return data;
}

// Gate: one thread per thumbnail (it is ~0.26 MMAC; calibration runs once per video).
function gateShader(w: UNetWeights, o: Record<string, number>): string {
    const g = w.meta.gate;
    const F = g.feat, NB = w.meta.banks;
    const leaky = (v: string) => `select(${g.slope} * ${v}, ${v}, ${v} > 0.0)`;
    return `
@group(0) @binding(0) var<storage, read> thumbs: array<f32>;
@group(0) @binding(1) var<storage, read> gw: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;     // conv1 out, ${F}x16x16 per thumbnail
@group(0) @binding(3) var<storage, read_write> alphas: array<f32>;
@group(0) @binding(4) var<uniform> num: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= num.x) { return; }
    let t0 = n * ${THUMB_FLOATS}u;
    let s0 = n * ${F * 256}u;
    // conv1: 3 -> F, 3x3 stride 2 pad 1, 32x32 -> 16x16, LeakyReLU
    for (var oc = 0u; oc < ${F}u; oc = oc + 1u) {
        for (var y = 0u; y < 16u; y = y + 1u) {
            for (var x = 0u; x < 16u; x = x + 1u) {
                var v = gw[${o.conv1_b}u + oc];
                for (var ic = 0u; ic < 3u; ic = ic + 1u) {
                    for (var ky = 0u; ky < 3u; ky = ky + 1u) {
                        let iy = i32(y * 2u + ky) - 1;
                        if (iy < 0 || iy >= 32) { continue; }
                        for (var kx = 0u; kx < 3u; kx = kx + 1u) {
                            let ix = i32(x * 2u + kx) - 1;
                            if (ix < 0 || ix >= 32) { continue; }
                            v = v + gw[${o.conv1_w}u + ((oc * 3u + ic) * 3u + ky) * 3u + kx]
                                  * thumbs[t0 + (ic * 32u + u32(iy)) * 32u + u32(ix)];
                        }
                    }
                }
                scratch[s0 + (oc * 16u + y) * 16u + x] = ${leaky("v")};
            }
        }
    }
    // conv2: F -> F, stride 2, 16x16 -> 8x8, LeakyReLU, then the global mean
    var h: array<f32, ${F}>;
    for (var oc = 0u; oc < ${F}u; oc = oc + 1u) {
        var s = 0.0;
        for (var y = 0u; y < 8u; y = y + 1u) {
            for (var x = 0u; x < 8u; x = x + 1u) {
                var v = gw[${o.conv2_b}u + oc];
                for (var ic = 0u; ic < ${F}u; ic = ic + 1u) {
                    for (var ky = 0u; ky < 3u; ky = ky + 1u) {
                        let iy = i32(y * 2u + ky) - 1;
                        if (iy < 0 || iy >= 16) { continue; }
                        for (var kx = 0u; kx < 3u; kx = kx + 1u) {
                            let ix = i32(x * 2u + kx) - 1;
                            if (ix < 0 || ix >= 16) { continue; }
                            v = v + gw[${o.conv2_w}u + ((oc * ${F}u + ic) * 3u + ky) * 3u + kx]
                                  * scratch[s0 + (ic * 16u + u32(iy)) * 16u + u32(ix)];
                        }
                    }
                }
                s = s + ${leaky("v")};
            }
        }
        h[oc] = s / 64.0;
    }
    // linear + softmax(z / T)
    var z: array<f32, ${NB}>;
    var zmax = -1e30;
    for (var b = 0u; b < ${NB}u; b = b + 1u) {
        var v = gw[${o.fc_b}u + b];
        for (var k = 0u; k < ${F}u; k = k + 1u) { v = v + gw[${o.fc_w}u + b * ${F}u + k] * h[k]; }
        z[b] = v / ${g.temperature.toFixed(6)};
        zmax = max(zmax, z[b]);
    }
    var den = 0.0;
    for (var b = 0u; b < ${NB}u; b = b + 1u) { z[b] = exp(z[b] - zmax); den = den + z[b]; }
    for (var b = 0u; b < ${NB}u; b = b + 1u) { alphas[n * ${NB}u + b] = z[b] / den; }
}`;
}

// thumbnails -> the video's alpha (mean of the per-thumbnail softmaxes).
export async function runGate(device: GPUDevice, w: UNetWeights, thumbs: Float32Array): Promise<Float32Array> {
    const g = w.meta.gate;
    const NB = w.meta.banks;
    const n = thumbs.length / THUMB_FLOATS;
    if (NB > 8) throw new Error("runGate: more than 8 banks is not supported");
    const parts = ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"] as const;
    const off: Record<string, number> = {};
    let total = 0;
    for (const k of parts) { off[k] = total; total += g[k].count; }
    const packed = new Float32Array(total);
    for (const k of parts) packed.set(w.tensor(g[k]), off[k]);

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const mk = (size: number, usage: number) => device.createBuffer({size, usage});
    const gwBuf = mk(packed.byteLength, S);
    const thBuf = mk(thumbs.byteLength, S);
    const scratch = mk(n * g.feat * 256 * 4, GPUBufferUsage.STORAGE);
    const alphas = mk(n * NB * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const cnt = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(gwBuf, 0, packed);
    device.queue.writeBuffer(thBuf, 0, thumbs);
    device.queue.writeBuffer(cnt, 0, new Uint32Array([n, 0, 0, 0]));

    const pl = device.createComputePipeline({
        label: "unet-gate", layout: "auto",
        compute: {module: device.createShaderModule({code: gateShader(w, off)}), entryPoint: "main"},
    });
    const enc = device.createCommandEncoder({label: "unet-gate"});
    const pass = enc.beginComputePass();
    pass.setPipeline(pl);
    pass.setBindGroup(0, device.createBindGroup({
        layout: pl.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: thBuf}},
            {binding: 1, resource: {buffer: gwBuf}},
            {binding: 2, resource: {buffer: scratch}},
            {binding: 3, resource: {buffer: alphas}},
            {binding: 4, resource: {buffer: cnt}},
        ],
    }));
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    device.queue.submit([enc.finish()]);

    const per = await readBack(device, alphas, n * NB * 4);
    [gwBuf, thBuf, scratch, alphas, cnt].forEach(b => b.destroy());
    const alpha = new Float32Array(NB);
    for (let k = 0; k < n; k++) for (let b = 0; b < NB; b++) alpha[b] += per[k * NB + b] / n;
    return alpha;
}

// Fold: out[i] = sum_b alpha_b * banks[b * len + i], over vec4s. Banks stay f32; out matches the
// conv's precision (f16 networks fold in f32 and narrow once).
export function foldShader(f16: boolean): string {
    return `${f16 ? "enable f16;\n" : ""}
struct P { len: u32, banks: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> banks: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> out: array<${f16 ? "vec4<f16>" : "vec4f"}>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<uniform> alpha: array<vec4f, 2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    var acc = vec4f(0.0);
    for (var b = 0u; b < p.banks; b = b + 1u) {
        acc = acc + alpha[b / 4u][b % 4u] * banks[b * p.len + i];
    }
    out[i] = ${f16 ? "vec4<f16>(acc)" : "acc"};
}`;
}
