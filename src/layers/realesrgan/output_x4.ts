import RenderLayer from "../base_render_layer";

// Final output layer for RealESRGAN x4v3.
// Inputs:
//   inputBuffer0 — final conv output, 48ch (12 vec4 per input pixel)
//   inputTexture1 — original input image (regular or external texture)
// Output: canvas texture at 4× input resolution.
//
// Fused operations:
//   1. PixelShuffle ×4 on the 48ch buffer → 3ch at 4× res
//      (PyTorch convention: out[oy, ox, c] = in[oy/4, ox/4, c*16 + (oy%4)*4 + (ox%4)])
//   2. Nearest-neighbour ×4 of the original input
//   3. Element-wise add → canvas

class RealESRGANOutputX4 extends RenderLayer {
    label = "RealESRGANOutputX4";
    private useF16: boolean;

    constructor(inputs: (GPUBuffer | GPUTexture | GPUExternalTexture)[], output: GPUTexture, useF16: boolean = false) {
        super(inputs, output);
        this.useF16 = useF16;
        // vertexScale=(1,1) so tex_coord ends up in [0,1] across the output;
        // we then scale by processResolution to recover input-pixel coords with sub-pixel fract.
        this.vertexScale = { width: 1, height: 1 };
        this.sampler = this.device.createSampler({
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
            magFilter: "nearest",
            minFilter: "nearest",
        });
    }

    private _builtForExternal: boolean | undefined;

    lazyLoadSetup() {
        const externalTexture = this.inputs[1] instanceof GPUExternalTexture;
        if (this.pipeline && this._builtForExternal === externalTexture) { this.bindGroup = this.defaultBindGroup(); return; }   // rebind: the texture may have changed
        this._builtForExternal = externalTexture;

        const textureType = externalTexture ? "texture_external" : "texture_2d<f32>";
        const textureSample = externalTexture
            ? "textureSampleBaseClampToEdge(inputTexture, ourSampler, in.tex_coord)"
            : "textureSample(inputTexture, ourSampler, in.tex_coord)";

        // f16 mode: the final conv buffer is vec4<f16>. Read it and widen to f32 so the
        // residual add and the vec4f fragment output stay f32 (`enable f16` must lead the module).
        const enable = this.useF16 ? "enable f16;\n" : "";
        const bufType = this.useF16 ? "vec4<f16>" : "vec4f";
        const rp = this.useF16 ? "f32(" : "";
        const rs = this.useF16 ? ")" : "";

        this.shader = this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `${enable}
              ${this.defaultVertexShader()}

              @group(0) @binding(0) var<storage, read> inputBuffer0: array<${bufType}>;
              @group(0) @binding(1) var inputTexture: ${textureType};
              @group(0) @binding(2) var ourSampler: sampler;

              @fragment fn fragmentMain(in: VertexShaderOutput) -> @location(0) vec4f {
                  // tex_coord ∈ [0,1] across output; map to input-pixel coords.
                  let x = ${this.resolution.width}.0 * in.tex_coord.x;
                  let y = ${this.resolution.height}.0 * in.tex_coord.y;
                  let ix = u32(floor(x));
                  let iy = u32(floor(y));
                  let sub_x = u32(floor(fract(x) * 4.0));
                  let sub_y = u32(floor(fract(y) * 4.0));

                  // PixelShuffle ×4: 12 vec4 per input pixel (48 channels).
                  // c=0 → in_vec4 = sub_y;     c=1 → in_vec4 = 4 + sub_y;  c=2 → in_vec4 = 8 + sub_y.
                  // Component within vec4 = sub_x.
                  let pixel_base = (iy * ${this.resolution.width}u + ix) * 12u;
                  let r = ${rp}inputBuffer0[pixel_base + sub_y][sub_x]${rs};
                  let g = ${rp}inputBuffer0[pixel_base + 4u  + sub_y][sub_x]${rs};
                  let b = ${rp}inputBuffer0[pixel_base + 8u  + sub_y][sub_x]${rs};

                  // Nearest-neighbour residual from the original input.
                  let res = ${textureSample};

                  return vec4f(r + res.r, g + res.g, b + res.b, 1.0);
              }
            `,
        });

        this.pipeline = this.device.createRenderPipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
        this.renderPassDescriptor = this.defaultRenderPassDescriptor();
    }

    defaultBindGroup(): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [];
        this.inputs.forEach((input, i) => {
            if (input instanceof GPUExternalTexture) {
                entries.push({ binding: i, resource: input });
            } else if (input instanceof GPUTexture) {
                entries.push({ binding: i, resource: input.createView() });
            } else if (input instanceof GPUBuffer) {
                entries.push({ binding: i, resource: { buffer: input } });
            }
        });
        entries.push({ binding: this.inputs.length, resource: this.sampler });
        return this.device.createBindGroup({
            label: `${this.label}-bg`,
            layout: this.pipeline.getBindGroupLayout(0),
            entries,
        });
    }
}

export default RealESRGANOutputX4;
