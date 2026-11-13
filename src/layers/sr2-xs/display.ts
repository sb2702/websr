import RenderLayer from "../base_render_layer";

// SR2-XS display: fuses depth_to_space + bilinear upsample + residual add +
// canvas write into a single fragment shader.
//
// Inputs:
//   [0]  16-channel LR buffer (4 vec4 per pixel) — the model's final residual
//        prediction, raw signed values pre-DepthToSpace
//   [1]  source RGBA texture — used for the bilinear residual base via the
//        GPU's hardware linear sampler
// Output: canvas texture at HR (2× LR) resolution
//
// Per HR output pixel (HR_x, HR_y):
//   - LR pixel position: (floor(HR_x/2), floor(HR_y/2))
//   - 2×2 sub-pixel index: dy*2 + dx where (dx, dy) = (HR_x%2, HR_y%2)
//   - R residual = inputBuffer0[lr_idx*4 + 0][sub_idx]   (channel group 0, lane sub_idx)
//   - G residual = inputBuffer0[lr_idx*4 + 1][sub_idx]   (channel group 1)
//   - B residual = inputBuffer0[lr_idx*4 + 2][sub_idx]   (channel group 2)
//   - The 4th channel group (channels 12-15) is unused — it's the wasted slot
//     from the 4-channel padding (alpha) of the 16-channel projection output.
//   - bilinear = textureSample(input, ...) — hardware linear filtering ≈
//     PyTorch's bilinear interpolation (align_corners=False default)
//   - canvas = bilinear + vec4(R, G, B, 0)
class SR2XSDisplay extends RenderLayer {

    label = "SR2XSDisplay"

    constructor(inputs: (GPUTexture | GPUExternalTexture | GPUBuffer)[], output: GPUTexture) {
        super(inputs, output);

        // Same vertex-scale trick as Anime4K's DisplayLayer3C: keep tex_coord
        // normalized in [0,1] and rescale to LR coords inside the fragment.
        this.vertexScale = { width: 1, height: 1 };

        this.sampler = this.device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter:    "linear",
            minFilter:    "linear",
            mipmapFilter: "linear",
        });
    }

    lazyLoadSetup() {
        const externalTexture = this.inputs[1] instanceof GPUExternalTexture;
        const textureLoad = externalTexture
            ? 'textureSampleBaseClampToEdge(inputTexture, ourSampler, input.tex_coord)'
            : 'textureSample(inputTexture, ourSampler, input.tex_coord)';

        this.shader = this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `
                ${this.defaultVertexShader()}
                @group(0) @binding(0) var<storage, read_write> inputBuffer0: array<vec4f>;
                @group(0) @binding(1) var inputTexture: ${externalTexture ? 'texture_external' : 'texture_2d<f32>'};
                @group(0) @binding(2) var ourSampler: sampler;

                @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                    let x = ${this.resolution.width}.0 * (input.tex_coord.x);
                    let y = ${this.resolution.height}.0 * (input.tex_coord.y);

                    let lr_x = u32(floor(x));
                    let lr_y = u32(floor(y));
                    let lr_idx = lr_y * ${Math.floor(this.resolution.width)}u + lr_x;

                    let x_floor = u32(fract(x) * 2.0);
                    let y_floor = u32(fract(y) * 2.0);
                    let sub: u32 = x_floor + y_floor * 2u;

                    let r_residual = inputBuffer0[lr_idx * 4u + 0u][sub];
                    let g_residual = inputBuffer0[lr_idx * 4u + 1u][sub];
                    let b_residual = inputBuffer0[lr_idx * 4u + 2u][sub];

                    let bilinear = ${textureLoad};
                    return bilinear + vec4f(r_residual, g_residual, b_residual, 0.0);
                }
            `,
        });

        this.pipeline = this.device.createRenderPipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
        this.renderPassDescriptor = this.defaultRenderPassDescriptor();
    }

    defaultBindGroup(): GPUBindGroup {
        const entries: any[] = [];
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
            layout: this.pipeline.getBindGroupLayout(0),
            entries,
        });
    }
}

export default SR2XSDisplay;
