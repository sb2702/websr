import ComputeLayer from "../base_compute_layer";

// SR2-XS stem: 3×3 conv from RGBA texture → 16-channel buffer (4 vec4 per LR pixel).
// No CReLU on input (the texture is in [0, 1] — no negatives to capture); raw
// signed output, the CReLU expansion happens INSIDE block0's depthwise.
//
// Buffer layout: `outputBuffer[(y*W + x)*4 + o]` for output group o ∈ [0, 4).
//
// Weights: 9 kernel positions × 4 output groups × 1 input group × 1 mat4x4 each
// = 36 mat4x4 (576 floats). Kernel position order is Anime4K's column-major
// (kx outer, ky inner), the same as the anime4k layers.
// Bias: 16 floats = 4 vec4.
class SR2XSStem extends ComputeLayer {

    label = "SR2XSStem"

    constructor(inputTextures: (GPUTexture | GPUExternalTexture)[], outputBuffer: GPUBuffer, weights: any) {
        super(inputTextures, outputBuffer, weights);

        const kernels: number[] = weights.weights;   // 576 floats
        const bias:    number[] = weights.bias;      // 16 floats

        this.createUniform("kernel_offsets", "array<vec4f, 9>");
        this.createUniform("kernels",        "array<mat4x4f, 36>");
        this.createUniform("bias",           "array<vec4f, 4>");

        // Anime4K column-major kernel order (kx outer, ky inner).
        this.setUniform("kernel_offsets", new Float32Array([
            -1, -1, 0, 0,
            -1,  0, 0, 0,
            -1,  1, 0, 0,
             0, -1, 0, 0,
             0,  0, 0, 0,
             0,  1, 0, 0,
             1, -1, 0, 0,
             1,  0, 0, 0,
             1,  1, 0, 0,
        ]));
        this.setUniform("kernels", new Float32Array(kernels));
        this.setUniform("bias",    new Float32Array(bias));
    }

    lazyLoadSetup() {
        const externalTexture = this.inputs[0] instanceof GPUExternalTexture;
        const textureLoad = externalTexture
            ? 'textureLoad(inputTexture0, coord + offset)'
            : 'textureLoad(inputTexture0, coord + offset, 0)';

        this.shader = this.createStandardShader(`
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups})
          fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                if (id.x >= ${this.resolution.width}u || id.y >= ${this.resolution.height}u) { return; }
                let x = id.x;
                let y = id.y;
                let coord = vec2<i32>(i32(x), i32(y));
                let pix_base = (y * ${this.resolution.width}u + x) * 4u;

                // Loop over 4 output groups (16 channels = 4 vec4 per pixel).
                for (var o = 0u; o < 4u; o++) {
                    var result = vec4f(0.0);
                    for (var k = 0u; k < 9u; k++) {
                        let offset = vec2<i32>(kernel_offsets[k].xy);
                        result += kernels[k * 4u + o] * ${textureLoad};
                    }
                    result += bias[o];
                    outputBuffer[pix_base + o] = result;
                }
          }
        `);

        this.pipeline = this.device.createComputePipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
    }
}

export default SR2XSStem;
