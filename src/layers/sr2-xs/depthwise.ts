import ComputeLayer from "../base_compute_layer";

// SR2-XS depthwise (16-channel) with CReLU on the input. Buffer-in / buffer-out
// are both 4 vec4 per LR pixel (16 channels). Per output channel: positive
// kernel × max(input, 0) + negative kernel × max(-input, 0), summed over the
// 3×3 spatial kernel.
//
// Weights: 9 kernel positions × 4 channel groups × 2 (pos + neg) = 72 vec4 (288 floats).
//   Layout: kernels[k*4 + g]      for pos at position k, group g  (indices 0..35)
//           kernels[36 + k*4 + g] for neg                          (indices 36..71)
// Bias: 16 floats = 4 vec4.
//
// Bounds-check on neighbour reads (skip → 0) so border behaviour matches
// PyTorch's zero-padding — the test page's MSE diff covers the whole frame,
// no border-mask hack needed.
class SR2XSDepthwise extends ComputeLayer {

    label = "SR2XSDepthwise"

    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any) {
        super(inputs, outputBuffer, weights);

        const kernels: number[] = weights.weights;   // 288 floats
        const bias:    number[] = weights.bias;      // 16 floats

        this.createUniform("kernel_offsets", "array<vec4f, 9>");
        this.createUniform("kernels",        "array<vec4f, 72>");
        this.createUniform("bias",           "array<vec4f, 4>");

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
        const w = this.resolution.width;
        const h = this.resolution.height;

        this.shader = this.createStandardShader(`
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups})
          fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let coord = vec2<i32>(i32(x), i32(y));
                let pix_base = (y * ${w}u + x) * 4u;

                for (var g = 0u; g < 4u; g++) {
                    var result = vec4f(0.0);
                    for (var k = 0u; k < 9u; k++) {
                        let p = coord + vec2<i32>(kernel_offsets[k].xy);
                        if (p.x < 0 || p.y < 0 || p.x >= ${w} || p.y >= ${h}) { continue; }
                        let in_idx = (u32(p.y) * ${w}u + u32(p.x)) * 4u + g;
                        let pix = inputBuffer0[in_idx];

                        result += kernels[k * 4u + g]        * max( pix, vec4f(0.0));
                        result += kernels[36u + k * 4u + g]  * max(-pix, vec4f(0.0));
                    }
                    result += bias[g];
                    outputBuffer[pix_base + g] = result;
                }
          }
        `);

        this.pipeline = this.device.createComputePipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
    }
}

export default SR2XSDepthwise;
