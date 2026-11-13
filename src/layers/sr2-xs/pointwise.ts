import ComputeLayer from "../base_compute_layer";

// SR2-XS pointwise (1×1) with CReLU on the input, 16 → 16.
//
// Weights: 4 output groups × 4 input groups × 2 (pos + neg) × 1 mat4x4 = 32 mat4x4
//   (512 floats). Layout:
//     kernels[o*4 + i]       for pos: output group o ∈ [0,4), input group i ∈ [0,4)
//     kernels[16 + o*4 + i]  for neg
// Bias: 16 floats = 4 vec4.
//
// No spatial kernel ⇒ no neighbour reads ⇒ no bounds check.
class SR2XSPointwise extends ComputeLayer {

    label = "SR2XSPointwise"

    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any) {
        super(inputs, outputBuffer, weights);

        const kernels: number[] = weights.weights;   // 512 floats
        const bias:    number[] = weights.bias;      // 16 floats

        this.createUniform("kernels", "array<mat4x4f, 32>");
        this.createUniform("bias",    "array<vec4f, 4>");

        this.setUniform("kernels", new Float32Array(kernels));
        this.setUniform("bias",    new Float32Array(bias));
    }

    lazyLoadSetup() {
        const w = this.resolution.width;

        this.shader = this.createStandardShader(`
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups})
          fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let pix_base = (y * ${w}u + x) * 4u;

                for (var o = 0u; o < 4u; o++) {
                    var result = vec4f(0.0);
                    for (var i = 0u; i < 4u; i++) {
                        let pix = inputBuffer0[pix_base + i];
                        result += kernels[o * 4u + i]        * max( pix, vec4f(0.0));
                        result += kernels[16u + o * 4u + i]  * max(-pix, vec4f(0.0));
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

export default SR2XSPointwise;
