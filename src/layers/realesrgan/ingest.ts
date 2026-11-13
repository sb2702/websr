import ComputeLayer from "../base_compute_layer";

// Copy an input image (texture or external texture) into a vec4-packed
// storage buffer with shape [H * W] vec4f, components (R, G, B, 0).
// This decouples the conv chain from texture-vs-buffer concerns and
// gives the residual path a consistent source.

class RealESRGANIngest extends ComputeLayer {
    label = "RealESRGANIngest";
    num_work_groups = 16;
    private useF16: boolean;

    constructor(inputTexture: GPUTexture | GPUExternalTexture, outputBuffer: GPUBuffer, useF16: boolean = false) {
        super([inputTexture], outputBuffer);
        this.useF16 = useF16;
    }

    private _builtForExternal: boolean | undefined;

    lazyLoadSetup() {
        const externalTexture = this.inputs[0] instanceof GPUExternalTexture;
        if (this.pipeline && this._builtForExternal === externalTexture) { this.bindGroup = this.defaultBindGroup(); return; }   // rebind: the texture may have changed
        this._builtForExternal = externalTexture;

        const textureType = externalTexture ? "texture_external" : "texture_2d<f32>";
        const textureLoadExpr = externalTexture
            ? "textureLoad(inputTexture0, coord)"
            : "textureLoad(inputTexture0, coord, 0)";

        // f16 mode: output is a vec4<f16> buffer (the conv chain runs in f16). The texture
        // sample stays f32; only the store narrows. Build the shader directly so the output
        // binding type matches (createStandardShader hardcodes vec4f).
        const v4 = this.useF16 ? "vec4<f16>" : "vec4f";
        const enable = this.useF16 ? "enable f16;\n" : "";
        const store = this.useF16
            ? "vec4<f16>(vec3<f16>(v.rgb), 0.0h)"
            : "vec4f(v.rgb, 0.0)";

        this.shader = this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `${enable}
          @group(0) @binding(0) var inputTexture0: ${textureType};
          @group(0) @binding(1) var<storage, read_write> outputBuffer: array<${v4}>;

          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups})
          fn main(@builtin(global_invocation_id) id: vec3<u32>) {
              if (id.x >= ${this.resolution.width}u || id.y >= ${this.resolution.height}u) { return; }
              let coord = vec2<i32>(i32(id.x), i32(id.y));
              let v = ${textureLoadExpr};
              let idx = id.y * ${this.resolution.width}u + id.x;
              outputBuffer[idx] = ${store};
          }
        `,
        });

        this.pipeline = this.device.createComputePipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
    }
}

export default RealESRGANIngest;
