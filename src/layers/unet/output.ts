import RenderLayer from "../base_render_layer";

// UNet output: PixelShuffle x2 of the 12-channel "out" buffer + Catmull-Rom bicubic residual of
// the input -> render target at 2x the process resolution.
//
//   out buffer: 3 vec4 per LR pixel; vec4 c (R, G, B) lane dy*2+dx  (== torch PixelShuffle(2))
//   residual:   websr display_3c's Catmull-Rom (a = -0.5), half-pixel centres, clamp-to-edge,
//               weights normalised by their sum — what the cnn-* networks ship, and what the UNet
//               students were trained against. Integer textureLoad with clamped coordinates
//               instead of a sampler, so it is exact and identical for texture_2d / external.
// The output pixel comes from @builtin(position) (pixel centre = index + 0.5): no reliance on
// interpolated texture coordinates.

class UNetOutput extends RenderLayer {
    label = "UNetOutput";
    private useF16: boolean;
    private _builtForExternal: boolean | undefined;

    constructor(inputs: (GPUBuffer | GPUTexture | GPUExternalTexture)[], output: GPUTexture, useF16: boolean = false) {
        super(inputs, output);
        this.useF16 = useF16;
    }

    lazyLoadSetup() {
        const external = this.inputs[1] instanceof GPUExternalTexture;
        if (this.pipeline && this._builtForExternal === external) { this.bindGroup = this.defaultBindGroup(); return; }   // rebind: the texture may have changed
        this._builtForExternal = external;

        const W = this.resolution.width;
        const H = this.resolution.height;
        const texType = external ? "texture_external" : "texture_2d<f32>";
        const load = external ? "textureLoad(inputTexture, c)" : "textureLoad(inputTexture, c, 0)";
        const bufType = this.useF16 ? "vec4<f16>" : "vec4f";
        const widen = this.useF16 ? "vec4f" : "";

        this.shader = this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `${this.useF16 ? "enable f16;\n" : ""}
              ${this.defaultVertexShader()}

              @group(0) @binding(0) var<storage, read> outBuf: array<${bufType}>;
              @group(0) @binding(1) var inputTexture: ${texType};

              fn cr_weight(t: f32) -> f32 {           // Catmull-Rom, a = -0.5
                  let a = abs(t);
                  if (a <= 1.0) { return 1.5 * a * a * a - 2.5 * a * a + 1.0; }
                  if (a < 2.0)  { return -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0; }
                  return 0.0;
              }

              fn src(x: i32, y: i32) -> vec3f {
                  let c = vec2<i32>(clamp(x, 0, ${W - 1}), clamp(y, 0, ${H - 1}));
                  return ${load}.rgb;
              }

              @fragment fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
                  let ox = u32(pos.x);
                  let oy = u32(pos.y);
                  let lx = ox / 2u;
                  let ly = oy / 2u;
                  let sub = (oy % 2u) * 2u + (ox % 2u);
                  let base = (ly * ${W}u + lx) * 3u;
                  let r = ${widen}(outBuf[base + 0u])[sub];
                  let g = ${widen}(outBuf[base + 1u])[sub];
                  let b = ${widen}(outBuf[base + 2u])[sub];

                  // Catmull-Rom residual sampled at the output pixel centre (2x, half-pixel).
                  let px = (f32(ox) + 0.5) * 0.5 - 0.5;
                  let py = (f32(oy) + 0.5) * 0.5 - 0.5;
                  let bx = floor(px);
                  let by = floor(py);
                  let fx = px - bx;
                  let fy = py - by;
                  var acc = vec3f(0.0);
                  var wsum = 0.0;
                  for (var j = -1; j <= 2; j = j + 1) {
                      let wy = cr_weight(fy - f32(j));
                      for (var i = -1; i <= 2; i = i + 1) {
                          let w = cr_weight(fx - f32(i)) * wy;
                          acc = acc + src(i32(bx) + i, i32(by) + j) * w;
                          wsum = wsum + w;
                      }
                  }
                  return vec4f(vec3f(r, g, b) + acc / wsum, 1.0);
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
            if (input instanceof GPUExternalTexture) entries.push({binding: i, resource: input});
            else if (input instanceof GPUTexture) entries.push({binding: i, resource: input.createView()});
            else if (input instanceof GPUBuffer) entries.push({binding: i, resource: {buffer: input}});
        });
        return this.device.createBindGroup({
            label: `${this.label}-bg`,
            layout: this.pipeline.getBindGroupLayout(0),
            entries,
        });
    }
}

export default UNetOutput;
