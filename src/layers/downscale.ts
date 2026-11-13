import WebGPUContext from "../context";

export default class DownscaleLayer {

    private device: GPUDevice;
    private context: WebGPUContext;
    private pipeline: GPURenderPipeline;
    private sampler: GPUSampler;
    private bindGroup: GPUBindGroup;
    private renderPassDescriptor: GPURenderPassDescriptor;

    constructor(input: GPUTexture, output: GPUTexture) {

        this.context = globalThis.context;
        this.device = this.context.device;

        this.sampler = this.device.createSampler({
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge",
            magFilter: "linear",
            minFilter: "linear",
        });

        const shader = this.device.createShaderModule({
            label: 'downscale-shader',
            code: `
                struct VertexShaderOutput {
                    @builtin(position) position: vec4f,
                    @location(0) tex_coord: vec2f,
                };

                @vertex
                fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexShaderOutput {
                    let pos = array(
                        vec2f(-1.0, -1.0),
                        vec2f( 1.0, -1.0),
                        vec2f(-1.0,  1.0),
                        vec2f(-1.0,  1.0),
                        vec2f( 1.0, -1.0),
                        vec2f( 1.0,  1.0),
                    );

                    var vsOutput: VertexShaderOutput;
                    let xy = pos[vertexIndex];
                    vsOutput.position = vec4f(xy, 0.0, 1.0);
                    vsOutput.tex_coord = xy * 0.5 + 0.5;
                    vsOutput.tex_coord.y = 1.0 - vsOutput.tex_coord.y;
                    return vsOutput;
                }

                @group(0) @binding(0) var inputTexture: texture_2d<f32>;
                @group(0) @binding(1) var ourSampler: sampler;

                fn bicubic_weight(t: f32) -> f32 {
                    let abs_t = abs(t);
                    if (abs_t >= 2.0) {
                        return 0.0;
                    }
                    let t2 = t * t;
                    let t3 = abs_t * t2;
                    if (abs_t <= 1.0) {
                        return 1.5 * t3 - 2.5 * t2 + 1.0;
                    } else {
                        return -0.5 * t3 + 2.5 * t2 - 4.0 * abs_t + 2.0;
                    }
                }

                @fragment
                fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                    let tex_size = vec2<f32>(textureDimensions(inputTexture));
                    let pixel_coord = input.tex_coord * tex_size - 0.5;
                    let base_coord = floor(pixel_coord);
                    let fract_coord = pixel_coord - base_coord;

                    var result = vec3<f32>(0.0);
                    var weight_sum = 0.0;

                    for (var y = -1; y <= 2; y = y + 1) {
                        for (var x = -1; x <= 2; x = x + 1) {
                            let sample_coord = (base_coord + vec2<f32>(f32(x), f32(y)) + 0.5) / tex_size;
                            let sample_color = textureSample(inputTexture, ourSampler, sample_coord).rgb;

                            let weight_x = bicubic_weight(fract_coord.x - f32(x));
                            let weight_y = bicubic_weight(fract_coord.y - f32(y));
                            let weight = weight_x * weight_y;

                            result += sample_color * weight;
                            weight_sum += weight;
                        }
                    }

                    if (weight_sum > 0.0) {
                        result = result / weight_sum;
                    }

                    return vec4f(result, 1.0);
                }
            `
        });

        this.pipeline = this.device.createRenderPipeline({
            label: 'downscale-pipeline',
            layout: 'auto',
            vertex: {
                module: shader,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shader,
                entryPoint: 'fragmentMain',
                targets: [{ format: output.format }],
            },
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: this.sampler },
            ],
        });

        this.renderPassDescriptor = {
            label: 'downscale-render-pass',
            colorAttachments: [{
                view: output.createView(),
                clearValue: [0, 0, 0, 1],
                loadOp: 'clear' as const,
                storeOp: 'store' as const,
            }],
        };
    }

    setOutput(output: GPUTexture) {
        this.renderPassDescriptor = {
            label: 'downscale-render-pass',
            colorAttachments: [{
                view: output.createView(),
                clearValue: [0, 0, 0, 1],
                loadOp: 'clear' as const,
                storeOp: 'store' as const,
            }],
        };
    }

    run() {
        const encoder = this.device.createCommandEncoder({ label: 'downscale' });
        const pass = encoder.beginRenderPass(this.renderPassDescriptor);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}
