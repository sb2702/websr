import RenderLayer from "../base_render_layer";


class DisplayLayer3C extends RenderLayer {

    label = "DisplayLayer3C"

    constructor(inputs: (GPUTexture|GPUExternalTexture|GPUBuffer)[], output: GPUTexture){
        super(inputs, output)

        this.vertexScale = {
            width: 1,
            height: 1
        };


        // Use nearest neighbor since we'll do manual bicubic sampling in the shader
        this.sampler =  this.device.createSampler({
            addressModeU: "clamp-to-edge",
            addressModeV: "clamp-to-edge", 
            magFilter: "nearest",
            minFilter: "nearest",
        });


    }


    private _builtForExternal: boolean | undefined;

    lazyLoadSetup(){
        const externalTexture = this.inputs[3] instanceof GPUExternalTexture;
        if (this.pipeline && this._builtForExternal === externalTexture) {
          // Same shader; the input texture itself may have changed, so rebind.
          this.bindGroup = this.defaultBindGroup();
          return;
        }
        this._builtForExternal = externalTexture;



        // We'll replace this with bicubic sampling function
        const textureLoad = 'sampleBicubic(inputTexture, ourSampler, input.tex_coord)';

        this.shader = this.device.createShaderModule({
                label: `${this.label}-shader`,
                code: `
                   ${this.defaultVertexShader()}
                   @group(0) @binding(0) var<storage, read_write> inputBuffer0: array<vec4f>;
                   @group(0) @binding(1) var<storage, read_write> inputBuffer1: array<vec4f>;
                   @group(0) @binding(2) var<storage, read_write> inputBuffer2: array<vec4f>;
                   @group(0) @binding(3) var inputTexture: ${externalTexture?  'texture_external': 'texture_2d<f32>'};
                   @group(0) @binding(4) var ourSampler: sampler;
                   
                   // Bicubic weight function (Catmull-Rom)
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
                   
                   // Bicubic sampling function
                   fn sampleBicubic(texture: ${externalTexture?  'texture_external': 'texture_2d<f32>'}, sampler: sampler, tex_coord: vec2<f32>) -> vec4<f32> {
                        let tex_size = vec2<f32>(textureDimensions(texture));
                        let pixel_coord = tex_coord * tex_size - 0.5;
                        let base_coord = floor(pixel_coord);
                        let fract_coord = pixel_coord - base_coord;
                        
                        var result = vec3<f32>(0.0);
                        var weight_sum = 0.0;
                        
                        // Sample 4x4 neighborhood for bicubic
                        for (var y = -1; y <= 2; y = y + 1) {
                            for (var x = -1; x <= 2; x = x + 1) {
                                let sample_coord = (base_coord + vec2<f32>(f32(x), f32(y)) + 0.5) / tex_size;
                                let sample_color = ${externalTexture ? 'textureSampleBaseClampToEdge(texture, sampler, sample_coord).rgb' : 'textureSample(texture, sampler, sample_coord).rgb'};
                                
                                let weight_x = bicubic_weight(fract_coord.x - f32(x));
                                let weight_y = bicubic_weight(fract_coord.y - f32(y));
                                let weight = weight_x * weight_y;
                                
                                result += sample_color * weight;
                                weight_sum += weight;
                            }
                        }
                        
                        // Normalize
                        if (weight_sum > 0.0) {
                            result = result / weight_sum;
                        }
                        
                        return vec4<f32>(result, 1.0);
                   }
                  
                   @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                      
                        let x = ${this.resolution.width}.0*(input.tex_coord.x);
                        let y = ${this.resolution.height}.0*(input.tex_coord.y);
                        
                        let y2 = u32(floor(y));
                        let x2 = u32(floor(x));
                        
                        let i = y2*${Math.floor(this.resolution.width)} +  x2;
                       
                        let x_floor  = u32(fract(x)*2.0);
                        let y_floor  = u32(fract(y)*2.0);
                        
                        //I don t know, I think this is right? I found this by trial and error
                        let c_index: u32 = x_floor + y_floor*2;  
        
                        let value = f32(inputBuffer0[i][c_index]);
                        let value1 = f32(inputBuffer1[i][c_index]);
                        let value2 = f32(inputBuffer2[i][c_index]);
                        
                        let bicubic = ${textureLoad};
                        
                        return bicubic + vec4f(value, value1, value2, 1.0);
                    
                      }            
            `
            });

        this.pipeline =  this.device.createRenderPipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
        this.renderPassDescriptor = this.defaultRenderPassDescriptor();

    }


    defaultPipelineConfig(): GPURenderPipelineDescriptor {

        return {
            label: `${this.label}-pipeline`,
            layout: 'auto',
            vertex: {
                module: this.shader,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: this.shader,
                entryPoint: 'fragmentMain',
                targets: [{format: this.output.format}],
            },
        }

    }


    defaultBindGroup(): GPUBindGroup {


        const entries: any[]  = [];

        this.inputs.forEach(function (input, i) {
            if(input instanceof GPUExternalTexture){
                entries.push({ binding: i, resource: input})
            } else if (input instanceof GPUTexture){
                entries.push({ binding: i, resource: input.createView()})
            } else  if(input instanceof GPUBuffer){
                entries.push({ binding: i, resource: {buffer: input}})
            }

        });


        entries.push({ binding: this.inputs.length, resource: this.sampler })


        return this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries
        });


    }



}

export default DisplayLayer3C;