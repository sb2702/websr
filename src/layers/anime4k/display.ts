import RenderLayer from "../base_render_layer";


class DisplayLayer extends RenderLayer {

    label = "DisplayLayer"

    constructor(inputs: (GPUTexture|GPUExternalTexture|GPUBuffer)[], output: GPUTexture){
        super(inputs, output)

        this.vertexScale = {
            width: 1,
            height: 1
        };


        this.sampler =  this.device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
        });


    }


    lazyLoadSetup(){


        const externalTexture = this.inputs[1] instanceof GPUExternalTexture;

        const textureLoad = externalTexture ? 'textureSampleBaseClampToEdge(inputTexture, ourSampler, input.tex_coord)' :
            'textureSample(inputTexture, ourSampler, input.tex_coord)';

        this.shader = this.device.createShaderModule({
                label: `${this.label}-shader`,
                code: `
                
                   ${this.defaultVertexShader()}
                   @group(0) @binding(0) var<storage, read_write> inputBuffer0: array<vec4f>;
                   @group(0) @binding(1) var inputTexture: ${externalTexture?  'texture_external': 'texture_2d<f32>'};
                   @group(0) @binding(2) var ourSampler: sampler;
                  
                   @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                      
                        let x = i32(${this.resolution.width*2}.0*(input.tex_coord.x));
                        let y = i32(${this.resolution.height*2}.0*(input.tex_coord.y));
                        
                        let i  = y*${this.resolution.width*2} + x;

                        let y2 = u32(floor(f32(y)/2.0));
                        let x2 = u32(floor(f32(x)/2.0));
                        
                        let i2 = y2*${Math.floor(this.resolution.width)} +  x2;
                       
                        let x_floor  = u32(fract(f32(x)/2.0)*2.0);
                        let y_floor  = u32(fract(f32(y)/2.0)*2.0);
                        
                        //I don t know, I think this is right? I found this by trial and error
                        let c_index: u32 = x_floor + y_floor*2;  
        
                        let value = inputBuffer0[i2][c_index];
                        
                        let bicubic = ${textureLoad};
                        
                        return bicubic + vec4f(value);
                    
                      }            
            `
            });

        this.pipeline =  this.device.createRenderPipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
        this.renderPassDescriptor = this.defaultRenderPassDescriptor();

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

    setOutput(outputTexture: GPUTexture){

        this.output = outputTexture;

        this.renderPassDescriptor = this.defaultRenderPassDescriptor();
    }



}

export default DisplayLayer;