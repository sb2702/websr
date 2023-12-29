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
                      
                        let x = ${this.resolution.width}.0*(input.tex_coord.x);
                        let y = ${this.resolution.height}.0*(input.tex_coord.y);
                        
                        let y2 = u32(floor(y));
                        let x2 = u32(floor(x));
                        
                        let i = y2*${Math.floor(this.resolution.width)} +  x2;
      
                        let bicubic = ${textureLoad};
                        
                        return bicubic + inputBuffer0[i];
                    
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


}

export default DisplayLayer;