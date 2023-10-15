import Layer from "../base_layer";
import RenderLayer from "../base_render_layer";


class DisplayLayer extends RenderLayer {

    label = "DisplayLayer"

    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputTexture: GPUTexture){
        super(inputTextures, outputTexture)

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


        const externalTexture = this.inputTextures[1] instanceof GPUExternalTexture;

        const textureLoad = externalTexture ? 'textureSampleBaseClampToEdge(inputTexture, ourSampler, input.tex_coord)' :
            'textureSample(inputTexture, ourSampler, input.tex_coord)';

        this.shader = this.device.createShaderModule({
                label: `${this.label}-shader`,
                code: `
                
                   ${this.defaultVertexShader()}
                  
                   @group(0) @binding(0) var pixelShuffle: texture_2d<f32>;
                   @group(0) @binding(1) var inputTexture: ${externalTexture?  'texture_external': 'texture_2d<f32>'};
                   @group(0) @binding(2) var ourSampler: sampler;
                  
                   @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                      
                  
                        let x = i32(${this.resolution.width*2}.0*(input.tex_coord.x));
                        let y = i32(${this.resolution.height*2}.0*(input.tex_coord.y));
                        
                        let value = textureLoad(pixelShuffle, vec2<i32>(x, y), 0).x;
                        
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


        this.inputTextures.forEach(function (texture, i) {


            if(texture instanceof GPUExternalTexture){
                entries.push({ binding: i, resource: texture})
            } else if (texture instanceof GPUTexture){
                entries.push({ binding: i, resource: texture.createView()})
            }

        });

        entries.push({ binding: this.inputTextures.length, resource: this.sampler })


        return this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries
        });


    }

    setOutput(outputTexture: GPUTexture){

        this.outputTexture = outputTexture;

        this.renderPassDescriptor = this.defaultRenderPassDescriptor();
    }



}

export default DisplayLayer;