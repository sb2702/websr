import Layer from "../base_layer";


class DisplayLayer extends Layer {

    label = "DisplayLayer"

    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputTexture: GPUTexture){
        super(inputTextures, outputTexture)



        this.shader = this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `
            
            
              struct VertexShaderOutput {
                @builtin(position) position: vec4f,
                @location(0) tex_coord: vec2f,
              };

            @vertex
            fn vertexMain( @builtin(vertex_index) vertexIndex : u32) ->  VertexShaderOutput{
                let pos = array(
                // 1st triangle
                vec2f( -1.0,  -1.0),  // center
                vec2f( 1.0,  -1.0),  // right, center
                vec2f( -1.0,  1.0),  // center, top
             
                // 2st triangle
                vec2f( -1.0,  1.0),  // center, top
                vec2f( 1.0,  -1.0),  // right, center
                vec2f( 1.0,  1.0),  // right, top
              );
             
              var vsOutput: VertexShaderOutput;
              let xy = pos[vertexIndex];
              vsOutput.position = vec4f(xy, 0.0, 1.0);
              vsOutput.tex_coord = xy*0.5 + 0.5;
              vsOutput.tex_coord.y = - 1.0* vsOutput.tex_coord.y  + 1.0;
              vsOutput.tex_coord = vsOutput.tex_coord;
              return vsOutput;
            }
          

              
               @group(0) @binding(0) var pixelShuffle: texture_2d<f32>;
               @group(0) @binding(1) var inputTexture: texture_external;
               @group(0) @binding(2) var ourSampler: sampler;
              
               @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
              
                    let x = i32(${this.resolution.width*2}.0*(input.tex_coord.x));
                    let y = i32(${this.resolution.height*2}.0*(input.tex_coord.y));
                    
                    let value = textureLoad(pixelShuffle, vec2<i32>(x, y), 0).x;
                    
                    let bicubic = textureSampleBaseClampToEdge(inputTexture, ourSampler, input.tex_coord);
                    
                    return bicubic + vec4f(value);
                
                  }            
        `
        });

        this.sampler =  this.device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
        });


        this.pipeline = this.device.createRenderPipeline(this.defaultPipelineConfig());

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