import Layer from "./base_layer";


class PixelShuffle2X extends Layer {

    label = "PixelShuffle2X"


    constructor(device: GPUDevice, inputTextures: GPUTexture[], outputTexture: GPUTexture){
        super(device, inputTextures, outputTexture)



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
          

              
               @group(0) @binding(0) var featureMap: texture_2d<f32>;
               @group(0) @binding(1) var inputTexture: texture_2d<f32>;
               @group(0) @binding(2) var ourSampler: sampler;
              
               @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
                    let x_floor  = u32(fract(input.tex_coord.x*256.0)*2.0);
                    
                    
                    let y_floor  = u32(fract(input.tex_coord.y*256.0)*2.0);
                    
                    //I don t know, I think this is right? I found this by trial and error
                    let c_index: u32 = 2*x_floor + 1-y_floor;  
              
                    let x = i32(256.0*(input.tex_coord.x));
                    let y = i32(256.0*(input.tex_coord.y));
                    
                    let value = textureLoad(featureMap, vec2<i32>(x, y), 0)[c_index];
                    
                    let bicubic = textureSample(inputTexture, ourSampler, input.tex_coord);
                    
                    
                  
                    return vec4f(bicubic.x+value, bicubic.y+value, bicubic.z+value, 1.0);
                
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



        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [

                { binding: 0, resource: this.inputTextures[0].createView() },
                { binding: 1, resource: this.inputTextures[1].createView() },
                { binding: 2, resource: this.sampler }
            ]
        });



        this.renderPassDescriptor = this.defaultRenderPassDescriptor();
    }
    




}

export default PixelShuffle2X;