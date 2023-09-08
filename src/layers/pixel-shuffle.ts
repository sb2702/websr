import Layer from "./base_layer";


class PixelShuffle2X extends Layer {

    label = "PixelShuffle2X"


    constructor(device: GPUDevice, inputTexture: GPUTexture, outputTexture: GPUTexture){
        super(device, inputTexture, outputTexture)



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
              vsOutput.tex_coord = vsOutput.tex_coord*512.0;
              return vsOutput;
            }
          

              
              ${this.fragmentShaderInputs()}
              
               @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
                    let x_int  = i32(floor(input.tex_coord.x/256.0));
                    let y_int  = i32(floor(input.tex_coord.y/256.0));
                    let c_index = x_int + y_int*2;
                    
                    let value = textureLoad(inputTexture, vec2<i32>(i32(fract(input.tex_coord.x/256.0)*256.0), i32(fract(input.tex_coord.y/256.0)*256.0)), 0)[c_index];
                   
                    
                    return vec4f(value, value, value, 1.0);
                
                  }            
        `
        });


        this.defaultSetup();



    }




}

export default PixelShuffle2X;