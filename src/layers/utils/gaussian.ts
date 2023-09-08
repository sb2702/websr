import Layer from "../base_layer";

class GuassianLayer extends Layer {

    label = "Gaussian"


    constructor(device: GPUDevice, inputTextures: GPUTexture[], outputTexture: GPUTexture){
        super(device, inputTextures, outputTexture)


        this.createUniform("gaussian", "array<vec3f, 3>");
        this.createUniform("kernel_offsets", "array<vec4f, 9>");


        this.shader = this.createStandardShader(`
        
                  @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
                     var val  = 0.0;
                      
                     for(var i = 0u; i < 3; i++){
                     
                        let a = vec3f(
                            textureLoad(inputTexture0, vec2<i32>(input.tex_coord + kernel_offsets[i*3].xy), 0).x,
                            textureLoad(inputTexture0, vec2<i32>(input.tex_coord + kernel_offsets[i*3].xy), 0).x,
                            textureLoad(inputTexture0, vec2<i32>(input.tex_coord + kernel_offsets[i*3].xy), 0).x
                        );
                        
                        val += dot(a, gaussian[i]);
                      
                    } 
                  
                    
                    return vec4f(val, val, val, 1.0);
                  }                 
        `);



        this.setUniform("gaussian",  new Float32Array([
            0.0675,  0.125,  0.0675, 0.0,
            0.125,  0.250,  0.1250, 0.0,
            0.0675,  0.125,  0.0675 , 0.0
        ]));



        this.setUniform("kernel_offsets",  new Float32Array([
            -1, -1, 0, 0,
            0     , -1, 0, 0,
            1 , -1, 0, 0,
            -1,      0, 0, 0,
            0     ,      0, 0, 0,
            1 ,      0, 0, 0,
            -1,  1, 0, 0,
            0     ,  1, 0, 0,
            1 ,  1, 0, 0,
        ]));


        this.defaultSetup();

    }


}

export default GuassianLayer;