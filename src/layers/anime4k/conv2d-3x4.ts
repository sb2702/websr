import Layer from "../base_layer";


class Anime4KConv3x4 extends Layer {

    label = "Anime4KConv3x4"


    constructor(device: GPUDevice, inputTextures: GPUTexture[], outputTexture: GPUTexture, weights: any){
        super(device, inputTextures, outputTexture, weights)


        const kernels: number[] = weights.weights;
        const bias: number[] = weights.bias;


        this.createUniform("kernel_offsets", "array<vec4f, 9>");
        this.createUniform("kernels", "array<mat4x4f, 9>");
        this.createUniform("bias", "vec4f");

        this.shader = this.createStandardShader(`
        
                  @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
                     var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                      
                     for(var i = 0u; i < 9; i++){
                       result += kernels[i]*textureLoad(inputTexture0, vec2<i32>(input.tex_coord + kernel_offsets[i].xy), 0);
                    } 
                    
                    result += bias;
                  
                    
                    return result;
                  }                 
        `);


        this.setUniform("kernel_offsets",  new Float32Array([
            -1,  -1, 0, 0,
            -1 ,  0, 0, 0,
            -1 ,  1, 0, 0,
             0,  -1, 0, 0,
             0,   0, 0, 0,
             0,   1, 0, 0,
             1,  -1, 0, 0,
             1 ,  0, 0, 0,
             1 ,  1, 0, 0,
        ]));


        this.setUniform("kernels",  new Float32Array(kernels));
        this.setUniform("bias",  new Float32Array(bias));


        this.defaultSetup();

    }


}

export default Anime4KConv3x4;