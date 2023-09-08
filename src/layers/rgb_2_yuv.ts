import Layer from "./base_layer";


class RGB2YUV extends Layer {

    label: "RGB2YUV"


    constructor(device: GPUDevice, inputTextures: GPUTexture[], outputTexture: GPUTexture){

        super(device, inputTextures, outputTexture)

        this.createUniform("rgb2yuv", "mat3x3f");

        this.shader = this.createStandardShader(`
        
               @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
              
                    let color = textureLoad(inputTexture0, vec2<i32>(input.tex_coord), 0);       
                    let yuv = rgb2yuv*color.xyz;
          
                return vec4f(yuv, 1.0);
              }     
        `);



        this.setUniform("rgb2yuv",  new Float32Array([
            0.299, -0.1473, 0.615, 1.0,
            0.587, -.2886, -.51499, 1.0,
            0.114,  0.436, -.1001, 1.0
        ]));



        this.defaultSetup();

    }


}

export default RGB2YUV;