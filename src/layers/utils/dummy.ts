import RenderLayer from "../base_render_layer";


class DummyLayer extends RenderLayer {

    label = "Dummy"


    constructor(inputTextures: GPUTexture[], outputTexture: GPUTexture){
        super(inputTextures, outputTexture)


        this.shader = this.createStandardShader(`
        
              @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                
                return vec4f(0.0, 0.0, 0.0, 1.0);
              }                      
        `);


        this.defaultSetup();
    }

}

export default DummyLayer;