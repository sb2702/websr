import RenderLayer from "../base_render_layer";


class RGB2YUV extends RenderLayer {

    label = "RGB2YUV"

    private _builtForExternal: boolean | undefined;

    // inputs[0] is the source texture; it is patched per frame by the network, so the
    // pipeline is built lazily (external textures need a different shader).
    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputTexture: GPUTexture){

        super(inputTextures, outputTexture)

        this.createUniform("rgb2yuv", "mat3x3f");

        this.setUniform("rgb2yuv",  new Float32Array([
            0.299, -0.1473, 0.615, 1.0,
            0.587, -.2886, -.51499, 1.0,
            0.114,  0.436, -.1001, 1.0
        ]));
    }

    lazyLoadSetup(){
        const externalTexture = this.inputs[0] instanceof GPUExternalTexture;
        if (this.pipeline && this._builtForExternal === externalTexture) {
            this.bindGroup = this.defaultBindGroup();
            return;
        }
        this._builtForExternal = externalTexture;

        const load = externalTexture ? 'textureLoad(inputTexture0, vec2<i32>(input.tex_coord))' :
            'textureLoad(inputTexture0, vec2<i32>(input.tex_coord), 0)';

        this.shader = this.createStandardShader(`

               @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {

                    let color = ${load};
                    let yuv = rgb2yuv*color.xyz;

                return vec4f(yuv, 1.0);
              }
        `);

        this.defaultSetup();
    }


}

export default RGB2YUV;
