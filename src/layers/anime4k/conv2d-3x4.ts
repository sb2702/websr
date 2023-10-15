import RenderLayer from "../base_render_layer";


class Anime4KConv3x4 extends RenderLayer {

    label = "Anime4KConv3x4"

    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputTexture: GPUTexture, weights: any){
        super(inputTextures, outputTexture, weights)


        const kernels: number[] = weights.weights;
        const bias: number[] = weights.bias;


        this.createUniform("kernel_offsets", "array<vec4f, 9>");
        this.createUniform("kernels", "array<mat4x4f, 9>");
        this.createUniform("bias", "vec4f");

        // Set up pipeline in Lazy Load

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

    }

    lazyLoadSetup(){


        const externalTexture = this.inputTextures[0] instanceof GPUExternalTexture;

        const textureLoad = externalTexture ? 'textureLoad(inputTexture0, vec2<i32>(input.tex_coord + kernel_offsets[i].xy));' :
            'textureLoad(inputTexture0, vec2<i32>(input.tex_coord + kernel_offsets[i].xy), 0);';


        this.shader = this.createStandardShader(`
        
                  @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
                     var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                      
                     for(var i = 0u; i < 9; i++){
                       result += kernels[i]*${textureLoad}
                    } 
                    
                    result += bias;
                  
                    
                    return result;
                  }                 
        `);


        this.pipeline =  this.device.createRenderPipeline(this.defaultPipelineConfig());
        this.renderPassDescriptor = this.defaultRenderPassDescriptor();

    }


    run(){



        const encoder = this.device.createCommandEncoder({label: this.label});

        if(!this.pipeline) {
            this.lazyLoadSetup();
        }

        const pass = encoder.beginRenderPass(this.renderPassDescriptor);



        pass.setPipeline(this.pipeline);

        this.bindGroup = this.defaultBindGroup();

        pass.setBindGroup(0, this.bindGroup);


        pass.draw(6);  // call our vertex shader 6 times
        pass.end();

        this.device.queue.submit([encoder.finish()]);



    }




}

export default Anime4KConv3x4;