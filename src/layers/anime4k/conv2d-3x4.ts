import Layer from "../base_layer";


class Anime4KConv3x4 extends Layer {

    label = "Anime4KConv3x4"


    constructor(inputTextures: GPUExternalTexture[], outputTexture: GPUTexture, weights: any){
        super(inputTextures, outputTexture, weights)


        const kernels: number[] = weights.weights;
        const bias: number[] = weights.bias;


        this.createUniform("kernel_offsets", "array<vec4f, 9>");
        this.createUniform("kernels", "array<mat4x4f, 9>");
        this.createUniform("bias", "vec4f");




        this.shader = this.createStandardShader(`
        
                  @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
                     var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                      
                     for(var i = 0u; i < 9; i++){
                       result += kernels[i]*textureLoad(inputTexture0, vec2<i32>(input.tex_coord + kernel_offsets[i].xy));
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



        this.pipeline = this.device.createRenderPipeline(this.defaultPipelineConfig());

        this.renderPassDescriptor = this.defaultRenderPassDescriptor();


    }

    run(){



        const encoder = this.device.createCommandEncoder({label: this.label});

        const pass = encoder.beginRenderPass(this.renderPassDescriptor);

        pass.setPipeline(this.pipeline);

        this.bindGroup = this.defaultBindGroup();

        console.log("Bind grouo layout");
        console.log(this.pipeline.getBindGroupLayout(0));

        console.log(this.bindGroup);

        if(this.bindGroup) {
            pass.setBindGroup(0, this.bindGroup);
        }

        pass.draw(6);  // call our vertex shader 6 times
        pass.end();

        this.device.queue.submit([encoder.finish()]);



    }




}

export default Anime4KConv3x4;