import Layer from "./base_layer";


class DummyLayer extends Layer {

    label = "Dummy"


    constructor(device: GPUDevice, inputTexture: GPUTexture, outputTexture: GPUTexture){
        super(device, inputTexture, outputTexture)




        this.shader = this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `
          
              ${this.defaultVertexShader()}
              
                  @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                    
                    return vec4f(0.0, 0.0, 0.0, 1.0);
                  }                 
        `
        });





        this.pipeline = this.device.createRenderPipeline(this.defaultPipelineConfig());

        this.renderPassDescriptor = this.defaultRenderPassDescriptor();




    }

    run(){


        const encoder = this.device.createCommandEncoder({label: this.label});

        const pass = encoder.beginRenderPass(this.renderPassDescriptor);

        pass.setPipeline(this.pipeline);
        pass.draw(6);  // call our vertex shader 6 times
        pass.end();

        this.device.queue.submit([encoder.finish()]);


    }


}

export default DummyLayer;