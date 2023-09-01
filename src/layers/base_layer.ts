
class Layer {

    device: GPUDevice;
    shader: GPUShaderModule;
    pipeline: GPURenderPipeline;
    sampler: GPUSampler;
    renderPassDescriptor: GPURenderPassDescriptor;
    bindGroup: GPUBindGroup;
    label: string;
    inputTexture: GPUTexture;
    outputTexture: GPUTexture;

    constructor(device:GPUDevice, inputTexture: GPUTexture, outputTexture:GPUTexture){
        this.device = device;
        this.inputTexture = inputTexture;
        this.outputTexture = outputTexture;

    }

    run(){


        const encoder = this.device.createCommandEncoder({label: this.label});

        const pass = encoder.beginRenderPass(this.renderPassDescriptor);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);  // call our vertex shader 6 times
        pass.end();

        this.device.queue.submit([encoder.finish()]);

        console.log("Calling layer run");

    }


}

export default Layer;