
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



    defaultVertexShader(){

        return `
        
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
              return vsOutput;
            }
        `
    }



    defaultPipelineConfig(): GPURenderPipelineDescriptor{

        return {
            label: `${this.label}-pipeline`,
            layout: 'auto',
            vertex: {
                module: this.shader,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: this.shader,
                entryPoint: 'fragmentMain',
                targets: [{format: this.outputTexture.format}],
            },
        }

    }

    defaultRenderPassDescriptor(): GPURenderPassDescriptor{

        return   {
            label: `${this.label}-render-pass`,
            colorAttachments: [
                {
                    view:  this.outputTexture.createView(),
                    clearValue: [0, 0, 0, 1],
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        };

    }


    createBuffer(label: string, value: Float32Array):GPUBuffer  {

        const buffer= this.device.createBuffer({
            label,
            size: value.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });



        this.device.queue.writeBuffer(buffer, /*bufferOffset=*/0, value);

        return  buffer;


    }

    run(){


        const encoder = this.device.createCommandEncoder({label: this.label});

        const pass = encoder.beginRenderPass(this.renderPassDescriptor);

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);  // call our vertex shader 6 times
        pass.end();

        this.device.queue.submit([encoder.finish()]);


    }


}

export default Layer;