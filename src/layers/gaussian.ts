import Layer from "./base_layer";


class GuassianLayer extends Layer {



    constructor(device: GPUDevice, inputTexture: GPUTexture, outputTexture: GPUTexture){
        super(device, inputTexture, outputTexture)

        this.shader = device.createShaderModule({
            label: 'Guassian',

            code: `
            
                    
                  ${this.defaultVertexShader()}
    
            
                  @group(0) @binding(0) var<uniform> gaussian: array<vec3f, 3>;
                  @group(0) @binding(1) var<uniform> kernel_offsets: array<vec4f, 9>;
                  @group(0) @binding(2) var ourSampler: sampler;
                  @group(0) @binding(3) var ourTexture: texture_2d<f32>;
            
                  @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                  
                     var val  = 0.0;
                      
                     for(var i = 0u; i < 3; i++){
                     
                        let a = vec3f(
                            textureSample(ourTexture, ourSampler, input.tex_coord + kernel_offsets[i*3].xy).x,
                            textureSample(ourTexture, ourSampler, input.tex_coord + kernel_offsets[i*3+1].xy).x,
                            textureSample(ourTexture, ourSampler, input.tex_coord + kernel_offsets[i*3+2].xy).x
                        );
                        
                        val += dot(a, gaussian[i]);
                      
                    } 
                  
                    
                    return vec4f(val, val, val, 1.0);
                  }              
            `
        });



        this.pipeline =  device.createRenderPipeline({
            label: 'Guassian Blur Pipeline',
            layout: 'auto',
            vertex: {
                module: this.shader,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: this.shader,
                entryPoint: 'fragmentMain',
                targets: [{ format:  navigator.gpu.getPreferredCanvasFormat()}],
            },
        });

        this.sampler = device.createSampler();




        const gaussianBufferValues = new Float32Array([
            0.0675,  0.125,  0.0675, 0.0,
            0.125,  0.250,  0.1250, 0.0,
            0.0675,  0.125,  0.0675 , 0.0
        ]);

        const gaussianBuffer= device.createBuffer({
            label: "Guassian Buffer Kernel",
            size: gaussianBufferValues.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });



        device.queue.writeBuffer(gaussianBuffer, /*bufferOffset=*/0, gaussianBufferValues);


        const kernelOffsetsValue = new Float32Array([
            -1/256, -1/256, 0, 0,
            0     , -1/256, 0, 0,
            1/256 , -1/256, 0, 0,
            -1/256,      0, 0, 0,
            0     ,      0, 0, 0,
            1/256 ,      0, 0, 0,
            -1/256,  1/256, 0, 0,
            0     ,  1/256, 0, 0,
            1/256 ,  1/256, 0, 0,
        ]);


        const kernelOffsetsBuffer= device.createBuffer({
            label: "Guassian Buffer Kernel",
            size: kernelOffsetsValue.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });



        device.queue.writeBuffer(kernelOffsetsBuffer, /*bufferOffset=*/0, kernelOffsetsValue);





        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: {buffer: gaussianBuffer} },
                { binding: 1, resource: {buffer: kernelOffsetsBuffer} },
                { binding: 2, resource: this.sampler },
                { binding: 3, resource: this.inputTexture.createView() },

            ],
        });


        this.renderPassDescriptor = {
            label: 'our basic canvas renderPass',
            colorAttachments: [
                {
                    view:  outputTexture.createView(),
                    clearValue: [0, 0, 0, 1],
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        };



    }


}

export default GuassianLayer;