import Layer from "./base_layer";


class RGB2YUV extends Layer {



    constructor(device, inputTexture, outputTexture){
        super(device, inputTexture, outputTexture)

        this.shader = device.createShaderModule({
            label: 'RGB2YUV',

            code: `
            
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
              vsOutput.tex_coord = xy;
              return vsOutput;
            }
    
          @group(0) @binding(0) var<uniform> rgb2yuv: mat3x3f;
          @group(0) @binding(1) var ourSampler: sampler;
          @group(0) @binding(2) var ourTexture: texture_2d<f32>;
    
          @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
          
            let color = textureSample(ourTexture, ourSampler, input.tex_coord);
            
            let yuv = rgb2yuv*color.xyz;
      
            return vec4f(yuv, 1.0);
          }
              
            `
        });



        this.pipeline =  device.createRenderPipeline({
            label: 'RGB 2 YUV Pipeline',
            layout: 'auto',
            vertex: {
                module: this.shader,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: this.shader,
                entryPoint: 'fragmentMain',
                targets: [{ format:  outputTexture.format}],
            },
        });

        this.sampler = device.createSampler();




        const rgb2yuv = new Float32Array([
            0.299, -0.1473, 0.615, 1.0,
            0.587, -.2886, -.51499, 1.0,
            0.114,  0.436, -.1001, 1.0
        ]);

        const rgb2yuvBuffer= device.createBuffer({
            label: "RGB to YUV Conversion Matrix Buffer",
            size: rgb2yuv.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(rgb2yuvBuffer, /*bufferOffset=*/0, rgb2yuv);





        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: {buffer: rgb2yuvBuffer} },
                { binding: 1, resource: this.sampler },
                { binding: 2, resource: inputTexture.createView() },

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

export default RGB2YUV;