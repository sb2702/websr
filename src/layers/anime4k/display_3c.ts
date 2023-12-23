import RenderLayer from "../base_render_layer";


class DisplayLayer3C extends RenderLayer {

    label = "DisplayLayer3C"
    private bind_group_layout: GPUBindGroupLayout;
    private pipeline_layout: GPUPipelineLayout;

    constructor(inputs: (GPUTexture|GPUExternalTexture|GPUBuffer)[], output: GPUTexture){
        super(inputs, output)

        this.vertexScale = {
            width: 1,
            height: 1
        };


        this.sampler =  this.device.createSampler({
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
        });


    }


    lazyLoadSetup(){


        const externalTexture = this.inputs[1] instanceof GPUExternalTexture;

        const textureLoad = externalTexture ? 'textureSampleBaseClampToEdge(inputTexture, ourSampler, input.tex_coord)' :
            'textureSample(inputTexture, ourSampler, input.tex_coord)';

        this.shader = this.device.createShaderModule({
                label: `${this.label}-shader`,
                code: `
                
                   ${this.defaultVertexShader()}
                   @group(0) @binding(0) var<storage, read_write> inputBuffer0: array<vec4f>;
                   @group(0) @binding(1) var<storage, read_write> inputBuffer1: array<vec4f>;
                   @group(0) @binding(2) var<storage, read_write> inputBuffer2: array<vec4f>;
                   @group(0) @binding(3) var inputTexture: ${externalTexture?  'texture_external': 'texture_2d<f32>'};
                   @group(0) @binding(4) var ourSampler: sampler;
                  
                   @fragment fn fragmentMain(input: VertexShaderOutput) -> @location(0) vec4f {
                      
                        let x = ${this.resolution.width}.0*(input.tex_coord.x);
                        let y = ${this.resolution.height}.0*(input.tex_coord.y);
                        
                        let y2 = u32(floor(y));
                        let x2 = u32(floor(x));
                        
                        let i = y2*${Math.floor(this.resolution.width)} +  x2;
                       
                        let x_floor  = u32(fract(x)*2.0);
                        let y_floor  = u32(fract(y)*2.0);
                        
                        //I don t know, I think this is right? I found this by trial and error
                        let c_index: u32 = x_floor + y_floor*2;  
        
                        let value = inputBuffer0[i][c_index];
                        let value1 = inputBuffer1[i][c_index];
                        let value2 = inputBuffer2[i][c_index];
                        
                        let bicubic = ${textureLoad};
                        
                        return bicubic*0.01 + vec4f(value, 0, 0, 0);
                    
                      }            
            `
            });

        this.pipeline =  this.device.createRenderPipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();
        this.renderPassDescriptor = this.defaultRenderPassDescriptor();

    }


    createLayout(){

        this.bind_group_layout =  this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture:{}

                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {}

                }
            ]
        });

        this.pipeline_layout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bind_group_layout]
        })

    }


    defaultPipelineConfig(): GPURenderPipelineDescriptor {

        this.createLayout();

        return {
            label: `${this.label}-pipeline`,
            layout: this.pipeline_layout,
            vertex: {
                module: this.shader,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: this.shader,
                entryPoint: 'fragmentMain',
                targets: [{format: this.output.format}],
            },
        }

    }


    defaultBindGroup(): GPUBindGroup {


        const entries: any[]  = [];

        this.inputs.forEach(function (input, i) {
            if(input instanceof GPUExternalTexture){
                entries.push({ binding: i, resource: input})
            } else if (input instanceof GPUTexture){
                entries.push({ binding: i, resource: input.createView()})
            } else  if(input instanceof GPUBuffer){
                entries.push({ binding: i, resource: {buffer: input}})
            }

        });


        entries.push({ binding: this.inputs.length, resource: this.sampler })


        return this.device.createBindGroup({
            layout: this.bind_group_layout,
            entries
        });


    }

    setOutput(outputTexture: GPUTexture){

        this.output = outputTexture;

        this.renderPassDescriptor = this.defaultRenderPassDescriptor();
    }



}

export default DisplayLayer3C;