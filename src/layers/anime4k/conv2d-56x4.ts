import ComputeLayer from "../base_compute_layer";


class Anime4KConv56x4 extends ComputeLayer {

    label = "Anime4KConv56x4"
    private bind_group_layout: GPUBindGroupLayout;
    private pipeline_layout: GPUPipelineLayout;

    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any){
        super(inputs, outputBuffer, weights)


        const kernels: number[] = weights.weights;
        const bias: number[] = weights.bias;


        this.createUniform("kernels", "array<mat4x4f, 14>");
        this.createUniform("bias", "vec4f");

        this.shader = this.createStandardShader(`
        
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups}) fn main( @builtin(global_invocation_id) id: vec3<u32>) {
          
                let x = id.x;
                let y = id.y;
                
                let i = id.y*${this.resolution.width} + x;
                var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                
                let coord = vec2<i32>( i32(x), i32(y));
                      
                 for(var i = 0u; i < 7; i++){
                   let buff_ind = coord.y*${this.resolution.width} + coord.x;
                   let pixel_val = inputBuffer0[buff_ind];
                   result += kernels[i]*max(pixel_val, vec4f(0.0));
                   result += kernels[i]*max(-1.0*pixel_val, vec4f(0.0));
                 } 
                    
                result += bias;
                
                outputBuffer[i] = result;
          }
        `);



        this.setUniform("kernels",  new Float32Array(kernels));
        this.setUniform("bias",  new Float32Array(bias));


        this.defaultSetup();

    }

    defaultPipelineConfig(): GPUComputePipelineDescriptor{

        this.createLayout();

        return {
            label: `${this.label}-pipeline`,
            layout: this.pipeline_layout,
            compute: {
                module: this.shader,
                entryPoint: 'main',
            },
        }

    }

    createLayout(){
        this.bind_group_layout =  this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
                {
                    binding: 7,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }

                },
                {
                    binding: 8,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform"
                    }

                },
                {
                    binding: 9,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }

                },
            ]
        });

        this.pipeline_layout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bind_group_layout]
        })

    }


    defaultBindGroup(){

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


        this.uniforms.forEach((uniform, i)=>{
            entries.push(
                {
                    binding: i+this.inputs.length,
                    resource: {
                        buffer: this.buffers[uniform.name]
                    }
                }
            )
        });

        if(this.output instanceof GPUBuffer){

            entries.push(
                {
                    binding: this.inputs.length + this.uniforms.length,
                    resource: {
                        buffer: this.output
                    }
                }
            )

        }
        
        if(entries.length === 0) return  null;

        return this.device.createBindGroup({
            layout: this.bind_group_layout,
            entries
        });
    }



}

export default Anime4KConv56x4;