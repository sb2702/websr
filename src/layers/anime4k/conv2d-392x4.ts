import ComputeLayer from "../base_compute_layer";


class Anime4KConv392x4 extends ComputeLayer {

    label = "Anime4KConv392x4"

    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any, index: number){
        super(inputs, outputBuffer, weights)


        const kernels: number[] = weights.weights;


        this.createUniform("kernels", "array<mat4x4f, 98>");

        let read_buffers = '';

        for (let i=0; i < 7; i++){


            if(index == 0){

                read_buffers +=`
                let pixel_val${i} = inputBuffer${i}[buff_ind];
                result += kernels[${14*i}]*max(pixel_val${i}, vec4f(0.0));
                result += kernels[${14*i+7}]*max(-1.0*pixel_val${i}, vec4f(0.0));
            `;
            } else if(index == 1){

                read_buffers +=`
                let pixel_val${i} = inputBuffer${i}[buff_ind];
                result += kernels[${14*i+1}]*max(pixel_val${i}, vec4f(0.0));
                result += kernels[${14*i+8}]*max(-1.0*pixel_val${i}, vec4f(0.0));`;

            } else if(index == 2){

                read_buffers +=`
                let pixel_val${i} = inputBuffer${i}[buff_ind];
                result += kernels[${14*i+2}]*max(pixel_val${i}, vec4f(0.0));
                result += kernels[${14*i+9}]*max(-1.0*pixel_val${i}, vec4f(0.0));`;

            } else if(index == 3){

                read_buffers +=`
                let pixel_val${i} = inputBuffer${i}[buff_ind];
                result += kernels[${14*i+3}]*max(pixel_val${i}, vec4f(0.0));
                result += kernels[${14*i+10}]*max(-1.0*pixel_val${i}, vec4f(0.0));`;

            } else if(index == 4){

                read_buffers +=`
                let pixel_val${i} = inputBuffer${i}[buff_ind];
                result += kernels[${14*i+4}]*max(pixel_val${i}, vec4f(0.0));
                result += kernels[${14*i+11}]*max(-1.0*pixel_val${i}, vec4f(0.0));`;

            } else if(index == 5){

                read_buffers +=`
                let pixel_val${i} = inputBuffer${i}[buff_ind];
                result += kernels[${14*i+5}]*max(pixel_val${i}, vec4f(0.0));
                result += kernels[${14*i+12}]*max(-1.0*pixel_val${i}, vec4f(0.0));`;

            } else if(index == 6){

                read_buffers +=`
                let pixel_val${i} = inputBuffer${i}[buff_ind];
                result += kernels[${14*i+6}]*max(pixel_val${i}, vec4f(0.0));
                result += kernels[${14*i+13}]*max(-1.0*pixel_val${i}, vec4f(0.0));`;
            }

        }


        this.shader = this.createStandardShader(`
        
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups}) fn main( @builtin(global_invocation_id) id: vec3<u32>) {
                if (id.x >= ${this.resolution.width}u || id.y >= ${this.resolution.height}u) { return; }
          
                let x = id.x;
                let y = id.y;
                
                let i = id.y*${this.resolution.width} + x;
                var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                
                let coord = vec2<i32>( i32(x), i32(y));
               
                let buff_ind = coord.y*${this.resolution.width} + coord.x;
                ${read_buffers}
                
                outputBuffer[i] = result;
          }
        `);

        this.setUniform("kernels",  new Float32Array(kernels));


        this.defaultSetup();

    }

    defaultPipelineConfig(): GPUComputePipelineDescriptor{

        return {
            label: `${this.label}-pipeline`,
            layout: 'auto',
            compute: {
                module: this.shader,
                entryPoint: 'main',
            },
        }

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
            layout: this.pipeline.getBindGroupLayout(0),
            entries
        });
    }



}

export default Anime4KConv392x4;