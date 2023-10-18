import Layer from "./base_layer";



class ComputeLayer extends Layer {

    pipeline: GPUComputePipeline;
    num_work_groups: number;

    constructor(inputTextures: (GPUTexture|GPUExternalTexture|GPUBuffer)[], outputBuffer:GPUBuffer, weights?: any){
       super(inputTextures, outputBuffer, weights);
       this.num_work_groups = 8;
    }



    createStandardShader(computeShader: string): GPUShaderModule{

        return  this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `
              
              ${this.computeShaderInputs()}
              
              ${computeShader}
        `
        });


    }

    computeShaderInputs(){

        const inputs = [];

        for (let i=0; i < this.inputs.length; i++){


            if(this.inputs[i] instanceof GPUTexture){
                inputs.push(`@group(0) @binding(${i}) var inputTexture${i}: texture_2d<f32>;`)
            } else if (this.inputs[i] instanceof GPUExternalTexture){
                inputs.push(`@group(0) @binding(${i}) var inputTexture${i}: texture_external;`)
            } else if(this.inputs[i] instanceof GPUBuffer) {
                inputs.push(`@group(0) @binding(${i}) var<storage, read_write> inputBuffer${i}: array<vec4f>;`)
            } else {
                throw new Error("Input is undefined or non of the correct input type");
            }
        }


        this.uniforms.forEach((uniform,i)=>{
            inputs.push(
                `@group(0) @binding(${i+this.inputs.length}) var <uniform> ${uniform.name}: ${uniform.type};`,
            )
        });

        inputs.push(`@group(0) @binding(${this.inputs.length + this.uniforms.length}) var <storage, read_write> outputBuffer: array<vec4f>;`)


        return inputs.join('\n');

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

    defaultSetup(){

        this.pipeline = this.device.createComputePipeline(this.defaultPipelineConfig());

        this.bindGroup = this.defaultBindGroup();
    }



    lazyLoadSetup(){

    }

    run(){


        const encoder = this.device.createCommandEncoder({label: this.label});

        if(!this.pipeline) this.lazyLoadSetup();

        const pass = encoder.beginComputePass({label: this.label});

        pass.setPipeline(this.pipeline);


        if(this.hasExternalTexture()){
            this.bindGroup = this.defaultBindGroup();
        }

        if(this.bindGroup) {
            pass.setBindGroup(0, this.bindGroup);
        }

        // Dividing into work groups speeds up inference. If width or height aren't cleandly divided by work groups, we round to the nearest multiple of work-groups
        // Physically, this means shaving a few pixels (up to num_work_groups-1) off the bottom and right edges of the canvas but users shouldn't notice?
        
        pass.dispatchWorkgroups(Math.floor(this.resolution.width/this.num_work_groups), Math.floor(this.resolution.height/this.num_work_groups));

        pass.end();

        this.device.queue.submit([encoder.finish()]);


    }


}

export default ComputeLayer;