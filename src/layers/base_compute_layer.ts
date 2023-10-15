import Layer from "./base_layer";



class ComputeLayer extends Layer {

    pipeline: GPUComputePipeline;

    constructor(inputTextures: (GPUTexture|GPUExternalTexture|GPUBuffer)[], outputBuffer:GPUBuffer, weights?: any){
       super(inputTextures, outputBuffer, weights);
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
            } else if (this.inputs[i] instanceof GPUTexture){
                inputs.push(`@group(0) @binding(${i}) var inputTexture${i}: texture_external;`)
            } else if(this.inputs[i] instanceof GPUBuffer) {
                inputs.push(`@group(0) @binding(${i}) var<storage, read_write> inputBuffer${i}: array<vec4f>`)
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

        // We should probably play around with width/height to make it more efficient
        pass.dispatchWorkgroups(this.resolution.width, this.resolution.height);

        pass.end();

        this.device.queue.submit([encoder.finish()]);


    }


}

export default ComputeLayer;