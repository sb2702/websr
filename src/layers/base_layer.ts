import WebGPUContext from "../context";
import {Resolution} from "../utils";

interface Uniform {
    name: string,
    type: string
}

class Layer {

    device: GPUDevice;
    shader: GPUShaderModule;
    pipeline: GPURenderPipeline | GPUComputePipeline;
    sampler: GPUSampler;
    renderPassDescriptor: GPURenderPassDescriptor;
    bindGroup: GPUBindGroup | null;
    label: string;
    inputs: (GPUTexture|GPUExternalTexture|GPUBuffer)[];
    output: GPUTexture|GPUBuffer;
    uniforms: Uniform[];
    weights: any;
    buffers: Record<string, GPUBuffer>;
    context: WebGPUContext;
    resolution: Resolution;

    constructor(inputs: (GPUTexture|GPUExternalTexture|GPUBuffer)[], output:GPUTexture|GPUBuffer, weights?: any){

        this.context = globalThis.context;
        this.device = this.context.device;
        this.resolution = this.context.resolution;
        this.inputs = inputs;
        this.output = output;
        this.uniforms =  [];
        this.buffers = {};
        this.weights = weights;
    }




    createUniform(name:string, type:string){

        this.uniforms.push({name, type});

    }

    setUniform(name: string, value: Float32Array)  {

        const buffer= this.device.createBuffer({
            label: `layer-${this.label}-buffer-${name}`,
            size: value.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(buffer, /*bufferOffset=*/0, value);

        this.buffers[name] = buffer;

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

    hasExternalTexture(){

        for (const input of this.inputs){
            if(input instanceof GPUExternalTexture) return true;
        }

        return  false;
    }

    lazyLoadSetup(){}

    run(){}


}

export default Layer;