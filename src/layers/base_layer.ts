import WebGPUContext from "../context";
import {Resolution} from "../utils";

interface Uniform {
    name: string,
    type: string
}

class Layer {

    device: GPUDevice;
    shader: GPUShaderModule;
    pipeline: GPURenderPipeline;
    sampler: GPUSampler;
    renderPassDescriptor: GPURenderPassDescriptor;
    bindGroup: GPUBindGroup | null;
    label: string;
    inputTextures: (GPUTexture|GPUExternalTexture)[];
    outputTexture: GPUTexture;
    uniforms: Uniform[];
    weights: any;
    buffers: Record<string, GPUBuffer>;
    context: WebGPUContext;
    resolution: Resolution;

    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputTexture:GPUTexture, weights?: any){

        this.context = globalThis.context;
        this.device = this.context.device;
        this.resolution = this.context.resolution;
        this.inputTextures = inputTextures;
        this.outputTexture = outputTexture;
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

        this.inputTextures.forEach(function (texture, i) {

            if(texture instanceof GPUExternalTexture){
                entries.push({ binding: i, resource: texture})
            } else if (texture instanceof GPUTexture){
                entries.push({ binding: i, resource: texture.createView()})
            }

        });


        this.uniforms.forEach((uniform, i)=>{
            entries.push(
                {
                    binding: i+this.inputTextures.length,
                    resource: {
                        buffer: this.buffers[uniform.name]
                    }
                }
            )
        });

        if(entries.length === 0) return  null;

         return this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries
        });
    }

    hasExternalTexture(){

        for (const texture of this.inputTextures){
            if(texture instanceof GPUExternalTexture) return true;
        }

        return  false;
    }

    lazyLoadSetup(){}

    run(){}


}

export default Layer;