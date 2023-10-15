import {Resolution} from "../utils";
import Layer from "./base_layer";



class RenderLayer extends Layer {

    vertexScale: Resolution;

    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputTexture:GPUTexture, weights?: any){
        super(inputTextures, outputTexture, weights);
        this.vertexScale = this.context.resolution;
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
              vsOutput.tex_coord.y = - 1.0* vsOutput.tex_coord.y  + 1.0;
               vsOutput.tex_coord.x =  vsOutput.tex_coord.x*${this.vertexScale.width};
               vsOutput.tex_coord.y =  vsOutput.tex_coord.y*${this.vertexScale.height};
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

    defaultSetup(){

        this.pipeline = this.device.createRenderPipeline(this.defaultPipelineConfig());

        this.bindGroup = this.defaultBindGroup();

        this.renderPassDescriptor = this.defaultRenderPassDescriptor();
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

    createStandardShader(fragmentShader: string): GPUShaderModule{

        return  this.device.createShaderModule({
            label: `${this.label}-shader`,
            code: `
          
              ${this.defaultVertexShader()}
              
              ${this.fragmentShaderInputs()}
              
              ${fragmentShader}
        `
        });



    }


    fragmentShaderInputs(){

        const inputs = [];

        for (let i=0; i < this.inputTextures.length; i++){

            let type = (this.inputTextures[i] instanceof GPUTexture) ? 'texture_2d<f32>' : 'texture_external';

            inputs.push(`@group(0) @binding(0) var inputTexture${i}: ${type};`)
        }


        this.uniforms.forEach((uniform,i)=>{
            inputs.push(
                `@group(0) @binding(${i+this.inputTextures.length}) var <uniform> ${uniform.name}: ${uniform.type};`,
            )
        });


        return inputs.join('\n');

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

    lazyLoadSetup(){

    }

    run(){

        const encoder = this.device.createCommandEncoder({label: this.label});

        if(!this.pipeline) this.lazyLoadSetup();

        const pass = encoder.beginRenderPass(this.renderPassDescriptor);


        pass.setPipeline(this.pipeline);


        if(this.hasExternalTexture()){
            this.bindGroup = this.defaultBindGroup();
        }

        if(this.bindGroup) {
            pass.setBindGroup(0, this.bindGroup);
        }

        pass.draw(6);  // call our vertex shader 6 times
        pass.end();

        this.device.queue.submit([encoder.finish()]);


    }


}

export default RenderLayer;