import ComputeLayer from "../base_compute_layer";


class Anime4KConv3x4 extends ComputeLayer {

    label = "Anime4KConv3x4"

    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputBuffer: GPUBuffer, weights: any){
        super(inputTextures, outputBuffer, weights)


                const kernels: number[] = weights.weights;
                const bias: number[] = weights.bias;


                this.createUniform("kernel_offsets", "array<vec4f, 9>");
                this.createUniform("kernels", "array<mat4x4f, 9>");
                this.createUniform("bias", "vec4f");

                // Set up pipeline in Lazy Load

                this.setUniform("kernel_offsets",  new Float32Array([
                    -1,  -1, 0, 0,
                    -1 ,  0, 0, 0,
                    -1 ,  1, 0, 0,
                     0,  -1, 0, 0,
                     0,   0, 0, 0,
                     0,   1, 0, 0,
                     1,  -1, 0, 0,
                     1 ,  0, 0, 0,
                     1 ,  1, 0, 0,
                ]));


                this.setUniform("kernels",  new Float32Array(kernels));
                this.setUniform("bias",  new Float32Array(bias));


    }

    private _builtForExternal: boolean | undefined;

    lazyLoadSetup(){
        const externalTexture = this.inputs[0] instanceof GPUExternalTexture;
        if (this.pipeline && this._builtForExternal === externalTexture) {
           // Same shader; the input texture itself may have changed, so rebind.
           this.bindGroup = this.defaultBindGroup();
           return;
        }
        this._builtForExternal = externalTexture;
    



        const textureLoad = externalTexture ? 'textureLoad(inputTexture0, coord + offset)' :
            'textureLoad(inputTexture0, coord + offset, 0)';

        this.shader = this.createStandardShader(`
        
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups}) fn main( @builtin(global_invocation_id) id: vec3<u32>) {
                if (id.x >= ${this.resolution.width}u || id.y >= ${this.resolution.height}u) { return; }
          
                let x = id.x;
                let y = id.y;
                
                let i = id.y*${this.resolution.width} + x;
                var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                
                let coord = vec2<i32>( i32(x), i32(y));
                      
                 for(var i = 0u; i < 9; i++){
                   let offset = vec2<i32>(kernel_offsets[i].xy);
                   result += kernels[i]*vec4f(${textureLoad});
                 } 
                    
                result += bias;
                
                outputBuffer[i] = result;
          }
        `);


        this.pipeline =  this.device.createComputePipeline(this.defaultPipelineConfig());
        this.bindGroup = this.defaultBindGroup();

    }


}

export default Anime4KConv3x4;