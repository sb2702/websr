import ComputeLayer from "../base_compute_layer";


class Anime4KConv16x4 extends ComputeLayer {

    label = "Anime4KConv16x4"


    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any){
        super(inputs, outputBuffer, weights)


        const kernels: number[] = weights.weights;
        const bias: number[] = weights.bias;

        this.createUniform("kernel_offsets", "array<vec4f, 9>");
        this.createUniform("kernels", "array<mat4x4f, 36>");
        this.createUniform("bias", "vec4f");

        this.shader = this.createStandardShader(`
        
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups}) fn main( @builtin(global_invocation_id) id: vec3<u32>) {
          
                let x = id.x;
                let y = id.y;
                
                let i = id.y*${this.resolution.width} + x;
                var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                
                let coord = vec2<i32>( i32(x), i32(y));
                      
                 for(var i = 0u; i < 9; i++){
                   let pixel_loc = coord + vec2<i32>(kernel_offsets[i].xy);
                   let buff_ind = pixel_loc.y*${this.resolution.width} + pixel_loc.x;
                   
                   let pix_val0 = inputBuffer0[buff_ind];
                   let pix_val1 = inputBuffer1[buff_ind];
                  
                   result += kernels[i]*max(pix_val0, vec4f(0.0));
                   result += kernels[i+9]*max(pix_val1, vec4f(0.0));
                   result += kernels[i+18]*max(-1.0*pix_val0, vec4f(0.0));
                   result += kernels[i+27]*max(-1.0*pix_val1, vec4f(0.0));
                 } 
                 

                    
                result += bias;
                
                outputBuffer[i] = result;
          }
        `);

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


        this.defaultSetup();

    }


}

export default Anime4KConv16x4;