import ComputeLayer from "../base_compute_layer";


class Anime4KConcat2 extends ComputeLayer {

    label = "Anime4KConcat2"

    constructor(inputs: GPUBuffer[], outputBuffer: GPUBuffer, weights: any){
        super(inputs, outputBuffer, weights)

        this.createUniform("bias", "vec4f");
        const bias: number[] = weights.bias;


        this.shader = this.createStandardShader(`
        
          @compute @workgroup_size(${this.num_work_groups}, ${this.num_work_groups}) fn main( @builtin(global_invocation_id) id: vec3<u32>) {
          
                let x = id.x;
                let y = id.y;
                
                let i = id.y*${this.resolution.width} + x;
                var result  = vec4f(0.0, 0.0, 0.0, 0.0);
                
                let coord = vec2<i32>( i32(x), i32(y));
               
                let buff_ind = coord.y*${this.resolution.width} + coord.x;
               
                outputBuffer[buff_ind] = inputBuffer0[buff_ind] + inputBuffer1[buff_ind] + bias;
          }
        `);


        this.setUniform("bias",  new Float32Array(bias));

        this.defaultSetup();


    }




}

export default Anime4KConcat2;