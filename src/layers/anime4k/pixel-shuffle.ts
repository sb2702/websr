import RenderLayer from "../base_render_layer";
import ComputeLayer from "../base_compute_layer";


class PixelShuffle2X extends ComputeLayer {

    label = "PixelShuffle2X"


    constructor(inputs: GPUBuffer[], output: GPUBuffer){
        super(inputs, output)

        this.resolution = {
            width: this.resolution.width*2,
            height: this.resolution.height*2
        }

        this.shader = this.createStandardShader(`
        
          @compute @workgroup_size(1, 1) fn main( @builtin(global_invocation_id) id: vec3<u32>) {
          
                let x = id.x;
                let y = id.y;
                
                let i = id.y*512 + x;
                
                let y2 = u32(floor(f32(id.y)/2.0));
                let x2 = u32(floor(f32(id.x)/2.0));
                
                let i2 = y2*256 +  x2;
               
                let x_floor  = u32(fract(f32(x)/2.0)*2.0);
                let y_floor  = u32(fract(f32(y)/2.0)*2.0);
                
                //I don t know, I think this is right? I found this by trial and error
                let c_index: u32 = x_floor + y_floor*2;  

                let value = inputBuffer0[i2][c_index];
                    
                outputBuffer[i] = vec4f(value, 0, 0, 0);
          }
        `);



        this.defaultSetup();
    }








}

export default PixelShuffle2X;