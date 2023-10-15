import RenderLayer from "../base_render_layer";
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

    lazyLoadSetup(){



    }


    run(){




    }




}

export default Anime4KConv3x4;