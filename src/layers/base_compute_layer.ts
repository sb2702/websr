import Layer from "./base_layer";



class ComputeLayer extends Layer {


    constructor(inputTextures: (GPUTexture|GPUExternalTexture|GPUBuffer)[], outputBuffer:GPUBuffer, weights?: any){
       super(inputTextures, outputBuffer, weights);
    }



    lazyLoadSetup(){

    }

    run(){




    }


}

export default ComputeLayer;