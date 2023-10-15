import Layer from "./base_layer";



class ComputeLayer extends Layer {


    constructor(inputTextures: (GPUTexture|GPUExternalTexture)[], outputTexture:GPUTexture, weights?: any){
       super(inputTextures, outputTexture, weights);
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




    }


}

export default ComputeLayer;