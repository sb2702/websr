import ComputeLayer from "../base_compute_layer";
import { UNetLayerSpec } from "../../weights";
import { Resolution } from "../../utils";
export interface UNetConvGeometry {
    out: Resolution;
    in: Resolution;
    inStride: number;
    outStride: number;
}
declare class UNetConv extends ComputeLayer {
    label: string;
    private geom;
    constructor(spec: UNetLayerSpec, input: GPUBuffer, output: GPUBuffer, kernels: GPUBuffer, bias: GPUBuffer, geom: UNetConvGeometry, useF16: boolean);
    run(): void;
}
export default UNetConv;
