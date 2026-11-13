import NeuralNetwork from "./networks/base_network";
import { Resolution, MediaSource } from "./utils";
export default class WebSRRenderer {
    private context;
    private network;
    private tiles;
    private nominalTileResolution;
    private tilePadding;
    private downscaleLayer?;
    private resampleLayer?;
    source?: MediaSource;
    active: boolean;
    vfc: number;
    constructor(network: NeuralNetwork, nominalTileResolution?: Resolution, tilePadding?: number, source?: MediaSource);
    switchNetwork(network: NeuralNetwork): void;
    start(): Promise<void>;
    stop(): Promise<void>;
    renderStep(): Promise<void>;
    private needsDownscale;
    private isTiling;
    private calculateTileGrid;
    private materializeInput;
    render(source?: MediaSource, onTileProgress?: (done: number, total: number) => void): Promise<void>;
    private resampleIfNeeded;
    private resampler;
    private renderSingle;
    private renderTiled;
    destroy(): Promise<void>;
}
