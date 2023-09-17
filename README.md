# WebSR

A free and open-source Javascript library / SDK for running practical, real-time AI Upscaling of videos and images on the web.

<img width="640" title="Demo clip from Hero by the Blender Foundation, licensed under Creative Commons 4.0 Attribution licensed" alt="Demo clip from Hero by the Blender Foundation, licensed under Creative Commons 4.0 Attribution licensed" src="https://github.com/sb2702/websr/assets/5678502/82e6e764-89f4-4c8d-b8f6-dcd43d43c0f3">


[Check out the demo!](https://websr.dev/demo/)


**Warning**
This project is very new and still in development, and shouldn't be used (yet) in production until stable. Expect API changes between versions and bugs


### Installation

**npm**

```javascript
npm install @websr/websr
```


**yarn**

```javascript
yarn add @websr/websr
```


### Quick Start

```javascript
import WebSR from  '@websr/websr';

const gpu = await WebSR.initWebGPU();

if(!gpu) return console.log("Browser/device doesn't support WebGPU");

const websr = new WebSR({
    video: // An HTML Video Element
    network_name: "anime4k/cnn-2x-s",
    weights: await (await fetch('./cnn-2x-s.json')).json() //found in weights/anime4k folder
    gpu,
    canvas: //A canvas, with 2x the width and height of your input video
});

await websr.start();   // Play the video

```

### Why WebSR

[Super resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging) is a technique which uses Neural Networks to "reconstruct" a high-resolution image from a low resolution image. Practically speaking, it provides better visual results than traditional upscaling algorithms like Bicubic, which is what devices and browsers usually use by default, at the cost of significantly more computation.

A general rule of thumb is that bigger super-resolution networks (which use more computation) generally provide better results.

<img src="https://github.com/sb2702/websr/assets/5678502/6e040e84-cf6d-4cf0-a45c-638b41d011ec" width="400" />

This is why commercial AI upscaling software like [Topaz Labs](https://www.topazlabs.com/) can produce excellent quality results, though it can take 10 seconds or more on a dedicated GPU to upscale a single image. 
If the goal is to run super resolution on a video at 30 frames per second on mid-tier integrated graphcs cards in the browser, you need networks that run thousands of times faster. Computation is a huge challenge and a big reason that super resolution hasn't been more popular in web video. 

WebSR is an attempt to take on this challenge, and make super resolution practical in the browser. By taking advantage of the fact that super-resolution has proved relatively better at improving the video quality of Anime, Cartoons and Screen-sharing content, as well the faster graphics computation enabled by [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API), WebSR should be just-about practical enough for production video and WebRTC applicaitons in cases where bandwidth or video quality is an issue.


### How it works

WebSR uses hand-written WebGPU shaders to implement Convolutional Neural Networks (like [this one](https://github.com/sb2702/websr/blob/main/src/networks/anime4k/cnn-2x-s.ts) from [Anime4K](https://github.com/bloc97/Anime4K)).

When you create a `WebSR`instance and call `websr.start()`, that starts a render-loop which will take in video frames, one by one, and upscale them using the Neural Network, before painting the result to a `<canvas>`.

While there are many super-resolution networks and algorithms available, for a proof of concept, WebSR implements networks from [Anime4K](https://github.com/bloc97/Anime4K) as 

1.  It was designed for real time video
2.  Already has GPU shader written in the similar `glsl` language
3.  Already has wide community adoption.

This repo provides the [default production weights](https://github.com/sb2702/websr/blob/main/weights/anime4k/cnn-2x-s.json) asociated with the `cnn-2x-s` network directly from Anime4K. 

You can custom train or fine-tune the network and generate your own weights (details to follow).

The plan is to add additional neural networks, whether from other open-source projects or building custom networks specifically for WebSR.

### Acknowledgements

1. [Anime4K](https://github.com/bloc97/Anime4K) for an excellent project
2. [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) for enabling compute-intensive projects like this
3. [Blender foundation](https://studio.blender.org/films/hero/) for use of their hero movie in WebSR demos

### Roadmap

**Very soon**
- [ ] Performance tests
- [ ] Provide notebook and scripts for writing and training custom networks
- [ ] Automated builds & tests

**Soon-ish**
- [ ] Re-implement with compute shaders for better performance
- [ ] Provide lower level control for developers

**At some point**
- [ ] Add more upscaling networks
- [ ] Write a chrome plug-in optimizing to implement this easily
- [ ] Build a static website to upscale videos on your device for free
