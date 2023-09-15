# WebSR

A free and open-source Javascript library / SDK for running practical, real-time AI Upscaling of videos and images on the web.

<img width="640" title="Demo clip from Hero by the Blender Foundation, licensed under Creative Commons 4.0 Attribution licensed" alt="Demo clip from Hero by the Blender Foundation, licensed under Creative Commons 4.0 Attribution licensed" src="https://github.com/sb2702/websr/assets/5678502/82e6e764-89f4-4c8d-b8f6-dcd43d43c0f3">

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


### Usage

```javascript
import WebSR from  '@websr/websr';


WebSR.initWebGPU().then(async function (gpu) {

    if(!gpu) return console.log("Unable to load WebGPU");
    
    const weights = await (await fetch('./anime4k-cnn-2x-s.json')).json()

    const websr = new WebSR({
        video: // An HTML Video Element
        network_name: "anime4k/cnn-2x-s",
        weights: // The contents of weights/anime4k/cnn-2x-s.json, as a javascript object
        gpu,
        canvas: //A canvas, with 2x the width and height of your input video
        });

    // Play the video
    await websr.start();
    
    // Stop the video
    // await webs.stop()
    
    // Destroy WebSR and all it's resources
    
    // await websr.destroy()

});

```




**Coming very soon**
- [ ] Performance tests
- [ ] Provide notebook and scripts for writing and training custom networks
- [ ] Automated builds & tests

**Coming soon-ish**
- [ ] Re-implement with compute shaders for better performance
- [ ] Provide lower level control for developers


**Want to get around to**
- [ ] Add more upscaling networks
- [ ] Write a chrome plug-in optimizing to implement this easily
- [ ] Build a static website to upscale videos on your device for free
