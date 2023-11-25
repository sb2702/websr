const video  = document.getElementsByClassName("html5-main-video")[0];
const container = document.getElementsByClassName("html5-video-container")[0];
const positionInfo = container.parentNode.getBoundingClientRect();
const menu = document.getElementsByClassName("ytp-panel-menu")[0];
container.style.height =  positionInfo.height + "px";
// const settings = document.getElementsByClassName("ytp-settings-button")[0];

import vectorlyUpscaler from '../common/upscaler'

console.log('input')

const checkbox = document.createElement("input");
checkbox.type = "checkbox";

var upscaler;



addMenu();

let checkedBeforeDisable = false


function addMenu() {
    const upscalerMenuItem  = document.createElement("div");
    upscalerMenuItem.className = "ytp-menuitem";

    const upscalerMenuIcon = document.createElement("div");
    upscalerMenuIcon.className = "ytp-menuitem-icon";

    const upscalerMenuLabel = document.createElement("div");
    upscalerMenuLabel.className = "ytp-menuitem-label";

    const upscalerMenuContent = document.createElement("div");
    upscalerMenuContent.className = "ytp-menuitem-content";

    upscalerMenuLabel.innerHTML = "AI Upscale";

    upscalerMenuItem.appendChild(upscalerMenuIcon);
    upscalerMenuItem.appendChild(upscalerMenuLabel);
    upscalerMenuItem.appendChild(upscalerMenuContent);

    menu.appendChild(upscalerMenuItem);

    upscalerMenuContent.appendChild(checkbox);

    checkbox.onchange = function () {
        console.log('Checkbox changed onchanged', checkbox.checked)
        if(checkbox.checked)  enable();
        else disable();
    };
}


function inputChangePreHookFunc (data, instance) {
  const { after } = data
  console.log('In inputChangePreHookFunc', data, instance)
  if(Math.min(after.width, after.height) >= 720) {
      if (upscaler) disable()
      checkedBeforeDisable = checkbox.checked
      checkbox.checked = false;
      checkbox.disabled = true;
      return false
  } else {
      enable()
      if (checkbox.disabled !== false) {
        checkbox.disabled = false;
        checkbox.checked = checkedBeforeDisable
      }
      return true
  }
}



function setupUpscaler() {
    upscaler=  new vectorlyUpscaler(video, {
      container: document.getElementsByClassName('html5-video-player')[0],
      inputChangePreHook: inputChangePreHookFunc
    });
    upscaler.enable();
}


function enable() {
    if(!upscaler)setupUpscaler();
    else upscaler.enable();
}

function disable() {
    upscaler.disable();
}
