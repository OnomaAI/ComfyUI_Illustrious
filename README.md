# ComfyUI_Illustrious

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension for generating image via NovelAI API.

## Installation

- `git clone https://github.com/OnomaAI/ComfyUI_Illustrious.git` into the `custom_nodes` directory.
- or 'Install via Git URL' from [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Setting up Illustrious account

Before using the nodes, you should set ILLUSTRIOUS_API_KEY on `custom_nodes/ComfyUI_illustrious/.env` file.

```
ILLUSTRIOUS_API_KEY=<ACCESS_TOKEN>
```

You can get persistent API token by **My Profile > API Key Section > Create API Key** on Illustrious webpage.

Otherwise, you can get access token which is valid for 30 days using [Illustrious-API](https://www.illustrious-xl.ai/).

## Usage

Currently, only  Text-to-Image is supported, but we plan to support more features in the future.

### Text-to-Image

Simply connect `GenerateILL` node and `Save Image` node.

![[generate]([https://github.com/OnomaAI/ComfyUI_Illustrious/assets/illustrious_simple_generation.jpg](https://github.com/OnomaAI/ComfyUI_Illustrious/blob/main/assets/illustrious_simple_generation.jpg?raw=true))](https://github.com/OnomaAI/ComfyUI_Illustrious/blob/main/assets/illustrious_simple_generation.jpg?raw=true)
