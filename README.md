# SpineLoc: Spine X-ray Localization

Multi-task deep learning model for spine X-ray coordinate regression.

## Features
- Dense coordinate prediction
- View classification (frontal/lateral)
- Laterality detection (facing left/right)
- Uncertainty estimation

# Development
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) was loosely used as the baseline.

## Setup
- Install [pixi](https://pixi.prefix.dev/dev/)
- cd to the project directory
- Run `pixi install`

## Train
- cpu: `pixi run train`
- gpu: `pixi run -e gpu train trainer=gpu data=gpu`

## Validate and Predict
Calculate classification accuracy using images in `data/radiopaedia/test`
```bash
pixi run validate paths.output_dir=/tmp/validation safetensors_path=path/to/trained.safetensors
```

Predict on new images
```bash
pixi run -e gpu predict image_dir=/tmp/new/inputs paths.output_dir=/tmp/new/outputs safetensors_path=path/to/trained.safetensors
```
Optionally, add `-e gpu` to run in a gpu environment

## Software Test
- Run `pixi run test`
  - Run `pixel run text_visual -vs` to run visualization test

## TODOs
- [ ] Add oblique view?