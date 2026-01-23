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

## Software Test
- Run `pixi run test`
  - Run `pixel run text_visual -vs` to run visualization test