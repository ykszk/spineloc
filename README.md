# SpineLoc: Spine X-ray Localization

Multi-task deep learning model for spine X-ray coordinate regression.

## Features
- Dense coordinate prediction
- View classification (frontal/lateral)
- Laterality detection (facing left/right)
- Uncertainty estimation

# Materials and Methods
The training dataset consists of whole-spine radiographs of both frontal and lateral views. Each image in the training dataset is annotated with vertebral corners. See `data/radiopaedia/raw` for examples.

## Coordinate system
Anatomical coordinate system was defined per image as below:
- x and y axes: image's x and y axes
- unit size of x: average width of C7 to L5 vertebrae
- unit size of y: average height of C7 to L5 vertebrae
- origin: midpoint of C7 and L4 vertebrae

## Data generator
- During training, randomly cropped images were generated from each training image, and their anatomical coordinates relative to the original image were calculated.
- Random image flipping was used to generate right-facing lateral images
- The model was trained with two loss functions:
  - classification: frontal, lateral-left, or lateral-right
  - regression: anatomical coordinates

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