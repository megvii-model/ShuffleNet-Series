# ShuffleNetV2.Large

This repository contains ShuffleNetV2.Large implementation by Pytorch, which is a deeper version of ShuffleNetV2.

## Requirements
Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Usage
Train:
```shell
python train.py --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```
Eval:
```shell
python train.py --eval --eval-resume YOUR_WEIGHT_PATH --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```


## Trained Models
- OneDrive download: [Link](https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo)
- BaiduYun download: [Link](https://pan.baidu.com/s/1EUQVoFPb74yZm0JWHKjFOw) (extract code: mc24)


## Results

| Model                  | FLOPs | #Params   | Top-1     | Top-5 |
| :--------------------- | :---: | :------:  | :---:     | :---: |
| ShuffleNetV2.Large     | 12.7G | 140.7M    | **18.56** | 4.48  |
| SEnet                  | 20.7G |    -      | 18.68     | 4.47  |
