# ShuffleNetV2+

This repository contains ShuffleNetV2+ implementation by Pytorch, which is a strengthen version of ShuffleNetV2 by adding Hard-Swish, Hard-Sigmoid and [SE](https://arxiv.org/abs/1709.01507) modules.

The following is a comparison with MobileNetV3 in [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244).


## Requirements
Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Usage
Train:
```shell
python train.py --model-size Large --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```
Eval:
```shell
python train.py --eval --eval-resume YOUR_WEIGHT_PATH --model-size Large --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```

## Results

|    Model                 |  FLOPs    |   #Params |   Top-1   |   Top-5   |
|:------------------------:|:---------:|:---------:|:---------:|:---------:|
ShuffleNetV2+ Small        |   156M     |	5.1M    |      25.9    |       8.3    |
ShuffleNetV2+ Medium       |   222M     |	5.6M    |      24.3    |       7.4    |
ShuffleNetV2+ Large        |   360M     |	6.7M    |      22.9    |       6.7   |
