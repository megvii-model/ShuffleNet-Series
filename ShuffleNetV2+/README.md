# ShuffleNetV2+

This repository contains ShuffleNetV2+ implementation by Pytorch, which is a strengthen version of ShuffleNetV2 by adding Hard-Swish, Hard-Sigmoid and [SE](https://arxiv.org/abs/1709.01507) modules.



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

The following is a comparison with MobileNetV3 in [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244).

|    Model                 |  FLOPs    |   #Params |   Top-1   |   Top-5   |
|:------------------------:|:---------:|:---------:|:---------:|:---------:|
ShuffleNetV2+ Large        |   360M     |	6.7M    |      **22.9**    |       6.7   |
MobileNetV3 Large 224/1.25       |   356M     |	7.5M    |      23.4    |       /   |
ShuffleNetV2+ Medium       |   222M     |	5.6M    |      **24.3**    |       7.4    |
MobileNetV3 Large 224/1.0       |   217M     |	5.4M    |      24.8    |       /    |
ShuffleNetV2+ Small        |   156M     |	5.1M    |      **25.9**    |       8.3    |
MobileNetV3 Large 224/0.75        |   155M     |	4.0M    |      26.7    |       /    |

