# [ShuffleNetV1](https://arxiv.org/pdf/1707.01083.pdf)

This repository contains ShuffleNetV1 implementation by Pytorch.


## Requirements
Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Usage
Train:
```shell
python train.py --model-size 2.0x --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```
Eval:
```shell
python train.py --eval --eval-resume YOUR_WEIGHT_PATH --model-size 2.0x --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```

## Results

|    Model                 |  FLOPs    |   #Params |   Top-1   |   Top-5   |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
ShuffleNetV1 2.0x (group=3)|    524M    |	5.4M    |      **25.9**    |        8.6   |
ShuffleNetV1 2.0x (group=8)|    522M    |   6.5M    |      27.1    |        9.2   |
1.0 MobileNetV1-224 |    569M    |   4.2M    |      29.4    |        /   |
ShuffleNetV1 1.5x (group=3)|    292M    |	3.4M    |      **28.4**    |        9.8   |
ShuffleNetV1 1.5x (group=8)|    290M    |   4.3M    |      29.0    |       10.4   |
0.75 MobileNetV1-224 |    325M    |   2.6M    |      31.6    |        /   |
ShuffleNetV1 1.0x (group=3)|   138M     |	1.9M    |      32.2    |       12.3    |
ShuffleNetV1 1.0x (group=8)|    138M    |   2.4M    |      **32.0**    |       13.6   |
0.5 MobileNetV1-224 |    149M    |   1.3M    |      36.3    |        /   |
ShuffleNetV1 0.5x (group=3)|   38M      |	0.7M    |      42.7    |       20.0    |
ShuffleNetV1 0.5x (group=8)|    40M     |   1.0M    |      **41.2**    |       19.0   |
0.25 MobileNetV1-224 |    41M    |   0.5M    |      49.4    |        /   |



## Citation
If you use these models in your research, please cite:


    @inproceedings{zhang2018shufflenet,
                title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
                author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
                booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
                pages={6848--6856},
                year={2018}
    }

