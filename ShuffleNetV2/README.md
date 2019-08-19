# [ShuffleNetV2](https://arxiv.org/pdf/1807.11164.pdf)
This repository contains ShuffleNetV2 implementation by Pytorch.


## Requirements
Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Usage
Train:
```shell
python train.py --model-size 1.5x --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```
Eval:
```shell
python train.py --eval --eval-resume YOUR_WEIGHT_PATH --model-size 1.5x --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```

## Results

| Model                   | FLOPs | #Params  | Top-1        | Top-5     |
| :---------------------: | :---: | :------: | :----------: | :------:  |
|    ShuffleNetV2 0.5x    |  41M  |     1.4M |     38.9 	|    17.4   |
|    ShuffleNetV2 1.0x    | 146M  |     2.3M |     30.6 	|    11.1   |   
|    ShuffleNetV2 1.5x    | 299M  |     3.5M |     27.4 	|     9.4   |  
|    ShuffleNetV2 2.0x    | 591M  |     7.4M |     25.0 	|     7.6   |


## Citation
If you use these models in your research, please cite:


    @inproceedings{ma2018shufflenet, 
                title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},  
                author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},  
                booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
                pages={116--131}, 
                year={2018} 
    }
