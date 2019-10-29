# ShuffleNetV2.ExLarge

This repository contains ShuffleNetV2.ExLarge implementation by Pytorch, which is a extra large version of ShuffleNetV2.

## Requirements
Download the ImageNet dataset and move validation images to labeled subfolders. To do this, you can use the following script:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Usage
We used external training dataset to achieve the result, so you do not need to re-train it.

This is the evaluation script:
```shell
python eval.py --eval --eval-resume YOUR_WEIGHT_PATH --train-dir YOUR_TRAINDATASET_PATH --val-dir YOUR_VALDATASET_PATH
```


## Trained Models
- OneDrive download: [Link](https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo)
- BaiduYun download: [Link](https://pan.baidu.com/s/1EUQVoFPb74yZm0JWHKjFOw) (extract code: mc24)


## Results

| Model                  | FLOPs | #Params   | Top-1     | Top-5 |
| :--------------------- | :---: | :------:  | :---:     | :---: |
| ShuffleNetV2.ExLarge     | 46.2G | 254.7M    | 15.52 | 2.9  |
