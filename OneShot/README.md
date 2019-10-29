# [One-Shot NAS](https://arxiv.org/abs/1904.00420)
This repository contains single path one-shot NAS searched networks implementation by Pytorch.


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


| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot |  328M |  3.4M |  **25.1**   |   8.0   |
|    NASNET-A|  564M |  5.3M |  26.0   |   8.4   |
|    PNASNET|  588M |  5.1M |  25.8   |   8.1   |
|    MnasNet|  317M |  4.2M |  26.0   |  8.2   |
|    DARTS|  574M|  4.7M |  26.7   |   8.7  |
|    FBNet-B|  295M|  4.5M |  25.9   |   -   |
    
## Citation
If you use these models in your research, please cite:


    @article{guo2019single,
            title={Single path one-shot neural architecture search with uniform sampling},
            author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
            journal={arXiv preprint arXiv:1904.00420},
            year={2019}
    }
