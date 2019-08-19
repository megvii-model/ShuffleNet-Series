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
  
## Results


| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot |  328M |  3.4M |  25.4   |   8.0   |
    
## Citation
If you use these models in your research, please cite:


    @article{guo2019single,
            title={Single path one-shot neural architecture search with uniform sampling},
            author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
            journal={arXiv preprint arXiv:1904.00420},
            year={2019}
    }
