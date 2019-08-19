# ShuffleNet Series
ShuffleNet Series by Megvii Research.

## Introduction
This repository contains the following ShuffleNet series models:
- ShuffleNetV1:   [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
-  ShuffleNetV2:   [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
-  ShuffleNetV2+:  A strengthen version of ShuffleNetV2.
-  ShuffleNetV2.Large:  A deeper version based on ShuffleNetV2.
-  OneShot:    [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420)
-  DetNAS:     [DetNAS: Backbone Search for Object Detection](https://arxiv.org/abs/1903.10979)

## Trained Models
 OneDrive download: [Link](https://1drv.ms/f/s!AgaP37NGYuEXhRfQxHRseR7eSxXo)

## Details

### ShuffleNetV2+
The following is the comparison between ShuffleNetV2+ and [MobileNetV3](https://arxiv.org/pdf/1905.02244). Details can be seen in [ShuffleNetV2+](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B).

|    Model                 |  FLOPs    |   #Params |   Top-1   |   Top-5   |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
ShuffleNetV2+ Large        |   360M     |	6.7M    |      **22.9**    |       6.7   |
MobileNetV3 Large 224/1.25       |   356M     |	7.5M    |      23.4    |       /   |
ShuffleNetV2+ Medium       |   222M     |	5.6M    |      **24.3**    |       7.4    |
MobileNetV3 Large 224/1.0       |   217M     |	5.4M    |      24.8    |       /    |
ShuffleNetV2+ Small        |   156M     |	5.1M    |      **25.9**    |       8.3    |
MobileNetV3 Large 224/0.75        |   155M     |	4.0M    |      26.7    |       /    |

### ShuffleNetV2
The following is the comparison between ShuffleNetV2 and [MobileNetV2](https://arxiv.org/abs/1801.04381). Details can be seen in [ShuffleNetV2](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2).

| Model                   | FLOPs | #Params  | Top-1        | Top-5     |
| :--------------------- | :---: | :------: | :----------: | :------:  |
|    ShuffleNetV2 2.0x    | 591M  |     7.4M |     **25.0** 	|     7.6   |
| MobileNetV2 (1.4) | 585M | 6.9M | 25.3 | / |
|    ShuffleNetV2 1.5x    | 299M  |     3.5M |     **27.4** 	|     9.4   | 
| MobileNetV2 | 300M | 3.4M | 28.0 | / | 
|    ShuffleNetV2 1.0x    | 146M  |     2.3M |     30.6 	|    11.1   |   
|    ShuffleNetV2 0.5x    |  41M  |     1.4M |     38.9 	|    17.4   |

### ShuffleNetV2.Large
The following is the comparison between ShuffleNetV2.Large and [SENet](https://arxiv.org/abs/1709.01507). Details can be seen in [ShuffleNetV2.Large](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2.Large).

| Model                  | FLOPs | #Params   | Top-1     | Top-5 |
| :--------------------- | :---: | :------:  | :---:     | :---: |
| ShuffleNetV2.Large     | 12.7G | 140.7M    | **18.56** | 4.48  |
| SENet                  | 20.7G |    /      | 18.68     | 4.47  |


### ShuffleNetV1
The following is the comparison between ShuffleNetV1 and [MobileNetV1](https://arxiv.org/abs/1704.04861). Details can be seen in [ShuffleNetV1](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1).

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


### OneShot
The following is the comparison between Single Path One-Shot NAS and other NAS counterparts. Details can be seen in [OneShot](https://github.com/megvii-model/ShuffleNet-Series/tree/master/OneShot).

| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot |  328M |  3.4M |  **25.4**   |   8.0   |
|    NASNET-A|  564M |  5.3M |  26.0   |   8.4   |
|    PNASNET|  588M |  5.1M |  25.8   |   8.1   |
|    MnasNet|  317M |  4.2M |  26.0   |  8.2   |
|    DARTS|  574M|  4.7M |  26.7   |   8.7  |
|    FBNet-B|  295M|  4.5M |  25.9   |   /   |

### DetNAS
The following is the performance of DetNAS on ImageNet, compared with ResNet. Details can be seen in [DetNAS](https://github.com/megvii-model/ShuffleNet-Series/tree/master/DetNAS).

| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|DetNAS_small	| 300M	| 3.7M	 |  25.9 	|     8.3  |
|DetNAS_medium	| 1.3G	| 10.4M	 |  **22.8** 	|     6.5  |
|DetNAS_large	| 3.8G	| 29.5M	 |  **21.5** 	|     6.3  |
|ResNet50 | 3.8G	| / |  24.7 	|     7.8  |
|ResNet101 | 7.6G	| / |  23.6 	|     7.1  |


