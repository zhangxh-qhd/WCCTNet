# WCCTNet
WCCTNet: 基于小波域的复数卷积和复数Transformer的轻量级MR图像重建方法

Pytorch Code for the paper "基于小波域的复数卷积和复数Transformer的轻量级MR图像重建方法"


## Training and Testing
Use different configs files for fastMRI dataset (configs) and Calgary Campinas single-channel (CC) dataset  (configs_cc)

please modify the filepath in the configs.py or configs_cc.py according to your dataset path  
```path setting
trainDir = "xxx/xxx/xxx/xxx"   training set path
validDir = "xxx/xxx/xxx/xxx"   validation set path
```

## train WCCTNet on the fastMRI dataset: 
``` train
python train.py
```
## test WCCTNet on the fastMRI dataset: 
(Download [pre-trained weights]https://pan.baidu.com/s/1-bdOL2r7HCQK8gjc555wQw 提取码: 2502)
``` test
python test.py
```
## train WCCTNet on the CC datast:
``` train
python train_CC.py
```
## test WCCTNet on the CC dataset: 
``` test
python test_CC.py
```

## Ackonwledgements

We give acknowledgements to [fastMRI](https://github.com/facebookresearch/fastMRI), [Restormer](https://github.com/swz30/Restormer), 
[SwinMR](https://github.com/ayanglab/SwinMR), and [ReconFormer](https://github.com/guopengf/ReconFormer).

