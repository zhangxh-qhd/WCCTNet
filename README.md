# WCCTNet
WCCTNet: 基于小波域的复数卷积和复数Transformer的轻量级MR图像重建方法

Pytorch Code for the paper "基于小波域的复数卷积和复数Transformer的轻量级MR图像重建方法"


## Training and Testing
Use different configs files for fastMRI dataset (configs) and Calgary Campinas single-channel (CC) dataset  (configs_cc)
please modify the trainDir and validDir in the configs.py or configs_cc.py accordinig to your dataset path  

## train WCCTNet on the fastMRI dataset: 
``` train
python train.py
```
## test WCCTNet on the fastMRI dataset: 
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

