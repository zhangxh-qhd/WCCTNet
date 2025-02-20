# WCCTNet
WCCTNet: 基于小波域的复数卷积和复数Transformer的轻量级MR图像重建方法

Pytorch Code for the paper "基于小波域的复数卷积和复数Transformer的轻量级MR图像重建方法"


## Training and Testing
Use different options (configs files) for fastMRI dataset and Calgary Campinas single-channel (CC) dataset  
## train WCCTNet on fastMRI: 
``` train
python train.py
```
## test WCCTNet on fastMRI: 
``` test
python test.py
```
## train WCCTNet on CC datast:
``` train
python train_CC.py
```
## test WCCTNet on CC dataset: 
``` test
python test_CC.py
```

## Ackonwledgements

We give acknowledgements to [fastMRI](https://github.com/facebookresearch/fastMRI), [Restormer](https://github.com/swz30/Restormer), 
[SwinMR](https://github.com/ayanglab/SwinMR), and [ReconFormer](https://github.com/guopengf/ReconFormer).

This repository is based on:
``` bash
fastMRI: An Open Dataset and Benchmarks for Accelerated MRI (code and paper);
```
```bash
SwinMR: Swin Transformer for Fast MRI (code and paper);
````
```bash
Restormer: Efficient Transformer for High-Resolution Image Restoration (code and paper);
```
```bash
ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer (code and paper).
```
