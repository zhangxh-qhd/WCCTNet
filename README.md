# WCCTNet
this is the original code of our proposed WCCTNet:
基于小波域的复数卷积和复数Transformer的轻量级MR图像重建方法

Training and Testing
Use different options (configs files) for fastMRI dataset and Calgary Campinas single-channel (CC) dataset  
To train WCCTNet on fastMRI: python train.py  
To test WCCTNet on fastMRI: python test.py  

To train WCCTNet on CC datast: <br>
`python train_CC.py ` <br>
To test WCCTNet on CC dataset:<br> 
`python test_CC.py ' 


This repository is based on:
fastMRI: An Open Dataset and Benchmarks for Accelerated MRI (code and paper);
SwinMR: Swin Transformer for Fast MRI (code and paper);
Restormer: Efficient Transformer for High-Resolution Image Restoration (code and paper);
ReconFormer: Accelerated MRI Reconstruction Using Recurrent Transformer (code and paper).
