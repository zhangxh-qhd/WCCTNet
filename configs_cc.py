import os

class Config(object):
    """Global Config class"""

    # Dataset configs
    spatial_dimentions = 2
    input_shape = (256, 256, 2)  # (x_dim, y_dim, z_dim, real-imag)
    in_channels = 1
    out_channels = 1
    dims = 32
    batch_size = 8
    resolution = 256
    data_loaders_num_workers = 4

    trainDir = "/mnt/data/zxh/dataset/CC359/single_channel/Train"
    validDir = "/mnt/data/zxh/dataset/CC359/single_channel/Val"
    testDir = "/mnt/data/zxh/dataset/CC359/single_channel/Val"

    mask_path = "mask_CC"
    mask_type = "Gaussian_1D"  #'Gaussian_1D', ' Gaussian_2D', 'poisson_2D','radial','spiral'
    mask_name ='G1D10' #'G1D10-50','P2D10-50', 'R10-90', 'S10-90'


    fft_loss = True
    fft_lossweight = 10

    # Training configs
    normalize_input = False
    num_epochs = 200
    resume_epoch = None
    temp_epoch = 0
    temp_init = 1
    learning_rate = 2e-4
    lr_decay = "cos"  # "cos" or "schedule"
    decay_factor = 0.5
    decay_epochs = [60,  150, 180]
    momentum = 0.9
    betas = (0.9, 0.999)
    weight_decay = 0

    save_dir = 'trainResult/WCCTNet_CC'

    # models_dir = os.path.join(save_dir, "generalization", "models/")
    models_dir = os.path.join(save_dir,  mask_name, 'models/')
    workspace_dir = os.path.join(save_dir, mask_name, 'workspace/')
    img_dir = os.path.join(save_dir,  mask_name, 'image/')
    result_dir = os.path.join(save_dir, mask_name, 'result/')

    # test configs
    test_save_dir = "test_result/WCCTNet_CC"
    test_resultDir = os.path.join(test_save_dir, mask_type, mask_name[-2:])
    test_iterDir = os.path.join(test_save_dir, mask_type, mask_name[-2:], 'iter_images')
    test_imageDir = os.path.join(test_save_dir, mask_type, mask_name[-2:], 'images')

config = Config()
