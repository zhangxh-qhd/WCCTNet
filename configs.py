import os

class Config(object):
    """Global Config class"""

    # Dataset configs
    spatial_dimentions = 2
    # input_shape = (256, 256, 2)  # (x_dim, y_dim, z_dim, real-imag)
    in_channels = 1
    out_channels = 1
    dims = 32
    batch_size = 8
    resolution = 320
    data_loaders_num_workers = 4

    challenge = "singlecoil"
    mask_type = "random" #['random', 'equispaced']
    center_fractions = [0.04]
    accelerations = [8]
    #
    trainDir = "/mnt/data/zxh/dataset/fastMRI/train1"
    validDir = "/mnt/data/zxh/dataset/fastMRI/val1"
    testDir = "/mnt/data/zxh/dataset/fastMRI/val1"


    # Training configs
    normalize_input = False
    num_epochs = 2
    resume_epoch = None
    save_step = 1000
    learning_rate = 2e-4
    lr_decay = "cos"  # "cos" or "schedule"
    decay_factor = 0.5
    decay_epochs = [15, 30]
    momentum = 0.9
    betas = (0.9, 0.999)
    weight_decay = 0
    fft_lossweight = 10

    save_dir = 'trainResult/WCCTNet'
    save_image_idx = 25
    print_idx = 50

    models_dir = os.path.join(save_dir,  mask_type, str(accelerations[0]), 'models/')
    workspace_dir = os.path.join(save_dir, mask_type, str(accelerations[0]), 'workspace/')
    img_dir = os.path.join(save_dir,  mask_type, str(accelerations[0]), 'image/')
    result_dir = os.path.join(save_dir, mask_type, str(accelerations[0]), 'result/')

    # test configs
    test_save_dir = "test_result/WCCTNet"
    test_resultDir = os.path.join(test_save_dir, mask_type, str(accelerations[0]))
    test_iterDir = os.path.join(test_save_dir, mask_type, str(accelerations[0]), 'iter_images')
    test_imageDir = os.path.join(test_save_dir, mask_type, str(accelerations[0]), 'images')


config = Config()
