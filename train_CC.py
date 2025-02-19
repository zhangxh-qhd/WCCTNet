from __future__ import print_function
from typing import Tuple
import shutil

from tqdm import tqdm
import torch
import time
import datetime
import torch.nn.modules.loss as Loss
from torch import optim
from torch.utils.data import DataLoader
import logging

from data_cc.dataset_CC import *
from data_cc.select_mask import define_Mask
from evaluate import *
from model import *
from configs_cc import config

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


def set_seeds(seed):
    """Set the seeds for reproducibility
    
    Parameters
    ----------
    seed : int
        The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """Get device

    Returns
    -------
    device : torch.device
        The device to use.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fft_Realmap(x):
    #x is a 4-dim or 3-dim tensor
    fft_x = torch.fft.rfft2(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag
    return fft_x_real, fft_x_imag


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2 or data.size(-3) == 2
    return (data ** 2).sum(dim=-1).sqrt() if data.size(-1) == 2 else (data ** 2).sum(dim=-3).sqrt()


def train_epoch(net, optimizer, loss_criterion, tr_dataloader, epoch):
    """Train for one epoch of the data

    Parameters
    ----------
    net : torch.nn.Module
        The network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    tr_dataloader : torch.utils.data.DataLoader
        The training data loader.
    epoch : int
        The epoch number.

    Returns
    -------
    avg_loss : float
        The average loss for the epoch.
    net : torch.nn.Module
        The trained network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    """

    net.train()
    avg_loss = 0.0
    device = get_device()
    net = net.to(device)
    st = time.time()

    for itt, data in  enumerate(tr_dataloader, 0):

        zf, x_target = data['L'].to(device), data['H'].to(device)
        x_pred = net(zf)
        magx_pred, magx_target = complex_abs(x_pred), complex_abs(x_target)

        loss1 = loss_criterion(magx_target, magx_pred) / (2 * config.batch_size)

        if config.fft_loss:
            x_targetReal, x_targetImag = fft_Realmap(magx_target)
            x_predReal, x_predImag = fft_Realmap(magx_pred)
            loss2 = config.fft_lossweight * (F.l1_loss(x_targetReal, x_predReal)
                                             + F.l1_loss(x_targetImag, x_predImag)) / 2

        loss = loss1 + loss2

        et = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]

        if (itt + 1) % 50 == 0:
            logging.info('Epoch: {} - Itter: {}/{} - loss: {:.4f}, loss1:{:.3f}, loss2:{:.3f}, time:{}'
                         .format(epoch + 1, itt + 1, len(tr_dataloader), loss.item(), loss1,
                                 loss2, et))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.detach().item()

    return avg_loss/len(tr_dataloader), net, optimizer

def validate(net, val_dataloader):
    """Validate the model on the validation set

    Parameters
    ----------
    net : torch.nn.Module
        The network to validate.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    val_dataloader : torch.utils.data.DataLoader
        The validation data loader.
    epoch : int
        The epoch number.

    Returns
    -------
    avg_loss : float
        The average loss for the epoch.
    avg_ssim : float
        The average SSIM for the epoch.

    """
    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    device = get_device()

    net.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for itt, data in enumerate(tqdm(val_dataloader)):

            zf, x_target = data['L'].to(device), data['H'].to(device)
            x_pred = net(zf)
            magx_pred, magx_target, magx_zf = complex_abs(x_pred), complex_abs(x_target), complex_abs(zf)
            loss = F.l1_loss(magx_pred, magx_target)
            ssim = ssim_torch(magx_target, magx_pred)
            psnr = psnr_torch(magx_target, magx_pred)

            total_loss += loss.detach().item()
            total_ssim += ssim
            total_psnr += psnr

            torch.cuda.empty_cache()

    return total_loss/len(val_dataloader), total_psnr/len(val_dataloader), total_ssim/len(val_dataloader)


def train(net, optimizer, loss_criterion, tr_dataloader, val_dataloader, start_epoch):
    """Train the network

    Parameters
    ----------
    net : torch.nn.Module
        The network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    tr_dataloader : torch.utils.data.DataLoader
        The training data loader.
    val_dataloader : torch.utils.data.DataLoader
        The validation data loader.
    """
    best_loss = 999999999999
    best_ssim = 0
    best_psnr = 0
    best_epoch = 0
    st = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs,  eta_min=1e-6)

    for epoch in range(start_epoch, config.num_epochs):

        for param_group in optimizer.param_groups:
           lr = param_group['lr']

        logging.info("Training epoch {}/{}..., lr = {} ".format(epoch + 1, config.num_epochs, lr))

        # Training
        avg_tr_loss, net, optimizer = train_epoch(net, optimizer, loss_criterion, tr_dataloader, epoch)

        scheduler.step()

        et = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]

        logging.info('Epoch {} - Avg. training loss: {:.4f}, time:{:s}'
                     .format(epoch + 1, avg_tr_loss,  et))

        torch.cuda.empty_cache()

        # Validation
        logging.info("Begin validation!")
        avg_vld_loss, avg_vld_psnr, avg_vld_ssim = validate(net, val_dataloader)

        et = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]


        logging.info('Epoch {} - Avg. validation loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, time: {:s}s'
                     .format(epoch + 1, avg_vld_loss, avg_vld_psnr, avg_vld_ssim, et))

        save_checkpoint(
            {
                'epoch': epoch,
                'arch': 'complexnet',
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            is_best=avg_vld_loss < best_loss or avg_vld_ssim > best_ssim or avg_vld_psnr > best_psnr,
            filename=config.models_dir + '/net.pth'
        )
        logging.info('Model Saved!')

        if avg_vld_loss < best_loss:
            best_loss = avg_vld_loss
        if avg_vld_ssim > best_ssim:
            best_ssim = avg_vld_ssim
        if avg_vld_psnr > best_psnr:
            best_psnr = avg_vld_psnr
            best_epoch = epoch + 1

        logging.info('best_epoch: {:d}, best_loss : {:.4f}, best_PSNR: {:.4f}, best_SSIM: {:.4f}'
                     .format(best_epoch, best_loss, best_psnr, best_ssim))





def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save a checkpoint

    Parameters
    ----------
    state : dict
        The state to save.
    is_best : bool
        Whether this is the best model.
    filename : str
        The filename to save the checkpoint to.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,  config.models_dir + '/net_{}.pth'.format(state['epoch'] + 1))

def restore_model(model_save_dir, resume_epoch):
    """Restore the trained generator and discriminator."""
    print('Loading the models from step {}...'.format(resume_epoch))
    net_path = os.path.join(model_save_dir, 'net_{}.pth'.format(resume_epoch))
    net = torch.load(net_path, map_location=lambda storage, loc: storage)
    return net
    

def get_dataloaders(mask):
    """Get the dataset loaders.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The training and validation data loaders
    """

    imshape = (config.input_shape[0], config.input_shape[1])
    norm = np.sqrt(config.input_shape[0] * config.input_shape[1])

    train_counter, kspace_train, rec_train = getimage_slice(config.trainDir, imshape, norm)
    valid_counter, kspace_valid, rec_valid = getimage_slice(config.validDir, imshape, norm)

    train_dataset = DatasetCC(train_counter, kspace_train, rec_train, mask, is_training=True)
    valid_dataset = DatasetCC(valid_counter, kspace_valid, rec_valid, mask, is_training=False)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data_loaders_num_workers
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data_loaders_num_workers
    )

    print("# number of training samples: %d" % int(len(train_dataset)))
    print("# number of validating samples: %d" % int(len(valid_dataset)))
    return train_dataloader, valid_dataloader

    

#dyconv + l2_loss + mma_loss for du and epsv
if __name__ == '__main__':
    # set_seeds(222)
    device = get_device()

    loss_criterion =  Loss.MSELoss(reduction='sum')

    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.img_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)


    mask = define_Mask(config.mask_path, config.mask_name)
    train_dataset = get_datasets(config, mask, is_training=True, start_num=0)
    test_dataset = get_datasets(config, mask, is_training=False, start_num=0)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.data_loaders_num_workers
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data_loaders_num_workers
    )

    print("# number of training samples: %d" % int(len(train_dataset)))
    print("# number of testing samples: %d" % int(len(test_dataset)))

    net = CUnet(in_channels=config.in_channels,
                  out_channels=config.out_channels,
                  dim=config.dims)
    net = net.to(device)
    network_params(net)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    st = time.time()
    start_epoch = 0
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch
        checkpoint = restore_model(config.models_dir, config.resume_epoch)
        net.load_state_dict(checkpoint["state_dict"], strict=False)

    train(net, optimizer, loss_criterion, train_dataloader, test_dataloader, start_epoch)

    total_traintime = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]
    logging.info("total training time:{}".format(total_traintime))
