from __future__ import print_function
from typing import Tuple
import shutil
import os
import torch
import time
import datetime
import torch.nn.modules.loss as Loss
from torch import optim
import logging

from data_fastMRI.mri_data import *
from data_fastMRI.subsample import create_mask_for_mask_type
import evaluate
from model import *
from configs import config


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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_abs_complex(x):
    #the last dim of x is 2
    x_real, x_imag = torch.unbind(x, -1)
    return torch.sqrt(x_real ** 2 + x_imag ** 2)

def fft_Realmap(x):
    #x is a 4-dim or 3-dim tensor
    fft_x = torch.fft.rfft2(x)
    fft_x_real = fft_x.real
    fft_x_imag = fft_x.imag
    return fft_x_real, fft_x_imag


mag = lambda x: (x[:, 0:1, ...] ** 2 + x[:, 1:2, ...] ** 2) ** 0.5


def train_epoch(net, optimizer, loss_criterion, tr_dataloader, epoch, scheduler):

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
    avg_loss = 0.0
    net.train()
    device = get_device()
    net = net.to(device)
    iter_num = 0


    st = time.time()
    
    for itt, data in enumerate(tr_dataloader, 0):
        iter_num += 1

        zf, x_target, mean, std, norm, fname, slice, max, mask, y = data

        zf, x_target = zf.to(device), x_target.to(device)

        x_pred = net(zf)

        magx_pred, magx_target = transforms.complex_abs(x_pred), transforms.complex_abs(x_target)

        loss1 = loss_criterion[2](magx_target, magx_pred) / (2 * config.batch_size)

        x_targetReal, x_targetImag = fft_Realmap(magx_target)
        x_predReal, x_predImag = fft_Realmap(magx_pred)
        loss2 = config.fft_lossweight * (loss_criterion[1](x_targetReal, x_predReal)
                                         + loss_criterion[1](x_targetImag, x_predImag)) / 2

        loss =  loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.detach().item()


        et = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]


        if (itt+1) % config.print_idx == 0:
            logging.info('Epoch: {} - Itter: {}/{} - loss: {:.4f}, loss1:{:.3f}, loss2:{:.3f}, time:{}'
                         .format(epoch + 1, itt + 1, len(tr_dataloader), loss.item(), loss1,
                                  loss2,  et))
        if iter_num % config.save_step == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'arch': 'wcctnet',
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                is_best=False,
                filename=config.models_dir + 'net_{}.pth'.format(iter_num)
            )
            logging.info('Model Saved!')

    return avg_loss/len(tr_dataloader), net, optimizer


def validate(net, loss_criterion, val_dataloader, epoch):
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
    with torch.no_grad():
        for itt, data in enumerate(val_dataloader):

            zf, x_target, mean, std, norm, fname, slice, max, mask, y = data
            zf, x_target = zf.to(device), x_target.to(device)

            x_pred = net(zf)

            mean = mean.unsqueeze(1).unsqueeze(2).to(device)
            std = std.unsqueeze(1).unsqueeze(2).to(device)
            magx_pred = transforms.complex_abs(x_pred) * std + mean
            magx_target = transforms.complex_abs(x_target) * std + mean

            loss = loss_criterion[0](magx_pred, magx_target)
            ssim = evaluate.ssim_torch(magx_target, magx_pred)
            psnr = evaluate.psnr_torch(magx_target, magx_pred)

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
        avg_tr_loss,  net, optimizer = train_epoch(net, optimizer, loss_criterion, tr_dataloader, epoch, scheduler)
        scheduler.step()

        et = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]

        logging.info('Epoch {} - Avg. training loss: {:.4f},  time:{:s}'.format(epoch + 1, avg_tr_loss,  et))
        
        torch.cuda.empty_cache()

        # Validation
        logging.info("Begin validation!")
        avg_vld_loss, avg_vld_psnr, avg_vld_ssim = validate(net, loss_criterion, val_dataloader, epoch)

        et = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]
        logging.info('Epoch {} - Avg. validation loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, time:{:s}'
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
            filename=config.models_dir + 'net.pth'
            # filename = config.models_dir + 'net_{}.pth'.format(epoch + 1)
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
        shutil.copyfile(filename,  config.models_dir + 'net_{}.pth'.format(state['epoch'] + 1))

def restore_model(model_save_dir, resume_epoch):
    """Restore the trained generator and discriminator."""
    print('Loading the models from step {}...'.format(resume_epoch))
    net_path = os.path.join(model_save_dir, 'net_{}.pth'.format(resume_epoch))
    net = torch.load(net_path, map_location=lambda storage, loc: storage)
    return net


def _create_dataset(data_path, data_transform, bs, shuffle, phase, sample_rate=None, display=False):

    dataset = SliceData(
        root=data_path,
        transform=data_transform,
        sample_rate=sample_rate,
        challenge=config.challenge,
        phase=phase
    )
    if display:
        dataset = [dataset[i] for i in range(100, 108)]
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, pin_memory=True,
                      num_workers=config.data_loaders_num_workers)

def get_dataloaders():
    """Get the dataset loaders.

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        The training and validation data loaders
    """
    mask = create_mask_for_mask_type(config.mask_type, config.center_fractions,
                                     config.accelerations)

    train_data_transform = DataTransform(config.resolution, config.challenge, mask, use_seed=False)
    val_data_transform = DataTransform(config.resolution, config.challenge, mask, use_seed=True)

    train_dataset = _create_dataset(config.trainDir, train_data_transform, config.batch_size, True, 'train', 1.0)
    valid_dataset = _create_dataset(config.validDir, val_data_transform, 1, False, 'valid', 1.0)

    print("# number of training samples: %d" % int(len(train_dataset)))
    print("# number of validating samples: %d" % int(len(valid_dataset)))
    return train_dataset, valid_dataset
    

if __name__ == '__main__':
    tr_dataset, val_dataset = get_dataloaders()
    device = get_device()

    net = CUnet(in_channels=config.in_channels,
                  out_channels=config.out_channels,
                  dim=config.dims)

    net = net.to(device)
    network_params(net)


    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    loss_criterion_L1s = Loss.L1Loss(reduction='sum')
    loss_criterion_L1m = Loss.L1Loss()

    loss_criterion_L2s = Loss.MSELoss(reduction='sum')
    loss_criterion_L2m = Loss.MSELoss()


    loss_criterion = (loss_criterion_L1s, loss_criterion_L1m,
                      loss_criterion_L2s, loss_criterion_L2m)

    
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.img_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    st = time.time()
    start_epoch = 0
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch
        # step = start_epoch * len(dataset_train)/opt.batchSize
        checkpoint = restore_model(config.models_dir, config.resume_epoch)
        net.load_state_dict(checkpoint["state_dict"], strict=False)
    train(net, optimizer, loss_criterion, tr_dataset, val_dataset, start_epoch)
    total_traintime = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]
    logging.info("total training time:{}".format(total_traintime))
