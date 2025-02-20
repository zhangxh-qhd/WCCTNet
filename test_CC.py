import time
import csv
import cv2
import sys
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_cc.dataset_CC import *
from data_cc.select_mask import define_Mask
from data_cc import utils_image as util
from model import *
from configs_cc import config
import lpips
from pytorch_fid import fid_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def network_params(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print('[Network] Total number of parameters : %.6f M' % (num_params / 1e6))
    # print('# The number of net parameters:', sum(param.numel() for param in model.parameters()))
    return num_params

normalize = lambda img: (img - img.min()) / (img.max() - img.min())

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


def main(net_path, method="WCCTNet"):
    # Build model
    if not os.path.exists(config.test_resultDir):
        os.makedirs(config.test_resultDir)
    # if not os.path.exists(config.test_imageDir):
    #     os.makedirs(config.test_imageDir)
    # if not os.path.exists(config.test_iterDir):
    #     os.makedirs(config.test_iterDir)

    torch.random.manual_seed(42)
    print('Loading model ...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # net = CRestormer(in_channels=config.in_channels,
    #               x_pred_channels=config.x_pred_channels,
    #               dim=config.dims)
    net = CUnet(in_channels=config.in_channels,
                  out_channels=config.out_channels,
                  dim=config.dims)

    num_params = network_params(net)

    model = net.to(device)

    checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    test_results = OrderedDict()
    test_results['test_loss'] = []
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['nrmse'] = []
    test_results['lpips'] = []


    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    mask = define_Mask(config.mask_path, config.mask_name)
    test_dataset = get_datasets(config, mask, is_training=False, start_num=35)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data_loaders_num_workers
    )
    print("# number of testing samples: %d" % int(len(test_dataloader)))


    rec_time = 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader), 0):

            x_zf, x_target = data['L'].to(device), data['H'].to(device)
            start_time = time.time()

            x_pred = net(x_zf)
            elapse_time = time.time() - start_time
            rec_time += elapse_time

            magx_pred, magx_target, magx_zf = complex_abs(x_pred), complex_abs(x_target), complex_abs(x_zf)
            test_loss = F.l1_loss(magx_pred, magx_target)
            test_results["test_loss"].append(test_loss.detach().cpu().numpy())

            # evaluate lpips
            lpips_ = util.calculate_lpips_single(loss_fn_alex, magx_target, magx_pred)
            lpips_ = lpips_.data.squeeze().float().cpu().numpy()
         
            diff_pred_x10 = normalize(torch.mul(torch.abs(torch.sub(magx_target, magx_pred)), 10))

            magx_target = magx_target.data.squeeze().float().cpu().numpy()
            magx_pred = magx_pred.data.squeeze().float().cpu().numpy()

            # evaluate psnr/ssim/nrmse/lpips
            psnr = util.calculate_psnr_single(magx_target, magx_pred)
            ssim = util.calculate_ssim_single(magx_target, magx_pred)
            nrmse = util.calculate_nrmse_single(magx_target, magx_pred)

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['nrmse'].append(nrmse)
            test_results['lpips'].append(lpips_)


            diff_pred_x10 = diff_pred_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()

            magx_target = (np.clip(normalize(magx_target), 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
            magx_pred = (np.clip(normalize(magx_pred), 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8

            diff_pred_x10 = (diff_pred_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8


            isExists = os.path.exists(os.path.join(config.test_resultDir, 'GT'))
            if not isExists:
                os.makedirs(os.path.join(config.test_resultDir, 'GT'))

            isExists = os.path.exists(os.path.join(config.test_resultDir, 'Recon'))
            if not isExists:
                os.makedirs(os.path.join(config.test_resultDir, 'Recon'))

            isExists = os.path.exists(os.path.join(config.test_resultDir, 'Different_recon'))
            if not isExists:
                os.makedirs(os.path.join(config.test_resultDir, 'Different_recon'))

            cv2.imwrite(os.path.join(config.test_resultDir, 'GT', 'GT_{:05d}.png'.format(idx+1)), magx_target)
            cv2.imwrite(os.path.join(config.test_resultDir, 'Recon', '{:s}_{:05d}.png'.format(method, idx+1)), magx_pred)

            diff_pred_x10_color = cv2.applyColorMap(diff_pred_x10, cv2.COLORMAP_JET)

            cv2.imwrite(os.path.join(config.test_resultDir, 'Different_recon', 'Diff_{:s}_{:05d}.png'
                                     .format(method, idx+1)), diff_pred_x10_color)


        # summarize psnr/ssim
        ave_psnr = np.mean(test_results['psnr'])
        std_psnr = np.std(test_results['psnr'], ddof=1)
        ave_ssim = np.mean(test_results['ssim'])
        std_ssim = np.std(test_results['ssim'], ddof=1)
        ave_nrmse = np.mean(test_results['nrmse'])
        std_nrmse = np.std(test_results['nrmse'], ddof=1)
        ave_lpips = np.mean(test_results['lpips'])
        std_lpips = np.std(test_results['lpips'], ddof=1)

        print(' -- Average PSNR  {:.2f} ({:.4f}) \n -- Average SSIM  {:.4f} ({:.6f})'
              '\n-- Average NRMSE  {:.4f} ({:.6f})\n -- Average LPIPS  {:.4f} ({:.6f})'
              .format(ave_psnr, std_psnr, ave_ssim, std_ssim, ave_nrmse, std_nrmse, ave_lpips, std_lpips))

        # FID
        log = os.popen("{} -m pytorch_fid {} {} ".format(
            sys.executable,
            os.path.join(config.test_resultDir, 'GT'),
            os.path.join(config.test_resultDir, 'Recon'))).read()
        print(log)
        fid = eval(log.replace('FID:  ', ''))

    print("testing finished ...\n")


if __name__ == "__main__":

    net_path = os.path.join(config.models_dir, 'net.pth')
    main(net_path)

