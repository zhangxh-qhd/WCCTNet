import os
import time
from collections import defaultdict
from tqdm import tqdm
from data_fastMRI.mri_data import *
from data_fastMRI.subsample import create_mask_for_mask_type
import evaluate
from model import *
from configs import config


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def network_params(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print('[Network] Total number of parameters : %.6f M' % (num_params / 1e6))
    # print('# The number of net parameters:', sum(param.numel() for param in model.parameters()))
    return num_params


def main(net_path):
    # Build model
    if not os.path.exists(config.test_resultDir):
        os.makedirs(config.test_resultDir)


    torch.random.manual_seed(42)
    print('Loading model ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = CUnet(in_channels=config.in_channels,
                  out_channels=config.out_channels,
                  dim=config.dims)

    network_params(net)

    model = net.to(device)

    checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    mask = create_mask_for_mask_type(config.mask_type, config.center_fractions,
                                     config.accelerations)
    test_data_transform = DataTransform(config.resolution, config.challenge, mask, use_seed=True)

    test_dataloader = create_dataset(config.testDir, test_data_transform, bs=8,
                                  shuffle=False, phase="test", challenge=config.challenge,
                                  num_workers=4,
                                  sample_rate=1.0, display=False)

    print("# number of testing samples: %d" % int(len(test_dataloader)))


    test_logs = []
    rec_time = 0 

    model.eval()
    with torch.no_grad():
        for itt, data in enumerate(tqdm(test_dataloader)):
            x_zf, x_target, mean, std, norm, fname, slice, max, mask, masked_kspace = data
            x_zf, x_target = x_zf.to(device), x_target.to(device)

            start_time = time.time()
            x_pred = net(x_zf)
            elapse_time = time.time() - start_time
            rec_time += elapse_time

            mean = mean.unsqueeze(1).unsqueeze(2).to(device)
            std = std.unsqueeze(1).unsqueeze(2).to(device)
            magx_pred = transforms.complex_abs(x_pred) * std + mean
            magx_target = transforms.complex_abs(x_target) * std + mean

            test_loss = F.l1_loss(magx_pred, magx_target)

            test_logs.append({
                'fname': fname,
                'slice': slice,
                'output': (magx_pred).detach(),
                'target': (magx_target),
                'loss': test_loss.detach().cpu().numpy(),
            })

        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)

        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))

        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[], nmse_zf=[], ssim_zf=[], psnr_zf=[])
        outputs_save = {}
        targets_save = {}

        for fname in tqdm(outputs):
            output = torch.stack([out for _, out in sorted(outputs[fname])])
            target = torch.stack([tgt for _, tgt in sorted(targets[fname])])
            outputs_save[fname] = output
            targets_save[fname] = target

            metrics['nmse'].append(evaluate.nmse_torch(target, output))
            metrics['ssim'].append(evaluate.ssim_torch(target, output))
            metrics['psnr'].append(evaluate.psnr_torch(target, output))

        metrics = {metric: np.mean(values).tolist() for metric, values in metrics.items()}

        print('No. Slices: ', len(outputs))
        print('{:.6E} {:.4f}-{:.4f}-{:.4f}' .format(
            metrics['val_loss'], metrics['nmse'], metrics['ssim'], metrics['psnr']))

        print("testing finished ...\n")

if __name__ == "__main__":

    net_path = os.path.join(config.models_dir, 'net.pth')
    main(net_path)

