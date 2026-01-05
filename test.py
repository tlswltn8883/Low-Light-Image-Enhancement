import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import torch.nn.functional as F
import numpy as np
import os
import tqdm
import imageio
from model import RawFormer
from load_dataset import load_data
import glob
import time


if __name__ == '__main__':
    opt = {}
    opt['use_gpu'] = True
    opt['model_size'] = 'B'  # 32/48/64 --> small/base/large

    dataset_name = 'data'
    save_weights_file = os.path.join('weights')
    save_images_file = f'/media/jsshin/enhancement/result/temp/{dataset_name}'

    if not os.path.exists(save_images_file):
        os.makedirs(save_images_file)

    test_input_paths = sorted(
        glob.glob(os.path.join(f'/media/dataset/{dataset_name}/test/low/*')))
    test_gt_paths = sorted(
        glob.glob(os.path.join(f'/media/dataset/{dataset_name}/test/normal/*')))
    print('test data: %d pairs' % len(test_input_paths))
    test_data = load_data(test_input_paths, test_gt_paths, training=False)

    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=False)

    if opt['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    model = RawFormer()
    checkpoint = torch.load(save_weights_file + '/final.pth', map_location=device)

    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
        strict=True)
    epoch = checkpoint['epoch']
    print('load model from epoch:', epoch)

    model = model.to(device)

    print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))
    model.eval()

    with torch.no_grad():
        psnr_val = []
        ssim_val = []
        mul = 8
        total_time = 0
        number = len(dataloader_test)
        for ii, data_test in enumerate(tqdm.tqdm(dataloader_test)):
            gt = (data_test[1].numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8)
            input = data_test[0].to(device)
            filename_base = os.path.splitext(os.path.basename(test_input_paths[ii]))[0]

            h, w = input.shape[2], input.shape[3]
            H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
            padh = H - h if h % mul != 0 else 0
            padw = W - w if w % mul != 0 else 0
            input = F.pad(input, (0, padw, 0, padh), 'reflect')

            start_time = time.time()
            output = model(input)
            end_time = time.time() - start_time
            total_time += end_time

            pred = (torch.clamp(output, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = pred[:h, :w, :]
            psnr = PSNR(pred, gt)
            ssim = SSIM(pred, gt, multichannel=True, win_size=3)
            print('image:{}\tPSNR:{:.4f}\tSSIM:{:.4f}'.format(ii, psnr, ssim))
            psnr_val.append(psnr)
            ssim_val.append(ssim)
            save_path = os.path.join(save_images_file, filename_base + '.png')  # .png 확장자로 저장
            imageio.imwrite(save_path, pred)

    psnr_average = sum(psnr_val) / len(dataloader_test)
    ssim_average = sum(ssim_val) / len(dataloader_test)

    print("average_PSNR: %f, average_SSIM: %f" % (psnr_average, ssim_average))
    print("average_processing_time %f" % (total_time / number))
