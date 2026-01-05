from torch.utils.data import DataLoader
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler

import math
import numpy as np
import os
import tqdm
import glob
import time
import datetime
from model import RawFormer
from load_dataset import load_data
from loss.losses import *

def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def PSNR(img1, img2):
    mse_ = np.mean((img1 - img2) ** 2)
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

if __name__ == '__main__':
    opt = {}
    opt = {'base_lr':2e-4}      # base learning rate
    opt['batch_size'] = 8       # batch size
    opt['patch_size'] = 128     # cropped image patch size when training
    opt['epochs'] = 3000        # total training epochs

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print('GPU id:', os.environ["CUDA_VISIBLE_DEVICES"])

    save_weights_file = os.path.join('weights')   # save trained models

    if not os.path.exists(save_weights_file):
        os.makedirs(save_weights_file)

    use_pretrain = False
    pretrain_weights = os.path.join(save_weights_file, 'model_2000.pth')

    train_input_paths = sorted(glob.glob('/media/dataset/LOL-v1/train/low/*'))
    train_gt_paths = sorted(glob.glob('/media/dataset/LOL-v1/train/normal/*'))

    assert len(train_input_paths) == len(train_gt_paths), "Input과 GT의 개수가 일치하지 않습니다."

    test_input_paths = sorted(glob.glob('/media/dataset/LOL-v1/test/low/*'))
    test_gt_paths = sorted(glob.glob('/media/dataset/LOL-v1/test/normal/*'))

    assert len(test_input_paths) == len(test_gt_paths), "Test Input과 GT의 개수가 일치하지 않습니다."

    train_data = load_data(train_input_paths, train_gt_paths, patch_size=opt['patch_size'], training=True)
    test_data = load_data(test_input_paths, test_gt_paths, patch_size=opt['patch_size'], training=False)

    dataloader_train = DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    dataloader_val = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda")
    model = RawFormer()
    model = nn.DataParallel(model).cuda()

    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))
    model = model.to(device)

    start_epoch = 0
    end_epoch = opt['epochs']
    best_psnr = 0
    best_epoch = 0

    L1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])

    if use_pretrain:
        checkpoint = torch.load(pretrain_weights)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        start_epoch = checkpoint['epoch'] + 1

    warmup_epochs = 20
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, end_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    torch.cuda.empty_cache()
    loss_scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        for i, img in enumerate(tqdm.tqdm(dataloader_train)):
            optimizer.zero_grad()
            input = img[0].to(device)
            gt = img[1].to(device)

            with torch.amp.autocast(device_type='cuda'):
                output = model(input)
                L_loss = L1_loss(output, gt)
                output_fft = torch.fft.fft2(output)
                gt_fft = torch.fft.fft2(gt)
                fft_loss = 0.1 * L1_loss(output_fft, gt_fft)
                loss = L_loss + fft_loss

            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        with torch.no_grad():
            model.eval()
            psnr_val = []
            for ii, data_val in enumerate(tqdm.tqdm(dataloader_val)):
                input = data_val[0].to(device)
                gt = data_val[1].to(device)
                with torch.amp.autocast(device_type='cuda'):
                    output = model(input)
                pred = torch.clamp(output, 0, 1)
                psnr_val.append(calculate_psnr((data_val[1].numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8),
                                         (pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(
                                             np.uint8)))

            psnr_val = sum(psnr_val) / len(dataloader_val)

            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(save_weights_file, "model_best.pth"))

            print("------------------------------------------------------------------")
            print("[PSNR: %.4f] ----  [best_Ep: %d, Best_PSNR: %.4f] " % (psnr_val, best_epoch, best_psnr))
            model.train()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,epoch_loss, scheduler.get_last_lr()[0]))
        print("------------------------------------------------------------------")

        if epoch % 100 == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(save_weights_file, f"model_{epoch}.pth"))

    print("Now time is : ", datetime.datetime.now().isoformat())
    print('Model saved in: ', save_weights_file)