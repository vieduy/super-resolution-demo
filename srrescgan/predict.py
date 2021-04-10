import os.path as osp
import glob
import os
import cv2
import numpy as np
import torch
from srrescgan.models.SRResCGAN import Generator
from srrescgan.utils import timer
from collections import OrderedDict
import time
torch.cuda.empty_cache()


def _overlap_crop_forward(x, model, min_size=100000):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    scale = 4
    b, c, h, w = x.size()
    h_e, w_e = h // 4, w // 4
    lr_list = [
        x[:, :, 0:h_e, 0:w_e],
        x[:, :, 0:h_e, w_e:w_e * 2],
        x[:, :, 0:h_e, w_e * 2:w_e * 3],
        x[:, :, 0:h_e, w_e * 3:w],

        x[:, :, h_e:h_e * 2, 0:w_e],
        x[:, :, h_e:h_e * 2, w_e:w_e * 2],
        x[:, :, h_e:h_e * 2, w_e * 2:w_e * 3],
        x[:, :, h_e:h_e * 2, w_e * 3:w],

        x[:, :, h_e * 2:h_e * 3, 0:w_e],
        x[:, :, h_e * 2:h_e * 3, w_e:w_e * 2],
        x[:, :, h_e * 2:h_e * 3, w_e * 2:w_e * 3],
        x[:, :, h_e * 2:h_e * 3, w_e * 3:w],

        x[:, :, h_e * 3:h, 0:w_e],
        x[:, :, h_e * 3:h, w_e:w_e * 2],
        x[:, :, h_e * 3:h, w_e * 2:w_e * 3],
        x[:, :, h_e * 3:h, w_e * 3:w]
    ]

    h, w = scale * h, scale * w
    h_f = scale * h_e
    w_f = scale * w_e
    output = torch.zeros([b, c, h, w])

    if w_e * h_e < min_size:
        for i in range(0, 16, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch_temp = model(lr_batch)

            if isinstance(sr_batch_temp, list):
                sr_batch = sr_batch_temp[-1]
            else:
                sr_batch = sr_batch_temp

            if i == 0:
                output[:, :, 0:h_f, 0:w_f] = sr_batch
            elif i == 1:
                output[:, :, 0:h_f, w_f:w_f * 2] = sr_batch
            elif i == 2:
                output[:, :, 0:h_f, w_f * 2:w_f * 3] = sr_batch
            elif i == 3:
                output[:, :, 0:h_f, w_f * 3:w] = sr_batch
            elif i == 4:
                output[:, :, h_f:h_f * 2, 0:w_f] = sr_batch
            elif i == 5:
                output[:, :, h_f:h_f * 2, w_f:w_f * 2] = sr_batch
            elif i == 6:
                output[:, :, h_f:h_f * 2, w_f * 2:w_f * 3] = sr_batch
            elif i == 7:
                output[:, :, h_f:h_f * 2, w_f * 3:w] = sr_batch
            elif i == 8:
                output[:, :, h_f * 2:h_f * 3, 0:w_f] = sr_batch
            elif i == 9:
                output[:, :, h_f * 2:h_f * 3, w_f:w_f * 2] = sr_batch
            elif i == 10:
                output[:, :, h_f * 2:h_f * 3, w_f * 2:w_f * 3] = sr_batch
            elif i == 11:
                output[:, :, h_f * 2:h_f * 3, w_f * 3:w] = sr_batch
            elif i == 12:
                output[:, :, h_f * 3:h, 0:w_f] = sr_batch
            elif i == 13:
                output[:, :, h_f * 3:h, w_f:w_f * 2] = sr_batch
            elif i == 14:
                output[:, :, h_f * 3:h, w_f * 2:w_f * 3] = sr_batch
            elif i == 15:
                output[:, :, h_f * 3:h, w_f * 3:w] = sr_batch

            del sr_batch
            torch.cuda.empty_cache()

    return output


def main_srrescgan():
    model_path = 'srrescgan/trained_nets_x4/srrescgan_model.pth'  # trained G srfbn of SRResCGAN
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')

    test_img_folder = 'static/uploads/*'  # testset LR images path

    model = Generator(scale=4)  # SRResCGAN generator net
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    test_results = OrderedDict()
    test_results['time'] = []
    idx = 0

    for path_lr in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path_lr))[0]
        print('Img:', idx, base)

        # read images: LR
        img_lr = cv2.imread(path_lr, cv2.IMREAD_COLOR)
        img_LR = torch.from_numpy(np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img_LR.unsqueeze(0)
        img_LR = img_LR.to(device)

        # testing
        t = timer()
        t.tic()

        with torch.no_grad():
            output_SR = _overlap_crop_forward(img_LR, model)
        end_time = t.toc()

        output_sr = output_SR.data.squeeze().float().cpu().clamp_(0, 255).numpy()
        output_sr = np.transpose(np.squeeze(output_sr), (1, 2, 0))

        test_results['time'].append(end_time)
        print('{:->4d}--> {:>10s}, time: {:.4f} sec.'.format(idx, base, end_time))

        # # save images
        save_img_path = os.path.join('./static/downloads')

        if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        cv2.imwrite(os.path.join(save_img_path, 'SRResCGAN_'+path_lr.split('/')[-1]), cv2.cvtColor(output_sr, cv2.COLOR_RGB2BGR))
        del img_LR, img_lr
        del output_SR, output_sr

    avg_time = sum(test_results['time']) / len(test_results['time'])
    print('Avg. Time:{:.4f}'.format(avg_time))
