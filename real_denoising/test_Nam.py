import numpy as np
import os
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import utils
from DRANet import DRANet
from skimage import img_as_ubyte
import scipy.io as sio
from utils import utils_image as util

parser = argparse.ArgumentParser(description='Image Denoising using DRANet')
parser.add_argument('--input_dir', default='./Datasets/color/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/color/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

result_dir = os.path.join(args.result_dir)
utils.mkdir(result_dir)

model_restoration = DRANet(in_nc=3, out_nc=3, nc=128, bias=False)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

L_path = os.path.join(args.input_dir, 'input')
H_path = os.path.join(args.input_dir, 'target')
E_path = os.path.join(args.result_dir)   # E_path, for Estimated images

test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []

L_paths = util.get_image_paths(L_path)
H_paths = util.get_image_paths(H_path)

for idx, img in enumerate(L_paths):

    img_name, ext = os.path.splitext(os.path.basename(img))
    img_L = util.imread_uint(img, n_channels=3)
    img_L = util.uint2single(img_L)

    #img_L = util.single2tensor4(img_L)

    img_L = torch.from_numpy(img_L).unsqueeze(0).permute(0, 3, 1, 2).cuda()

    img_E = model_restoration(img_L)

    img_E = torch.clamp(img_E, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)

    #img_E = util.tensor2uint(img_E)
    img_E = img_E.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    img_E = np.uint8((img_E * 255.0).round())

    img_H = util.imread_uint(H_paths[idx], n_channels=3)
    img_H = img_H.squeeze()

    psnr = util.calculate_psnr(img_E, img_H, border=1)
    ssim = util.calculate_ssim(img_E, img_H, border=1)
    test_results['psnr'].append(psnr)
    test_results['ssim'].append(ssim)
    print('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

    util.imsave(img_E, os.path.join(E_path, img_name+ext))

ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
print('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format('DRANet', ave_psnr, ave_ssim))




