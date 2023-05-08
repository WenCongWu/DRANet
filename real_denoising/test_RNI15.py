import numpy as np
import os
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
import utils
from DCDNet import DCDNet
from skimage import img_as_ubyte
import scipy.io as sio
from utils import utils_image as util

parser = argparse.ArgumentParser(description='Image Denoising using MPRNet')
parser.add_argument('--input_dir', default='./Datasets/RNI15/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/RNI15/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

result_dir = os.path.join(args.result_dir)
utils.mkdir(result_dir)

model_restoration = DCDNet(in_nc=3, out_nc=3, nc=64, bias=False)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

L_path = os.path.join(args.input_dir, '')
E_path = os.path.join(args.result_dir)   # E_path, for Estimated images

L_paths = util.get_image_paths(L_path)

for idx, img in enumerate(L_paths):

    img_name, ext = os.path.splitext(os.path.basename(img))
    img_L = util.imread_uint(img, n_channels=3)
    img_L = util.uint2single(img_L)

    img_L = torch.from_numpy(img_L).unsqueeze(0).permute(0, 3, 1, 2).cuda()

    img_E, noise_level = model_restoration(img_L)

    img_E = torch.clamp(img_E, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)

    img_E = img_E.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    img_E = np.uint8((img_E * 255.0).round())

    util.imsave(img_E, os.path.join(E_path, img_name+ext))





