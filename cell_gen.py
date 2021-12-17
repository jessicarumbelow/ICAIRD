import wandb

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import io
from torch.utils.data import Dataset
import cv2
import torchvision.models as models
import argparse
from scipy.ndimage.filters import gaussian_filter
import glob
import shutil
import torchvision
import random
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from skimage.draw import disk
from PIL import Image
import torchvision.transforms.functional as TF

# torch.set_printoptions(precision=4, linewidth=300)
# np.set_printoptions(precision=4, linewidth=300)
from scipy.ndimage.filters import gaussian_filter

pd.set_option('display.max_columns', None)


################################################ SET UP SEED AND ARGS AND EXIT HANDLER


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--load', default='fiery-lake-1971')
parser.add_argument('--encoder', default='resnet50')
parser.add_argument('--note', default='')
parser.add_argument('--lr_decay', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--cpu', default=False, type=bool)
parser.add_argument('--base', default=0.0, type=float)
parser.add_argument('--random_transform', default=0, type=int)
parser.add_argument('--reg', default=0, type=float)
parser.add_argument('--rot', default=False, type=bool)
parser.add_argument('--jit', default=0, type=float)
parser.add_argument('--blur', default=False, type=bool)
parser.add_argument('--centroid', default=False, type=bool)
parser.add_argument('--normalise', default=False, type=bool)
parser.add_argument('--sgd', default=False, type=bool)
parser.add_argument('--relu', default=False, type=bool)



def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


args = parser.parse_args()

CLASS_LIST = ['OTHER', 'CD3']
print(CLASS_LIST)
print(args)

device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
os.environ["WANDB_SILENT"] = "true"

proj = "cell_gen"

wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

dim = 256

net = smp.Unet(encoder_name=args.encoder, in_channels=1, classes=len(CLASS_LIST))

state_dict = torch.load('params/' + args.load + ".pth", map_location=torch.device(device))

new_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        new_k = k.split('module.')[-1]
        new_dict[new_k] = v
    else:
        new_dict[k] = v

net.load_state_dict(new_dict)

net.to(device)

if (torch.cuda.device_count() > 1) and not args.cpu:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

wandb.watch(net)
net.eval()

input_img = torch.ones((1, 1, dim, dim)) * 0.5
input_img *= args.base


def jitter(im, j):
    if j > 0:
        j = np.random.randint(1, j + 1)
        dir = np.random.randint(0, 4)
        if dir == 0:
            im[:, :, :-j, :-j] = im[:, :, j:, j:]
        if dir == 1:
            im[:, :, j:, :-j] = im[:, :, :-j, j:]
        if dir == 2:
            im[:, :, j:, j:] = im[:, :, :-j, :-j]
        if dir == 3:
            im[:, :, :-j, j:] = im[:, :, j:, :-j]

    return im


def rotate(im):
    deg = [0, 90, 180, 270][np.random.randint(0, 4)]
    im = TF.rotate(im, deg)
    return im


def blur(im, k, s):
    im = TF.gaussian_blur(im, kernel_size=k, sigma=s)
    return im


def random_transform(im):
    im = jitter(im, args.jit)
    if args.rot:
        im = rotate(im)
    if args.blur:
        im = blur(im, args.kernel_size, args.sigma)
    return im


last_loss = 999999
lr = args.lr

target_img = torch.zeros(1, 2, dim, dim).to(device)
target_img[:, 1] = 1
init_input = input_img.clone().to(device)
cls_input = torch.nn.Parameter(torch.tensor(input_img).float().to(device))
if args.sgd:
    optimizer = optim.SGD([cls_input], lr=lr)
else:
    optimizer = optim.Adam([cls_input], lr=lr)

cls = 'CD3'
wandb.log({'Class': cls})

for e in range(args.epochs):

    if args.relu:
        cls_input = torch.relu(cls_input)

    if args.normalise:
        cls_input = cls_input.clone().detach()
        cls_input = normalise(cls_input)
        cls_input = torch.nn.Parameter(torch.tensor(cls_input).float().to(device))
        if args.sgd:
            optimizer = optim.SGD([cls_input], lr=lr)
        else:
            optimizer = optim.Adam([cls_input], lr=lr)

    if args.random_transform > 0 and e % args.random_transform == 0:
        cls_input = cls_input.clone().detach()
        cls_input = random_transform(cls_input.clone().detach())
        cls_input = torch.nn.Parameter(torch.tensor(cls_input).float().to(device))
        if args.sgd:
            optimizer = optim.SGD([cls_input], lr=lr)
        else:
            optimizer = optim.Adam([cls_input], lr=lr)

    output_imgs = net(cls_input)

    diff_img = cls_input - init_input

    if args.centroid:
        loss = -output_imgs[:, 1, dim // 2, dim // 2] + args.reg * torch.mean(torch.abs(cls_input))
    else:
        loss = -torch.mean(output_imgs[:, 1]) + args.reg * torch.mean(torch.abs(cls_input))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    last_loss = loss

    print('\n', params_id, e, loss.item(), lr, cls)

    if e % args.save_freq == 0:

        if (loss >= last_loss) and args.lr_decay != 1.0:
            print('Decaying lr from {} to {}'.format(lr, lr * args.lr_decay))
            lr = lr * args.lr_decay
            optimizer = optim.Adam([cls_input], lr=lr)
            last_loss = 999999

        diff_img = cls_input - init_input

        res = {}
        res.update({
            '{} model_output'.format(cls): wandb.Image(output_imgs[:, 1]),
            '{} output_masked_input'.format(cls): wandb.Image(normalise(output_imgs[:, 1]).detach().cpu() * normalise(cls_input[0]).detach().cpu()),
            "{} optim_input".format(cls): wandb.Image(cls_input[0].cpu()),
            "{} diff_input".format(cls): wandb.Image(diff_img[0].cpu()),
            '{} mean_input'.format(cls): torch.mean(cls_input),
            '{} min_input'.format(cls): torch.min(cls_input),
            '{} max_input'.format(cls): torch.max(cls_input)
            })
        wandb.log(res)

    wandb.log({
        "{} epoch".format(cls): e, "{} lr".format(cls): lr, "{} loss".format(cls): loss.item(),
        })