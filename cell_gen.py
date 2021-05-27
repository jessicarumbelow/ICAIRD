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

# torch.set_printoptions(precision=4, linewidth=300)
# np.set_printoptions(precision=4, linewidth=300)
pd.set_option('display.max_columns', None)


################################################ SET UP SEED AND ARGS AND EXIT HANDLER

def set_seed():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)


set_seed()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--load', default='laced-bird-1076')
parser.add_argument('--encoder', default='resnet34')
parser.add_argument('--note', default='')
parser.add_argument('--lr_decay', type=float, default=1.0)
parser.add_argument('--class_reg', type=float, default=0.0)
parser.add_argument('--input_reg', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--cpu', default=False, action='store_true')
parser.add_argument('--large', default=False, action='store_true')
parser.add_argument('--mean_img_base', default=False, action='store_true')
parser.add_argument('--ones_base', default=False, action='store_true')
parser.add_argument('--zero_base', default=False, action='store_true')
parser.add_argument('--random_base', default=False, action='store_true')
parser.add_argument('--example_base', default=False, action='store_true')
parser.add_argument('--centroids', default=False, action='store_true')
parser.add_argument('--pos_weight', default=False, action='store_true')
parser.add_argument('--example_target', default=False, action='store_true')
parser.add_argument('--abs_input', default=False, action='store_true')
parser.add_argument('--pos_only', default=False, action='store_true')
parser.add_argument('--sig_input', default=False, action='store_true')
parser.add_argument('--sig_output', default=False, action='store_true')
parser.add_argument('--bound_input', default=False, action='store_true')
parser.add_argument('--target_loss', default=False, action='store_true')
parser.add_argument('--filter_vis', default=False, action='store_true')

def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


args = parser.parse_args()

CLASS_LIST = ['0', 'CD8', 'CD3', 'CD20', 'CD8: CD3']
print(CLASS_LIST)
print(args)

device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
os.environ["WANDB_SILENT"] = "true"

proj = "cell_gen"
if args.filter_vis:
    proj = "filter_vis"

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

if args.filter_vis:
    for k, v in new_dict.items():
        print(k, v.shape)
    exit()

net.load_state_dict(new_dict)

net.to(device)

if (torch.cuda.device_count() > 1) and not args.cpu:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

wandb.watch(net)
net.eval()

input_img = torch.ones((dim, dim)) * 0.5

xd, yd = dim // 2, dim // 2

input_img = np.array(Image.open('mean_img.png'))
input_img = normalise(input_img)

ipt_mean = np.mean(input_img)

input_img = torch.ones((dim, dim)) * ipt_mean

if args.mean_img_base:

    crop = input_img.clone()[yd // 2:yd + yd // 2, xd // 2:xd + xd // 2]

    input_img[:yd, :xd] = crop
    input_img[:yd, xd:] = crop
    input_img[yd:, :xd] = crop
    input_img[yd:, xd:] = crop

    input_img = normalise(input_img)

if args.example_base:
    input_img = np.array(Image.open('example_img.png').convert('L'))
    input_img = normalise(input_img)

if args.ones_base:
    input_img = torch.ones((dim, dim))

if args.zero_base:
    input_img = torch.zeros((dim, dim))

if args.random_base:
    input_img = torch.rand((dim, dim))

wandb.log({'mean_img': wandb.Image(input_img)})

target_imgs = torch.zeros(1, len(CLASS_LIST), dim, dim).to(device)

if args.centroids:
    target_imgs[:, 1, yd // 2, xd // 2] = 1
    target_imgs[:, 2, yd // 2, xd + xd // 2] = 1
    target_imgs[:, 3, yd + yd // 2, xd // 2] = 1
    target_imgs[:, 4, yd + yd // 2, xd + xd // 2] = 1
elif args.example_target:
    ipt = np.array(Image.open('example_img.png').convert('L'))
    ipt = normalise(ipt)
    target_imgs = torch.round(torch.softmax(net(torch.tensor(ipt).float().unsqueeze(0).unsqueeze(0).to(device)), dim=1).detach())
else:
    target_imgs[:, 1, :yd, :xd] = 1
    target_imgs[:, 2, :yd, xd:] = 1
    target_imgs[:, 3, yd:, :xd] = 1
    target_imgs[:, 4, yd:, xd:] = 1


input_img = torch.tensor(input_img)

if args.sig_input and (args.mean_img_base or args.example_base):

    input_img = (1/(input_img+0.0001))-1 + 0.0001
    print(input_img.min(), input_img.mean(), input_img.max())
    input_img = -torch.log(input_img)

wandb.log({'target_output': [wandb.Image(target_imgs[:, cls].cpu()) for cls in range(len(CLASS_LIST))]})

input_img = torch.nn.Parameter(input_img.float().unsqueeze(0).unsqueeze(0).to(device))
init_input = input_img.clone().detach()

last_loss = 999999
lr = args.lr
optimizer = optim.SGD([input_img], lr=lr)

for e in range(args.epochs):

    if args.abs_input:
        output_imgs = net(torch.abs(input_img))
    elif args.sig_input:
        output_imgs = net(torch.sigmoid(input_img))
    elif args.bound_input:
        output_imgs = net(torch.clamp(input_img, 0, 1))
    else:
        output_imgs = net(input_img)

    if args.sig_output:
        output_imgs = torch.sigmoid(output_imgs)

    diff_img = input_img - init_input

    pos_out = output_imgs * target_imgs
    neg_out = torch.relu(output_imgs * torch.abs((target_imgs - 1)))

    if args.pos_weight:
        ratio = torch.sum(target_imgs) / (dim * dim * len(CLASS_LIST))
        neg_out *= ratio

    if args.pos_only:
        neg_out *= 0.0

    ipt_reg = args.input_reg * torch.mean(torch.abs(input_img))

    if args.target_loss:
        cls_reg = args.class_reg * torch.var(torch.mean(torch.abs(target_imgs[:, 1:] - output_imgs[:, 1:]), dim=(-1, -2)))
        loss = torch.mean(torch.abs(target_imgs-output_imgs)) + cls_reg + ipt_reg

    else:
        cls_reg = args.class_reg * torch.var(torch.mean(neg_out[:, 1:] - pos_out[:, 1:], dim=(-1, -2)))
        loss = torch.mean(neg_out - pos_out) + cls_reg + ipt_reg

    if e % args.save_freq == 0:

        if (loss >= last_loss) and args.lr_decay != 1.0:
            print('Decaying lr from {} to {}'.format(lr, lr * args.lr_decay))
            lr = lr * args.lr_decay
            optimizer = optim.SGD([input_img], lr=lr)
            last_loss = 999999

        diff_img = input_img - init_input
        print('\n', params_id, e, loss.item(), lr)
        print('output_means: ', torch.mean(output_imgs, dim=(-1, -2)).detach().cpu().numpy())
        print('input mean: ', torch.mean(input_img).item())
        if args.target_loss:
            class_losses = dict(zip(CLASS_LIST, list(torch.mean(torch.abs(target_imgs-output_imgs), dim=(-1, -2)).reshape(-1).detach().cpu().numpy())))
            print('loss: ', torch.mean(torch.abs(target_imgs-output_imgs)).item(), ' cls_reg: ', cls_reg.item(), ' ipt_reg: ', ipt_reg.item())

        else:
            class_losses = dict(zip(CLASS_LIST, list(torch.mean(neg_out - pos_out, dim=(-1, -2)).reshape(-1).detach().cpu().numpy())))
            print('loss: ', torch.mean(neg_out - pos_out).item(), ' cls_reg: ', cls_reg.item(), ' ipt_reg: ', ipt_reg.item())

        print(class_losses)
        res = {}
        res.update(class_losses)
        if args.sig_input:
            res.update({"sig_optim_input":         wandb.Image(torch.sigmoid(input_img[0]).cpu())})
        res.update({
            'model_output':        [wandb.Image(output_imgs[:, cls].cpu()) for cls in range(len(CLASS_LIST))],
            'output_masked_input': wandb.Image((input_img * torch.sum(output_imgs[:, 1:], dim=1)[0]).cpu()),
            "optim_input":         wandb.Image(input_img[0].cpu()),
            "diff_input":          wandb.Image(diff_img[0].cpu()),
            })

        wandb.log(res)

    wandb.log({
        "epoch":      e,
        "lr":         lr,
        "loss":       loss.item(),
        "input_mean": torch.mean(input_img).item()
        })

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    last_loss = loss