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
from PIL import Image, ImageOps
import torchvision.models as models
import argparse
from scipy.ndimage.filters import gaussian_filter
import glob
import shutil
import wandb
import torchvision
import random
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

# torch.set_printoptions(precision=4, linewidth=300)
# np.set_printoptions(precision=4, linewidth=300)
pd.set_option('display.max_columns', None)


################################################ SET UP SEED AND ARGS AND EXIT HANDLER

def set_seed():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    # torch.set_deterministic(True)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


set_seed()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--load', default='logical-smoke-626')
parser.add_argument('--note', default='')
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--cpu', default=False, action='store_true')


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
os.environ["WANDB_SILENT"] = "true"

proj = "cell_gen"

wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

dim = 256

net = smp.Unet(encoder_name=
               'efficientnet-b0', in_channels=1, classes=4)

net.load_state_dict(torch.load('params/' + args.load + ".pth", map_location=torch.device(device)))

net.to(device)

if (torch.cuda.device_count() > 1) and not args.cpu:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

wandb.watch(net)
net.eval()

lr = args.lr

target_imgs = torch.zeros((4, dim, dim)).to(device)

cxs = [0, 0, 0, 128, 128]
cys = [0, 0, 128, 0, 128]

for i in range(4):
    target_imgs[i, cxs[i]:cxs[i] + 128, cys[i]: cys[i] + 128] = 1

target_imgs[0] = torch.abs(torch.sum(target_imgs, dim=0) - 1)
mask = torch.abs(target_imgs[0] - 1)
last_loss = 999999

for c in range(4):
    wandb.log({'{}_target_output'.format(c): wandb.Image(target_imgs[c].cpu())})

input_img = torch.nn.Parameter(torch.zeros((1, dim, dim)).to(device))
optimizer = optim.Adam([input_img], lr=lr)

first_input = input_img.clone()

for e in range(args.epochs):
    diff_img = input_img - first_input

    output_imgs = net(input_img.unsqueeze(0))
    output_imgs = F.sigmoid(output_imgs)
    loss = F.mse_loss(output_imgs[0], target_imgs, input_img, args.reg)

    if e % args.save_freq == 0:

        for c in range(4):
            wandb.log({'{}_model_output'.format(c): wandb.Image(output_imgs[0, c].cpu())})
        wandb.log({
            "optim_input": wandb.Image(input_img[0].cpu()),
            "diff_input":  wandb.Image(diff_img[0].cpu()),
            "loss":        loss,
            "epoch":       e,
            "lr":          lr
            })

        if (loss == 0) or (loss == last_loss):
            break

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(e, loss.item(), lr)

    if loss >= last_loss:
        lr *= args.lr_decay
        optimizer = optim.Adam([input_img], lr=lr)
        last_loss = 999999

    last_loss = loss
