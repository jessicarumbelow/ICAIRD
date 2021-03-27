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
from hipe import hierarchical_perturbation
from hipe import blur

torch.set_printoptions(precision=4, linewidth=300)
np.set_printoptions(precision=4, linewidth=300)
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
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--conv_thresh', type=float, default=0)
parser.add_argument('--load', default='blooming-puddle-89')
parser.add_argument('--note', default='')
parser.add_argument('--cls', type=int, default=0)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--save_freq', default=100, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "cell_gen"

wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

dim = 256

ap = dict(
        pooling='avg',  # one of 'avg', 'max'
        classes=1,  # define number of output labels
        )

net = smp.Unet(encoder_name='resnet34', in_channels=1, classes=5, aux_params=ap)

net.load_state_dict(torch.load('params/' + args.load + ".pth", map_location=torch.device(device)))

net.to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

wandb.watch(net)
net.eval()

input_img = torch.nn.Parameter(torch.rand((1, 1, dim, dim), device=device))
lr = args.lr

# http://numpy-discussion.10968.n7.nabble.com/Drawing-circles-in-a-numpy-array-td4720.html
radius = 12
a = np.zeros((256, 256)).astype('uint8')
cx, cy = 128, 128  # The center of circle
y, x = np.ogrid[-radius: radius, -radius: radius]
index = x ** 2 + y ** 2 <= radius ** 2
a[cy - radius:cy + radius, cx - radius:cx + radius][index] = 1

target_img = torch.Tensor(a).unsqueeze(0).to(device)
last_loss = 999999
cls = args.cls
if cls == 0:
    target_img = torch.abs(target_img - 1)

wandb.log({'target_output': wandb.Image(target_img[0].cpu())})

optimizer = optim.Adam([input_img], lr=lr)

for e in range(args.epochs):
    output_imgs, _ = net(input_img)

    loss = torch.mean(torch.abs(output_imgs[0, cls] - target_img))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(cls, '-', loss)

    if loss >= last_loss:
        lr *= args.lr_decay
        optimizer = optim.Adam([input_img], lr=lr)
        print('Learning rate: ', lr)

    last_loss = loss

    if e % args.save_freq == 0:
        wandb.log({
            "model_output":   wandb.Image(output_imgs[0, cls].cpu()), "optim_input": wandb.Image(input_img[0].cpu()), "loss": loss,
            "epoch":          e,
            "lr".format(cls): lr
            })

    if loss == 0:
        break
