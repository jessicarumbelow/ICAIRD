import wandb
import numpy as np
import torch.optim as optim
import pandas as pd
import os
import argparse
import random
import torchvision.transforms.functional as TF
import torch
from utils.utils import *
from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
from models.resnet_custom import resnet50_baseline

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

import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--kernel_size', type=int, default=5)
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
parser.add_argument('--normalise', default=False, type=bool)
parser.add_argument('--sgd', default=False, type=bool)
parser.add_argument('--relu', default=False, type=bool)
parser.add_argument('--ckpt', default='../results/Endometrial/Results_Feb2022/Endo_all_classes_CLAM_sb_small')
parser.add_argument('--model_type', default='CLAM_SB')


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


args = parser.parse_args()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
os.environ["WANDB_SILENT"] = "true"

proj = "icaird_feature_vis"
run = wandb.init(project=proj, entity="jessicamarycooper", config=args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = ['malignant', 'insufficient', 'other_benign']

att_model = eval(args.model_type)(n_classes=len(labels))

ckpt_path = args.ckpt
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

ckpt_clean = {}
for key in ckpt.keys():
    ckpt_clean.update({key.replace('3', '2'): ckpt[key]})

att_model.load_state_dict(ckpt_clean)

att_model.relocate()
att_model.eval()

feature_model = resnet50_baseline(pretrained=True).to(device)
feature_model.eval()

def model(x):
    return att_model(feature_model(x))[0]

dim = 256
input_img = torch.ones((1, 3, dim, dim))
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

init_input = input_img.clone().to(device)
cls_input = torch.nn.Parameter(torch.tensor(input_img).float().to(device))

if args.sgd:
    optimizer = optim.SGD([cls_input], lr=lr)
else:
    optimizer = optim.Adam([cls_input], lr=lr)


for l in range(len(labels)):
    label = labels[l]
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

        output = model(cls_input)[:,l]

        diff_img = cls_input - init_input

        loss = -output + args.reg * torch.mean(torch.abs(cls_input))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        last_loss = loss

        print('\n', e, loss.item(), lr, labels[l])

        if e % args.save_freq == 0:

            if (loss >= last_loss) and args.lr_decay != 1.0:
                print('Decaying lr from {} to {}'.format(lr, lr * args.lr_decay))
                lr = lr * args.lr_decay
                optimizer = optim.Adam([cls_input], lr=lr)
                last_loss = 999999

            diff_img = cls_input - init_input

            res = {}
            res.update({
                "{} optim_input".format(label): wandb.Image(cls_input[0].cpu()),
                '{} mean_input'.format(label): torch.mean(cls_input),
                '{} min_input'.format(label): torch.min(cls_input),
                '{} max_input'.format(label): torch.max(cls_input)
                })
            wandb.log(res)

        wandb.log({
            "{} epoch".format(label): e, "{} lr".format(label): lr, "{} loss".format(label): loss.item(),
            })