


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
from O365 import Account
import argparse
import time
import atexit
from scipy.ndimage.filters import gaussian_filter
import glob
import shutil

torch.set_printoptions(precision=4, linewidth=300)
np.set_printoptions(precision=4, linewidth=300)

################################################ SET UP SEED AND ARGS AND EXIT HANDLER

seed = 99


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--conv_thresh', type=float, default=0.01)
parser.add_argument('--save_freq', default=1000, type=int)
parser.add_argument('--subset', default=9999999, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--preview', default=False, action='store_true')
parser.add_argument('--model', default='UNet2')
parser.add_argument('--cls', type=int, default=-1)

args = parser.parse_args()

time_taken = 0
epoch_loss = 0
epoch = 0

params_id = str(int(time.time()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = {'params_id': params_id}
results.update({'val_' + s: 0 for s in ['tp', 'fp', 'tn', 'fn', 'prec', 'rec', 'f1', 'time', 'epoch_loss']})


def save_results():
    global results

    results = {**results, **vars(args)}
    results = pd.DataFrame(results, index=[0])
    results.to_csv('run_mask_results.csv', index=False, mode='a+', header=False)
    shutil.copy2('run_mask.py', 'experiments/run_mask_{}.py'.format(params_id))
    print(results)


atexit.register(save_results)


################################################ HELPERS


def show_im(im):
    d = im.shape[-1]
    fig, ax = plt.subplots()
    im = im.reshape(-1, d)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.close(fig)


def save_im(im, name='image'):
    d = im.shape[-1]
    im = im.reshape(-1, d, d)

    for cls in range(im.shape[0]):
        fig, ax = plt.subplots()
        img = im[cls]
        plt.imshow(img, cmap='gray')
        plt.savefig('lc_imgs/' + name + '_' + str(cls) + '.png')
        plt.close(fig)


def standardise(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 0.00000001)
    return img


def normalise(x):
    x = (x - x.min()) / max(x.max() - x.min(), 0.0001)
    return x


def whiten(img):
    img = img - img.mean()
    cov = np.dot(img.T, img)
    d, vec = np.linalg.eigh(cov)
    diag = np.diag(1 / np.sqrt(d + 0.0001))
    w = np.dot(np.dot(vec, diag), vec.T)

    return np.dot(img, w)


def scores(target, output):
    output = torch.round(output)
    conf = output / target
    tp, fp, tn, fn = torch.sum(conf == 1).item(), torch.sum(conf == float('inf')).item(), torch.sum(torch.isnan(conf)).item(), torch.sum(conf == 0).item()

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    f1 = 2 * ((precision * recall) / max(precision + recall, 0.0001))

    return np.array([tp, fp, tn, fn, precision, recall, f1])


################################################ SETTING UP DATASET


if not os.path.exists('sample_paths.csv'):
    print('Building data paths...')

    data = glob.glob("data/new_exported_tiles/*/*")

    samples = {}
    for d in data:
        dn = d.split(']')[0] + ']'
        sample = samples[dn] if dn in samples else {}
        if '-labelled' in d:
            sample['Mask'] = d
        else:
            sample['Img'] = d
        samples[dn] = sample

    samples = pd.DataFrame.from_dict(samples, orient='index')
    samples = samples.dropna()
    samples.to_csv('sample_paths.csv', index=False)
    print('\n', len(samples))
    print('DONE!')


class lc_seg_tiles_dir(Dataset):

    def __init__(self, subset=-1):
        self.samples = pd.read_csv('sample_paths.csv')[:subset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples.iloc[index]
        dim = 256

        name = s['Img'].split('.')[0]

        img = np.array(Image.open(s['Img']))[:dim, :dim]
        img = np.pad(img, ((0, dim - img.shape[0]), (0, dim - img.shape[1])), 'minimum')

        mask = np.array(Image.open(s['Mask']))[:dim, :dim]
        mask = np.pad(mask, ((0, dim - mask.shape[0]), (0, dim - mask.shape[1])), 'minimum')

        return np.expand_dims(img.astype(np.float32), 0), np.expand_dims(mask.astype(np.float32), 0), name


class lc_seg_tiles_bc(Dataset):

    def __init__(self, subset=-1):
        self.samples = pd.read_csv('sample_paths.csv')[:subset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples.iloc[index]
        dim = 256

        name = s['Img'].split('.')[0]

        img = np.array(Image.open(s['Img']))[:dim, :dim]
        img = np.pad(img, ((0, dim - img.shape[0]), (0, dim - img.shape[1])), 'minimum')

        mask = np.array(Image.open(s['Mask']))[:dim, :dim]
        mask = np.pad(mask, ((0, dim - mask.shape[0]), (0, dim - mask.shape[1])), 'minimum')

        if args.cls > 0:
            mask[mask != args.cls] = 0
        mask[mask > 0] = 1

        return np.expand_dims(img.astype(np.float32), 0), np.expand_dims(mask.astype(np.float32), 0), name


class lc_seg_tiles_mc(Dataset):

    def __init__(self, subset=-1):
        self.samples = pd.read_csv('sample_paths.csv')[:subset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples.iloc[index]
        dim = 256

        name = s['Img'].split('.')[0]

        img = np.array(Image.open(s['Img']))[:dim, :dim]
        img = np.pad(img, ((0, dim - img.shape[0]), (0, dim - img.shape[1])), 'minimum')

        mask = np.array(Image.open(s['Mask']))[:dim, :dim]
        mask = np.pad(mask, ((0, dim - mask.shape[0]), (0, dim - mask.shape[1])), 'minimum')
        # oh = F.one_hot(torch.Tensor(mask).to(torch.int64), num_classes=5)

        return np.expand_dims(img.astype(np.float32), 0), mask.astype(np.long), name


################################################ MODELS


""" Parts of the U-Net model 
https://github.com/milesial/Pytorch-UNet
"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
                )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return torch.sum(x5, dim=tuple(range(1, len(x5.shape))))


class ResNet50(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        self.rn = models.resnet50(num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.rn(x)
        return x


class WideResNet50(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        self.rn = models.wide_resnet50_2(num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.rn(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        self.rn = models.alexnet(num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.rn(x)
        return x



class SimpleFC(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(96, 96, kernel_size=5, padding=2)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x, _ = x.max(dim=1)
        h = x.register_hook(self.activations_hook)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return x


################################################ TRAINING


def run_epochs(net, dataloader, criterion, optimizer, num_epochs, path, save_freq=100, train=True):
    mode = 'train' if train else 'val'
    global results
    last_epoch_loss = 999999

    start_time = time.time()

    if not train:
        net.eval()

    for epoch in range(num_epochs):
        sum_epoch_loss = 0.0
        sum_epoch_scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for i, data in enumerate(dataloader):
            inputs, masks, name = data
            inputs = inputs.to(device)
            masks = masks.to(device)
            if 'fcn_resnet' in args.model:
                outputs = net(inputs)['out']
            else:
                outputs = torch.sigmoid(net(inputs))

            loss = criterion(outputs, masks)
            batch_scores = scores(masks, outputs)

            print('({}) Epoch: {} Batch: {}/{} Batch Loss {}'.format(mode, epoch, i, len(dataloader), loss.item()))
            print(dict(zip(['tp', 'fp', 'tn', 'fn', 'prec', 'rec', 'f1'], batch_scores)))

            if True in torch.isnan(loss):
                print('NaaaaaaaaaaN!')
                return net

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                results['training_epochs'.format(mode)] = epoch

            outputs = torch.round(outputs)

            time_taken = np.round(time.time() - start_time)

            sum_epoch_loss += loss.item()
            sum_epoch_scores += batch_scores

            epoch_loss = sum_epoch_loss / (i + 1)
            epoch_scores = sum_epoch_scores / (i + 1)

            results['{}_time'.format(mode)] = time_taken
            results['{}_epoch_loss'.format(mode)] = epoch_loss
            results.update(dict(zip(['{}_'.format(mode) + s for s in ['tp', 'fp', 'tn', 'fn', 'prec', 'rec', 'f1']], epoch_scores)))

            if args.preview:
                im = inputs[0].detach().cpu().numpy()
                o = outputs[0].detach().cpu().numpy()
                m = masks[0].detach().cpu().numpy()
                save_im(im, 'input')
                save_im(m, 'mask')
                save_im(o, 'output')

            if (i > 0) and (i % save_freq == 0):
                print('Saving images...')
                im = inputs[0].detach().cpu().numpy()
                o = outputs[0].detach().cpu().numpy()
                m = masks[0].detach().cpu().numpy()
                progress = 'epoch_{}_batch_{}of{}_'.format(epoch, i, len(dataloader))
                save_im(im, progress + '{}_{}_image'.format(mode, params_id))
                save_im(m, progress + '{}_{}_mask'.format(mode, params_id))
                save_im(o, progress + '{}_{}_output'.format(mode, params_id))
                if train:
                    torch.save(net.state_dict(), path)

        print('Completed epoch {}. Avg epoch loss: {}'.format(epoch, sum_epoch_loss / (i + 1)))
        if train:
            torch.save(net.state_dict(), path)

        if (epoch_loss < args.conv_thresh) or (epoch_loss > last_epoch_loss):
            return net
        last_epoch_loss = epoch_loss

    print('Finished Training')
    return net


dataset = lc_seg_tiles_bc(args.subset)
net = eval(args.model)()
criterion = nn.BCELoss()

"""
# multiclass segmentation
dataset = lc_seg_tiles_mc(args.subset)
net = eval(args.model)(n_channels=1, n_classes=5)
criterion = nn.CrossEntropyLoss()

"""

set_seed(seed)

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model has {} parameters".format(num_params))

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

net.to(device)
num_epochs = args.epochs
optimizer = optim.Adam(net.parameters(), lr=args.lr)

train_size = int(0.7 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = int(0.5 * val_test_size)
test_size = val_test_size - val_size

train_data, val_test_data = torch.utils.data.random_split(dataset, [train_size, val_test_size])
val_data, test_data = torch.utils.data.random_split(val_test_data, [val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True)

print('{} training samples, {} validation samples...'.format(train_size, val_size))

if args.load is not '':
    net.load_state_dict(torch.load('params/' + args.load + ".pth", map_location=torch.device(device)))

if args.train:
    print('Training network...')
    net = run_epochs(net, train_loader, criterion, optimizer, num_epochs, 'params/' + params_id + '.pth', save_freq=min(train_size, args.save_freq))

################################################ VALIDATION

print('Evaluating network...')
net.load_state_dict(torch.load('params/' + args.load + '.pth'))
_ = run_epochs(net, eval_loader, criterion, None, 1, None, train=False, save_freq=args.save_freq)
