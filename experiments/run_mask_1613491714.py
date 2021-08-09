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
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--conv_thresh', type=float, default=0.01)
parser.add_argument('--save_freq', default=1000, type=int)
parser.add_argument('--subset', default=9999999, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--preview', default=False, action='store_true')
parser.add_argument('--model', default='UNet2')

args = parser.parse_args()

time_taken = 0
epoch_loss = 0
epoch = 0

params_id = str(int(time.time()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_results():
    results = {'params_id': params_id, 'time': time_taken, 'epoch_loss': epoch_loss, 'epoch': epoch}
    results = {**results, **vars(args)}
    print(results)
    results = pd.DataFrame(results, index=[0])
    results.to_csv('run_mask_results.csv', index=False, mode='a+', header=False)
    shutil.copy2('run_mask.py', 'experiments/run_mask_{}.py'.format(params_id))


atexit.register(save_results)


################################################ IMG HELPERS


def show_im(im):
    d = im.shape[-1]
    fig, ax = plt.subplots()
    im = im.reshape(-1, d)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.close(fig)


def save_im(im, name='image'):
    d = im.shape[-1]
    fig, ax = plt.subplots()
    im = im.reshape(-1, d)
    plt.imshow(im, cmap='gray')
    plt.savefig('lc_imgs/' + name + '.png')
    plt.close(fig)


def standardise(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 0.00000001)
    return img


def normalise(img):
    img = (img - np.min(img)) / max(np.max(img) - np.min(img), 0.0001)
    return img


def whiten(img):
    img = img - img.mean()
    cov = np.dot(img.T, img)
    d, vec = np.linalg.eigh(cov)
    diag = np.diag(1 / np.sqrt(d + 0.0001))
    w = np.dot(np.dot(vec, diag), vec.T)

    return np.dot(img, w)


################################################ SETTING UP DATASET


if not os.path.exists('sample_paths.csv'):
    print('Building data paths...')

    data = glob.glob("data/exported_tiles/*")

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


def prep1(img):
    img = normalise(img)
    mask = img.copy()
    mask[mask > 0.2] = 0
    mask[mask > 0] = 1
    mask = gaussian_filter(mask, sigma=0.2)
    mask[mask < 1] = 0.1
    mask = gaussian_filter(mask, sigma=1)
    img *= mask
    return


def prep(img):
    img = normalise(img)
    mask = img.copy()
    mask[mask > 0.2] = 0
    mask[mask > 0] = 1
    mask = gaussian_filter(mask, sigma=0.2)
    mask[mask < 1] = 0
    del_mask = mask.copy()
    mask = gaussian_filter(mask, sigma=1)
    img *= mask

    return img, del_mask


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
        mask[mask > 0] = 1

        return np.expand_dims(img.astype(np.float32), 0), np.expand_dims(mask.astype(np.float32), 0), name


class lc_seg_tiles_mc(Dataset):

    def __init__(self, subset=-1):
        self.samples = pd.read_csv('sample_paths.csv')[:subset]
        self.num_classes = 5

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
        #oh = F.one_hot(torch.Tensor(mask).to(torch.int64), num_classes=self.num_classes)

        return np.expand_dims(img.astype(np.float32), 0), mask.astype(np.float32), name


################################################ MODELS


class fcn_resnet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.fcn = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        self.ch_conv = nn.Conv2d(1, 3, 1)

    def forward(self, x):
        x = self.ch_conv(x)
        x = self.fcn(x)
        return x


class fcn_resnet101(nn.Module):

    def __init__(self):
        super().__init__()
        self.fcn = models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        self.ch_conv = nn.Conv2d(1, 3, 1)

    def forward(self, x):
        x = self.ch_conv(x)
        x = self.fcn(x)
        return x



"UNet implementation from: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py"

def double_conv(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
            )


class UNet1(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()
        self.ch_conv = nn.Conv2d(1, 3, 1)
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        x = self.ch_conv(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


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


class UNet2(nn.Module):

    def __init__(self, n_channels=1, n_classes=5, bilinear=True):
        super(UNet2, self).__init__()
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
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



################################################ TRAINING


def run_epochs(net, dataloader, criterion, optimizer, num_epochs, path, save_freq=100, train=True):
    mode = 'training' if train else 'validating'
    global time_taken
    global epoch_loss
    global epoch
    last_epoch_loss = 999999
    start_time = time.time()

    if not train:
        net.eval()

    for epoch in range(num_epochs):
        sum_epoch_loss = 0.0
        running_loss = 0.0
        last_running_loss = 999999
        for i, data in enumerate(dataloader):
            inputs, masks, name = data

            inputs = inputs.to(device)
            masks = masks.to(device)
            if 'fcn_resnet' in args.model:
                outputs = net(inputs)['out']
            else:
                outputs = net(inputs)

            print(outputs.shape, masks.shape)
            exit()

            loss = criterion(outputs, masks)

            print('({}) Epoch: {} Batch: {}/{} Batch Loss {}'.format(mode, epoch, i, len(dataloader), loss.item()))

            if True in torch.isnan(loss):
                print('NaaaaaaaaaaN!')
                return net, epoch_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            time_taken = np.round(time.time() - start_time)

            running_loss += loss.item()
            sum_epoch_loss += loss.item()
            epoch_loss = sum_epoch_loss / (i + 1)

            if args.preview:
                im = inputs[0].detach().cpu().numpy()
                o = outputs[0].detach().cpu().numpy()
                m = masks[0].detach().cpu().numpy()
                save_im(im, 'input')
                save_im(m, 'mask')
                save_im(o, 'output')

            if (i > 0) and (i % save_freq == 0):
                print('\n({}) Epoch: {} Batch: {}/{} Last running loss: {} Current running loss: {} over last {} batches.\n'
                      .format(mode, epoch, i, len(dataloader), last_running_loss, running_loss / save_freq, save_freq))
                last_running_loss = running_loss / save_freq
                running_loss = 0.0
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


dataset = lc_seg_tiles_mc(args.subset)

net = eval(args.model)()
set_seed(seed)

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model has {} parameters".format(num_params))

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

net.to(device)
num_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

train_size = int(0.7 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = int(0.5 * val_test_size)
test_size = val_test_size - val_size

train_data, val_test_data = torch.utils.data.random_split(dataset, [train_size, val_test_size])
val_data, test_data = torch.utils.data.random_split(val_test_data, [val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

if args.load is not '':
    net.load_state_dict(torch.load('params/' + args.load + ".pth", map_location=torch.device(device)))

if args.train:
    print('Training network...')
    net = run_epochs(net, train_loader, criterion, optimizer, num_epochs, 'params/' + params_id + '.pth', save_freq=min(train_size, args.save_freq))


################################################ VALIDATION

else:
    print('Evaluating network...')
    net.load_state_dict(torch.load('params/' + args.load + '.pth'))
    _ = run_epochs(net, eval_loader, criterion, None, 1, None, train=False, save_freq=args.save_freq)
