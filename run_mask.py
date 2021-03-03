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

torch.set_printoptions(precision=4, linewidth=300)
np.set_printoptions(precision=4, linewidth=300)


################################################ SET UP SEED AND ARGS AND EXIT HANDLER

def set_seed():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    #torch.set_deterministic(True)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


set_seed()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--conv_thresh', type=float, default=0)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--subset', default=9999999999, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--model', default='UNet2')
parser.add_argument('--cls', type=int, default=-1)
parser.add_argument('--augment', default=False, action='store_true')

args = parser.parse_args()

epoch_loss = 0
epoch = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="immune-seg", entity="jessicamarycooper", config=args)
args = wandb.config

params_id = wandb.run.name
print(params_id)
wandb.run.save()


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


def scores(output, target):
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

    return np.array([precision, recall, f1])


def gentle_scores(output, target):
    tp = torch.sum(target * output)
    tn = torch.sum((1 - target) * (1 - output))
    fp = torch.sum((1 - target) * output)
    fn = torch.sum(target * (1 - output))

    p = tp / (tp + fp + 0.0001)
    r = tp / (tp + fn + 0.0001)

    f1 = 2 * p * r / (p + r + 0.0001)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

    return torch.Tensor([p, r, torch.mean(f1)])


def f1_loss(output, target):
    tp = torch.sum(target * output)
    tn = torch.sum((1 - target) * (1 - output))
    fp = torch.sum((1 - target) * output)
    fn = torch.sum(target * (1 - output))

    p = tp / (tp + fp + 0.0001)
    r = tp / (tp + fn + 0.0001)

    f1 = 2 * p * r / (p + r + 0.0001)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

    return 1 - torch.mean(f1)


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

        img, mask = torch.Tensor(img.astype(np.float32)).unsqueeze(0), torch.Tensor(mask.astype(np.float32)).unsqueeze(0)

        if args.augment:
            angle = random.randint(-180, 180)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

            if random.choice([0, 1]) == 1:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.choice([0, 1]) == 1:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        return img, mask, name


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


class SegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """

    def __init__(self, num_classes=1, n_init_features=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SegNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(feat)


class _Encoder(nn.Module):

    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """

    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)


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

    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
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
    mode = 'train' if train else 'val'
    results = {}
    if not train:
        net.eval()

    for epoch in range(num_epochs):
        sum_epoch_loss = 0.0
        sum_epoch_scores = torch.zeros(3)

        for i, data in enumerate(dataloader):
            inputs, masks, name = data
            inputs = inputs.to(device)
            masks = masks.to(device)

            if 'fcn_resnet' in args.model:
                outputs = torch.sigmoid(net(inputs)['out'])
            else:
                outputs = torch.sigmoid(net(inputs))

            # Balancing loss (for binary task only!)
            # pos_cov = torch.sum(masks)/torch.numel(masks)
            # pos_weight = (1 - pos_cov) / pos_cov

            loss = criterion(outputs, masks)#, pos_weight=pos_weight)
            batch_scores = scores(outputs, masks)

            print('({}) Epoch: {}/{} Batch: {}/{} Batch Loss {}'.format(mode, epoch, num_epochs, i, len(dataloader), loss.item()))
            print(dict(zip(['prec', 'rec', 'f1'], batch_scores)))

            sum_epoch_loss += loss.item()
            sum_epoch_scores += batch_scores
            epoch_loss = sum_epoch_loss / (i + 1)
            epoch_scores = sum_epoch_scores / (i + 1)

            results = {'{}/batch_loss'.format(mode): loss.item(), '{}/epoch_loss'.format(mode): epoch_loss}
            results.update(dict(zip(['{}/epoch_'.format(mode) + s for s in ['prec', 'rec', 'f1']], epoch_scores)))
            results.update(dict(zip(['{}/batch_'.format(mode) + s for s in ['prec', 'rec', 'f1']], batch_scores)))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % save_freq == 0:
                results["{}/inputs".format(mode)] = [wandb.Image(i) for i in inputs]
                results["{}/outputs".format(mode)] = [wandb.Image(i) for i in outputs]
                results["{}/masks".format(mode)] = [wandb.Image(i) for i in masks]

            wandb.log(results)

        print('Completed epoch {}. Avg epoch loss: {}'.format(epoch, sum_epoch_loss / (i + 1)))

        if train:
            print('Saving model...')
            torch.save(net.state_dict(), path)
            print('Evaluating network...')
            with torch.no_grad():
                _ = run_epochs(net, eval_loader, criterion, None, 1, None, train=False, save_freq=args.save_freq)

    return net


dataset = lc_seg_tiles_bc(args.subset)
net = eval(args.model)()
criterion = f1_loss

"""
# multiclass segmentation
dataset = lc_seg_tiles_mc(args.subset)
net = eval(args.model)(n_channels=1, n_classes=5)
criterion = nn.CrossEntropyLoss()

"""

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

net.to(device)
num_epochs = args.epochs
optimizer = optim.Adam(net.parameters(), lr=args.lr)
wandb.watch(net)

train_size = int(0.8 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = int(0.5 * val_test_size)
test_size = val_test_size - val_size

train_data, val_test_data = torch.utils.data.random_split(dataset, [train_size, val_test_size])
val_data, test_data = torch.utils.data.random_split(val_test_data, [val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)
eval_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)

print('{} training samples, {} validation samples...'.format(train_size, val_size))

if args.load is not '':
    net.load_state_dict(torch.load('params/' + args.load + ".pth", map_location=torch.device(device)))

if args.train:
    print('Training network...')
    _ = run_epochs(net, train_loader, criterion, optimizer, num_epochs, 'params/' + params_id + '.pth', save_freq=min(train_size, args.save_freq))

else:
    with torch.no_grad():
        _ = run_epochs(net, eval_loader, criterion, None, 1, None, train=False, save_freq=args.save_freq)
