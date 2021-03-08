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
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--conv_thresh', type=float, default=0)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--subset', default=100, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--model', default='smp.PSPNet')
parser.add_argument('--encoder', default='resnet34')
parser.add_argument('--lf', default='f1_loss')
parser.add_argument('--cls', type=int, default=-1)
parser.add_argument('--slides', default=None)
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--experimental', default=False, action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.environ["WANDB_SILENT"] = "true"

proj = "immune-seg-exp" if args.experimental else "immune-seg"

wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

num_classes = 5
dim = 256


################################################ MODELS

class resnet50(nn.Module):

    def __init__(self, encoder, in_channels, classes):
        super().__init__()
        self.fcn = models.resnet50(num_classes=num_classes)
        self.ch_conv = nn.Conv2d(1, 3, 1)

    def forward(self, x):
        x = self.ch_conv(x)
        x = self.fcn(x)
        return x


class resnet101(nn.Module):

    def __init__(self, encoder, in_channels, classes):
        super().__init__()
        self.fcn = models.resnet101(num_classes=num_classes)
        self.ch_conv = nn.Conv2d(1, 3, 1)

    def forward(self, x):
        x = self.ch_conv(x)
        x = self.fcn(x)
        return x


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
    return (img - img.mean()) / (img.std() + 0.0001)


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def whiten(img):
    img = img - img.mean()
    cov = np.dot(img.T, img)
    d, vec = np.linalg.eigh(cov)
    diag = np.diag(1 / np.sqrt(d + 0.0001))
    w = np.dot(np.dot(vec, diag), vec.T)
    return np.dot(img, w)


def equalise(img):
    return np.sort(img.ravel()).searchsorted(img)


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
            sample['Classes'] = np.unique(np.array(Image.open(d)))
        else:
            sample['Img'] = d
        sample['Slide'] = dn.split('/')[2]
        samples[dn] = sample

    samples = pd.DataFrame.from_dict(samples, orient='index')
    samples = samples.dropna()
    samples.to_csv('sample_paths.csv', index=False)
    print('\n', len(samples))
    print('DONE!')


class lc_seg_tiles_bc(Dataset):

    def __init__(self):

        self.samples = pd.read_csv('sample_paths.csv').sample(frac=args.subset / 100, random_state=0).reset_index(drop=True)
        if args.slides is not None:
            self.samples = self.samples[self.samples['Slide'].isin(args.slides.split('_'))]

        if args.cls > 0:
            self.samples = self.samples[self.samples['Classes'].str.contains(str(args.cls))]

        if args.augment:
            flips = [[0, 0], [0, 1], [1, 0], [1, 1]]
            rots = [0, 90, 180, 270]
            augs = []
            for r in rots:
                for f1, f2 in flips:
                    augs.append([f1, f2, r])

            num_augs = len(augs)
            augs *= len(self.samples)
            self.samples = pd.concat([self.samples] * num_augs, ignore_index=True)
            self.samples['Aug'] = augs
            self.samples = self.samples.sample(frac=1, random_state=0).reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples.iloc[index]

        img = np.array(Image.open(s['Img']))[:dim, :dim]
        img = np.pad(img, ((0, dim - img.shape[0]), (0, dim - img.shape[1])), 'minimum')

        mask = np.array(Image.open(s['Mask']))[:dim, :dim]
        mask = np.pad(mask, ((0, dim - mask.shape[0]), (0, dim - mask.shape[1])), 'minimum')

        if args.cls > 0:
            mask[mask != args.cls] = 0
        mask[mask > 0] = 1

        img = normalise(img)
        img, mask = torch.Tensor(img.astype(np.float32)).unsqueeze(0), torch.Tensor(mask.astype(np.float32)).unsqueeze(0)

        if args.augment:
            hflip, vflip, angle = s['Aug']
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

            if hflip == 1:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if vflip == 1:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        return img, mask


class experimental(lc_seg_tiles_bc):

    def __init__(self):
        super().__init__()
        self.samples['Classes'] = self.samples['Classes'].apply(lambda x: eval(x.replace(' ', ',')))
        self.samples = self.samples.explode('Classes', ignore_index=True)
        self.samples = self.samples[self.samples['Classes'] > 0]

    def __getitem__(self, index):
        s = self.samples.iloc[index]

        img = np.array(Image.open(s['Img']))[:dim, :dim]
        img = np.pad(img, ((0, dim - img.shape[0]), (0, dim - img.shape[1])), 'minimum')

        mask = np.array(Image.open(s['Mask']))[:dim, :dim]
        mask = np.pad(mask, ((0, dim - mask.shape[0]), (0, dim - mask.shape[1])), 'minimum')

        cls = int(s['Classes'])
        mask = np.where(mask == cls, 1, 0)
        img = normalise(img) * mask
        target = np.zeros(num_classes)

        target[cls] = 1
        img = torch.Tensor(np.array(img).astype(np.float32)).unsqueeze(0)
        target = torch.Tensor(target)
        return img, target


################################################ TRAINING


def run_epochs(net, dataloader, criterion, optimizer, num_epochs, path, save_freq=100, train=True):
    mode = 'train' if train else 'val'
    avg_epoch_loss = 0
    if not train:
        net.eval()

    for epoch in range(num_epochs):
        sum_epoch_loss = 0.0
        sum_epoch_scores = torch.zeros(3)

        for i, data in enumerate(dataloader):

            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            cls = torch.argmax(targets, dim=1)

            if args.experimental:
                outputs = torch.softmax(net(inputs), dim=1)
                loss = criterion(outputs, cls)
            else:
                outputs = torch.sigmoid(net(inputs))
                loss = criterion(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('({}) Epoch: {}/{} Batch: {}/{} Batch Loss {}'.format(mode, epoch, num_epochs, i, len(dataloader), loss.item()))

            batch_scores = scores(outputs, targets)

            print(dict(zip(['prec', 'rec', 'f1'], batch_scores)))

            sum_epoch_loss += loss.item()
            sum_epoch_scores += batch_scores

            epoch_loss = sum_epoch_loss / (i + 1)
            epoch_scores = sum_epoch_scores / (i + 1)

            results = {'{}/batch'.format(mode): i, '{}/epoch'.format(mode): epoch, '{}/batch_loss'.format(mode): loss.item(), '{}/epoch_loss'.format(mode): epoch_loss}

            results.update(dict(zip(['{}/epoch_'.format(mode) + s for s in ['prec', 'rec', 'f1']], epoch_scores)))
            results.update(dict(zip(['{}/batch_'.format(mode) + s for s in ['prec', 'rec', 'f1']], batch_scores)))

            if i % save_freq == 0:
                results["{}/inputs".format(mode)] = [wandb.Image(inputs[0])]
                if args.experimental:
                    correct = torch.sum(torch.round(outputs) == targets)
                    print("{}/{} correct".format(correct, args.batch_size))
                    results['{}/correct'.format(mode)] = correct
                else:
                    results["{}/outputs".format(mode)] = [wandb.Image(i) for i in outputs]
                    results["{}/masks".format(mode)] = [wandb.Image(i) for i in targets]

            wandb.log(results)

        print('Completed epoch {}. Avg epoch loss: {}'.format(epoch, sum_epoch_loss / (i + 1)))

        if train:
            print('Saving model...')
            torch.save(net.state_dict(), path)
            print('Evaluating network...')
            with torch.no_grad():
                _ = run_epochs(net, eval_loader, criterion, None, 1, None, train=False, save_freq=args.save_freq)

        avg_epoch_loss += epoch_loss

    final_results = {'{}_samples'.format(mode): len(dataloader) * args.batch_size, '{}/avg_epoch_loss'.format(mode): avg_epoch_loss / num_epochs}
    wandb.log(final_results)

    return net


dataset = lc_seg_tiles_bc()
net = eval(args.model)(args.encoder, in_channels=1, classes=1)
criterion = eval(args.lf)

if args.experimental:
    dataset = experimental()
    net = eval(args.model)(args.encoder, in_channels=1, classes=num_classes)
    criterion = nn.NLLLoss()

"""
# multiclass segmentation
dataset = lc_seg_tiles_mc()
net = eval(args.model)(n_channels=1, n_classes=num_classes)
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
eval_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)

print('{} training samples, {} validation samples...'.format(train_size, val_size))

if args.load is not '':
    net.load_state_dict(torch.load('params/' + args.load + ".pth", map_location=torch.device(device)))

if args.train:
    print('Training network...')
    _ = run_epochs(net, train_loader, criterion, optimizer, num_epochs, 'params/' + params_id + '.pth', save_freq=min(train_size, args.save_freq))

else:
    with torch.no_grad():
        _ = run_epochs(net, eval_loader, criterion, None, 1, None, train=False, save_freq=args.save_freq)
