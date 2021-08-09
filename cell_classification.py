import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import atexit
from torch.utils.data import Dataset
import argparse
import glob
import wandb
import random
import segmentation_models_pytorch as smp
from hipe import hierarchical_perturbation
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as TF

torch.set_printoptions(precision=4, linewidth=300)
np.set_printoptions(precision=4, linewidth=300)
pd.set_option('display.max_columns', None)


def exit_handler():
    print('Finishing run...')
    run.finish()
    print('Done!')


atexit.register(exit_handler)


################################################ SET UP SEED AND ARGS AND EXIT HANDLER

def set_seed():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)


set_seed()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--subset', default=1, type=float)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--model', default='smp.Unet')
parser.add_argument('--encoder', default='resnet18')
parser.add_argument('--lf', default='ce_loss')
parser.add_argument('--hipe', default=False, action='store_true')
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--im_only', default=False, action='store_true')
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--num_wb_img', type=int, default=12)
parser.add_argument('--start_epoch', type=int, default=0)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "cell_classification"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name

if args.load:
    params_id = args.load
    wandb.run.name = params_id

print(params_id, args)

dim = 64

CLASS_LIST = ['OTHER', 'CD8_CD3LO', 'CD8_CD3HI', 'CD3', 'CD20']
if args.im_only:
    CLASS_LIST = ['CD8_CD3LO', 'CD8_CD3HI', 'CD3', 'CD20']

print(CLASS_LIST)


################################################ HELPERS


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def equalise(img):
    return np.sort(img.ravel()).searchsorted(img)


def save_im(img, name='image'):
    img = img.cpu().numpy()[0] * 255
    img = Image.fromarray(img).convert('L')
    img.save(name + '.png', 'PNG', quality=100, subsampling=0)


################################################ METRICS & MODEL


def ce_focal_loss(outputs, targets, alpha=0.8, gamma=2):
    ce = F.cross_entropy(outputs, targets)
    ce_exp = torch.exp(-ce)
    loss = alpha * (1 - ce_exp) ** gamma * ce
    return loss


def ce_loss(outputs, targets):
    return F.cross_entropy(outputs, targets)


################################################ SETTING UP DATASET


class cell_data(Dataset):

    def __init__(self, samples):
        self.samples = samples
        self.dim = dim
        if args.augment:
            self.augs = list(range(0, 360, 90))
            num_augs = len(self.augs)
            self.augs *= len(self.samples)
            self.samples *= num_augs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index]
        mask = Image.open(img.split('.tif')[0] + '-mask.png')
        img = Image.open(img)

        if args.augment:
            rot = self.augs[index]
            img = TF.rotate(img, rot)
            mask = TF.rotate(mask, rot)

        img = normalise(np.array(img)) * gaussian_filter(np.array(mask), sigma=1)

        w, h = img.shape
        if h > self.dim:
            ch1 = (h - self.dim) // 2
            ch2 = self.dim - ch1
            img = img[:, ch1:ch2]

        if w > self.dim:
            cw1 = (w - self.dim) // 2
            cw2 = self.dim - cw1
            img = img[cw1:cw2, :]

        w, h = img.shape

        pw1, ph1 = (self.dim - w) // 2, (self.dim - h) // 2
        pw2, ph2 = self.dim - w - pw1, self.dim - h - ph1

        img = np.pad(img, ((pw1, pw2), (ph1, ph2)), 'constant')

        img = torch.Tensor(img).unsqueeze(0)
        cls = self.samples[index].split('-')[1]
        target = CLASS_LIST.index(cls)
        return img, target


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, num_epochs, path, save_freq=100, train=True):
    lr = args.lr
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0

        if args.lr_decay and epoch > 0:
            print('Decaying learning rate...')
            lr = args.lr / (epoch + 1)
            optimizer = optim.AdamW(net.parameters(), lr=lr)

        if train:
            print('Training...')
            mode = 'train'
            net.train()
            dataloader = train_loader

        else:
            print('Evaluating...')
            mode = 'val'
            net.eval()
            dataloader = eval_loader
            lr = 0

        save_freq = max(min(save_freq, len(dataloader) - 1), 1)

        for i, data in enumerate(dataloader):

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)

            loss = eval(args.lf)(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            correct = torch.sum(torch.argmax(outputs, dim=1) == targets)

            print('{} ({}) Epoch: {}/{} Correct: {}/{} Batch: {}/{} Batch Loss {} LR {}'.format(params_id, mode, epoch + 1, num_epochs, correct, outputs.shape[0], i, len(dataloader), loss.item(), lr))

            class_correct = {}
            for c in range(len(CLASS_LIST)):
                c_mask = (targets == c)
                class_correct['{}/'.format(mode) + CLASS_LIST[c] + '_correct'] = torch.sum(torch.argmax(outputs, dim=1)[c_mask] == targets[c_mask]) / torch.sum(c_mask)

            results = {
                '{}/loss'.format(mode):    loss.item(), '{}/mean_loss'.format(mode): total_loss / (i + 1), '{}/lr'.format(mode): lr, '{}/batch'.format(mode): i + 1, '{}/epoch'.format(mode): epoch + 1,
                '{}/correct'.format(mode): correct / outputs.shape[0]
                }
            results.update(class_correct)

            wandb.log(results)

            if (i + 1) % save_freq == 0:
                results["{}/imgs".format(mode)] = [wandb.Image(inputs[b].detach().cpu(), caption='pred: {}\ntrue: {}'.format(CLASS_LIST[torch.argmax(outputs, dim=1)[b]], CLASS_LIST[targets[b]]))
                                                   for b in
                                                   range(
                                                           min(outputs.shape[0], args.num_wb_img))]

                if args.hipe:

                    for cls_im in range(1, len(CLASS_LIST)):
                        hbs = []
                        for hb in range(args.batch_size):

                            if cls_im in targets[hb]:
                                print('HiPe for {}'.format(CLASS_LIST[cls_im]))
                                hipe, hipe_depth, hipe_masks = hierarchical_perturbation(net, inputs[hb].unsqueeze(0).detach(), target=cls_im, batch_size=32, perturbation_type='fade')
                                hbs.append(wandb.Image(hipe_depth.cpu() * (targets[hb] > 0).cpu()))
                            else:
                                hbs.append(wandb.Image(torch.zeros_like(inputs[hb].unsqueeze(0))))

                        results["{}/{}/hipe".format(mode, CLASS_LIST[cls_im])] = hbs

                wandb.log(results)

                if train:
                    torch.save(net.state_dict(), path)
                    print('Saving model...')
                    with torch.no_grad():
                        run_epochs(net, None, eval_loader, 1, None, train=False, save_freq=args.save_freq)
                    net.train()

    return


samples = glob.glob("exported_cells/*/*.tif")
if args.im_only:
    samples = [s for s in samples if 'OTHER' not in s]
random.shuffle(samples)
if args.subset < 1:
    samples = samples[:int(len(samples) * args.subset)]

wandb.log({'samples': len(samples)})

train_size = int(0.8 * len(samples))
val_test_size = len(samples) - train_size
val_size = int(0.5 * val_test_size)

train_dataset = cell_data(samples[:train_size])
val_dataset = cell_data(samples[train_size:train_size + val_size])
test_dataset = cell_data(samples[train_size + val_size:])

print('{} training samples, {} validation samples, {} test samples...'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)

aux = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=args.dropout,  # dropout ratio, default is None
        activation=None,  # activation function, default is None
        classes=len(CLASS_LIST)  # define number of output labels
        )

net = eval(args.model)(encoder_name=args.encoder, in_channels=1, classes=len(CLASS_LIST), aux_params=aux)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

num_epochs = args.epochs

if args.load is not '':
    state_dict = torch.load('params/' + args.load + ".pth", map_location=torch.device(device))
    net.load_state_dict(state_dict)

net.to(device)
wandb.watch(net)

if args.train:
    print('Training network...')
    run_epochs(net, train_loader, eval_loader, num_epochs, 'params/' + params_id + '.pth', save_freq=args.save_freq)

elif args.test:
    with torch.no_grad():
        run_epochs(net, None, test_loader, 1, None, train=False, save_freq=args.save_freq)

else:
    with torch.no_grad():
        run_epochs(net, None, eval_loader, 1, None, train=False, save_freq=args.save_freq)

run.finish()
exit()
