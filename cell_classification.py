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
import torchvision.models as models
from hipe import hierarchical_perturbation
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as TF
from cnn_finetune import make_model

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
parser.add_argument('--binary_class', default='')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--lf', default='ce_loss')
parser.add_argument('--hipe', default=False, action='store_true')
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--im_only', default=False, action='store_true')
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--standardise', default=False, action='store_true')
parser.add_argument('--normalise', default=False, action='store_true')
parser.add_argument('--tabular', default=False, action='store_true')
parser.add_argument('--cd3', default=False, action='store_true')
parser.add_argument('--num_wb_img', type=int, default=12)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--early_stopping', type=int, default=10)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "cell_classification"
if args.tabular:
    proj = "tabular_cell_classification"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name

if args.load:
    params_id = args.load
    wandb.run.name = params_id

print(params_id, args)

dim = args.dim

ALL_CLASS_LIST = ['CD8', 'CD3', 'CD20']

if args.binary_class != '':
    CLASS_LIST = ['OTHER', args.binary_class]
elif args.cd3:
    CLASS_LIST = ['CD3', 'CD20']
else:
    CLASS_LIST = ALL_CLASS_LIST

ALL_SLIDES = ['L730', 'L749', 'L135', 'L149', 'L70', 'L722', 'L102', 'L111', 'L74', 'L93', 'L114', 'L47']

NUM_CLASSES = len(CLASS_LIST)

print(CLASS_LIST)
wandb.log({'Class list': CLASS_LIST})


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


def bce_loss(outputs, targets):
    return F.binary_cross_entropy_with_logits(outputs.reshape(-1).float(), targets.float())


class cell_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = make_model(args.model, num_classes=NUM_CLASSES, pretrained=True, dropout_p=args.dropout)
        self.channel_adjust = torch.nn.Conv2d(1, 3, 1)

    def forward(self, x):
        return self.net(self.channel_adjust(x))


class tab_cell_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(7, 1000)
        self.l2 = torch.nn.Linear(1000,3)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x


################################################ SETTING UP DATASET


class tab_cell_data(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.classes = self.samples['Class']
        self.img_names = self.samples['Image']
        self.samples = self.samples[[
       'Detection probability', 'Nucleus: Area µm^2', 'Nucleus: Length µm',
       'Nucleus: Circularity', 'Nucleus: Solidity', 'Nucleus: Max diameter µm',
       'Nucleus: Min diameter µm']]

        self.samples.apply(normalise, axis=1)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples.iloc[index]
        target = CLASS_LIST.index(self.classes.iloc[index])
        id = self.img_names.iloc[index].split('_')[0]
        return torch.Tensor(x.data), target, id


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
        id = img.split('/')[1].split('_')[0]

        mask = Image.open(img.split('.tif')[0] + '-mask.png')
        img = Image.open(img)

        if args.augment:
            rot = self.augs[index]
            img = TF.rotate(img, rot)
            mask = TF.rotate(mask, rot)

        img = np.array(img)

        if args.normalise:
            img = normalise(img)

        img *= gaussian_filter(np.array(mask), sigma=1)
        # img *= np.array(mask)

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

        img = TF.to_tensor(np.pad(img.astype(np.float32), ((pw1, pw2), (ph1, ph2)), 'constant'))

        if args.standardise:
            img = TF.normalize(img, torch.mean(img), torch.std(img))

        cls = self.samples[index].split('-')[1]
        if args.cd3 and cls == 'CD8':
            cls = 'CD3'

        target = ALL_CLASS_LIST.index(cls)

        if args.binary_class != '':
            target = 1 if cls == args.binary_class else 0

        return img, target, id


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, num_epochs, path, save_freq=100, train=True, lv=0, esc=args.early_stopping):
    lr = args.lr
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    last_val_correct = lv
    es_countdown = esc

    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0
        total_correct = 0

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

            inputs, targets, slide_ids = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = eval(args.lf)(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            slide_losses = dict(zip(['{}/{}'.format(mode, s) for s in ALL_SLIDES], [0] * len(ALL_SLIDES)))
            for s in range(inputs.shape[0]):
                s_id = slide_ids[s]
                s_id_count = slide_ids.count(s_id)
                slide_loss = eval(args.lf)(outputs[s].detach().unsqueeze(0), targets[s].detach().unsqueeze(0)).cpu()
                slide_losses['{}/{}'.format(mode, s_id)] += (slide_loss / s_id_count)

            correct = torch.sum(torch.argmax(outputs, dim=1) == targets)
            total_correct += correct / outputs.shape[0] / len(dataloader)

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
            results.update(slide_losses)

            wandb.log(results)

            if (i + 1) % save_freq == 0 and not args.tabular:

                results["{}/imgs".format(mode)] = [wandb.Image(inputs[b].detach().cpu(), caption='pred: {}\ntrue: {}'.format(CLASS_LIST[torch.argmax(outputs, dim=1)[b]], CLASS_LIST[targets[b]])) for b
                                                   in range(min(outputs.shape[0], args.num_wb_img))]

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
            with torch.no_grad():
                last_val_correct, es_countdown = run_epochs(net, None, eval_loader, 1, path, train=False, save_freq=args.save_freq, lv=last_val_correct, esc=es_countdown)
            net.train()

        else:
            print('Last num correct: {} Current num correct: {}'.format(last_val_correct, total_correct))
            'Early stopping countdown: {}'.format(es_countdown)
            wandb.log({'es_countdown': es_countdown, 'val/total_num_correct': total_correct})
            if total_correct < last_val_correct:
                es_countdown -= 1
                if es_countdown == 0:
                    print('Early stopping - val correct rate did not improve after {} epochs'.format(args.early_stopping))
                    exit()
            else:
                es_countdown = args.early_stopping
                last_val_correct = total_correct
                print('Saving model...')
                torch.save(net.state_dict(), path)

    return last_val_correct, es_countdown


if args.tabular:
    samples = pd.read_csv("cell_detections.csv").sample(frac=1)

else:
    samples = glob.glob("new_exported_cells/*/*.tif")
    random.shuffle(samples)
if args.subset < 1:
    samples = samples[:int(len(samples) * args.subset)]

train_size = int(0.8 * len(samples))
val_test_size = len(samples) - train_size
val_size = int(0.5 * val_test_size)

if args.tabular:
    train_dataset = tab_cell_data(samples[:train_size])
    val_dataset = tab_cell_data(samples[train_size:train_size + val_size])
    test_dataset = tab_cell_data(samples[train_size + val_size:])
else:
    train_dataset = cell_data(samples[:train_size])
    val_dataset = cell_data(samples[train_size:train_size + val_size])
    test_dataset = cell_data(samples[train_size + val_size:])

wandb.log({'train samples': len(train_dataset), 'val samples': len(val_dataset), 'test_samples': len(test_dataset)})

print('{} training samples, {} validation samples, {} test samples...'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)

if args.tabular:
    net = tab_cell_net()
else:
    net = cell_net()

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
    _, _ = run_epochs(net, train_loader, eval_loader, num_epochs, 'params/' + params_id + '.pth', save_freq=args.save_freq)

elif args.test:
    with torch.no_grad():
        _, _ = run_epochs(net, None, test_loader, 1, None, train=False, save_freq=args.save_freq)

else:
    with torch.no_grad():
        _, _ = run_epochs(net, None, eval_loader, 1, None, train=False, save_freq=args.save_freq)

run.finish()
exit()
