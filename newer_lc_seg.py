import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import atexit
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
from torchvision.utils import save_image
from skimage.draw import disk
from PIL import Image

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
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--subset', default=100, type=float)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--model', default='smp.Unet')
parser.add_argument('--encoder', default='resnet34')
parser.add_argument('--seg_lf', default='ce_loss')
parser.add_argument('--hipe', default=False, action='store_true')
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--stats', default=False, action='store_true')
parser.add_argument('--num_wb_img', type=int, default=12)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--sanity', default=False, action='store_true')
parser.add_argument('--find_bad', default=False, action='store_true')
parser.add_argument('--im_only', default=False, action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "immune_seg_multi"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name

print(params_id, args)

dim = 256
NUM_CLASSES = 4

CLASS_LIST = ['OTHER', 'CD8', 'CD3', 'CD20', 'CD8_PD1', 'CD3_PD1', 'CD20_PD1']
SLIDES = ['L102', 'L111', 'L114', 'L135', 'L149', 'L47', 'L70', 'L722', 'L730', 'L749', 'L74', 'L93']

print(CLASS_LIST)
print(SLIDES)


################################################ HELPERS


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def equalise(img):
    return np.sort(img.ravel()).searchsorted(img)


def save_im(img, name='image'):
    img = img.cpu().numpy()[0] * 255
    print(img.min(), img.max())

    print(img.shape)
    img = Image.fromarray(img).convert('L')
    print(np.array(img).min(), np.array(img).max())
    img.save(name + '.png', 'PNG', quality=100, subsampling=0)


def get_stats(loader):
    mean = 0
    std = 0
    var = 0
    minimum = 9999999
    maximum = 0
    class_presence = np.zeros(len(CLASS_LIST))
    mean_img = torch.zeros((1, dim, dim))

    for i, data in enumerate(loader):
        print(i, '/', len(loader))
        input, target  = data
        mean_img += input[0]
        mean += input.mean()
        std += input.std()
        var += input.var()
        maximum = max(input.max(), maximum)
        minimum = min(input.min(), minimum)
        class_presence += (np.array(cc) > 0).astype(float)

    mean_img = normalise(mean_img)

    img_name = 'mean_img'
    save_im(mean_img, img_name)
    wandb.log({img_name: wandb.Image(mean_img[0])})
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    var /= len(loader.dataset)


    class_presence = dict(zip(['{}_presence'.format(c) for c in CLASS_LIST], class_presence))

    stats = {'minimum': minimum, 'maximum': maximum, 'mean': mean, 'std': std, 'var': var}
    stats.update(class_presence)
    stats['Slides'] = SLIDES
    wandb.log(stats)
    return stats


def separate_masks(masks):
    with torch.no_grad():
        sep_masks = []
        for cls_num in range(len(CLASS_LIST)):
            m = (masks == cls_num).float()
            sep_masks.append(m)

    return sep_masks


################################################ METRICS


def scores(o, t):
    score_arr = np.zeros((NUM_CLASSES, 5))

    for cls_num in range(NUM_CLASSES):
        output = o[:, cls_num]
        target = (t == cls_num).float()

        with torch.no_grad():
            tp = torch.sum(target * output)
            tn = torch.sum((1 - target) * (1 - output))
            fp = torch.sum((1 - target) * output)
            fn = torch.sum(target * (1 - output))

            p = tp / (tp + fp + 0.0001)
            r = tp / (tp + fn + 0.0001)
            f1 = 2 * p * r / (p + r + 0.0001)
            acc = (tp + tn) / (tp + tn + fp + fn)
            iou = tp / ((torch.sum(output + target) - tp) + 0.0001)

        score_arr[cls_num] = np.array([p.item(), r.item(), f1.item(), acc.item(), iou.item()])
    return score_arr


def f1_loss(o, t):
    f1 = torch.zeros(1, device=device)

    for cls_num in range(NUM_CLASSES):
        output = o[:, cls_num]
        target = (t == cls_num).float()

        tp = torch.sum(target * output)
        fp = torch.sum((1 - target) * output)
        fn = torch.sum(target * (1 - output))

        p = tp / (tp + fp + 0.0001)
        r = tp / (tp + fn + 0.0001)

        f1 += 2 * p * r / (p + r + 0.0001)

    return 1 - f1 / NUM_CLASSES


def ce_focal_loss(outputs, targets, alpha=0.8, gamma=2):
    ce = F.cross_entropy(outputs, targets)
    ce_exp = torch.exp(-ce)
    loss = alpha * (1 - ce_exp) ** gamma * ce
    return loss


def ce_loss(outputs, targets):
    return F.cross_entropy(outputs, targets)


def mse_loss(outputs, targets):
    return F.mse_loss(outputs, targets)


################################################ SETTING UP DATASET


class lc_data(Dataset):

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.dim = dim
        self.augment=augment
        if augment:
            self.augs = list(range(0, 360, 90))
            num_augs = len(self.augs)
            self.augs *= len(self.samples)
            self.samples *= num_augs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index]
        target = Image.open(img.split('.tif')[0] + '-labelled.png')
        img = Image.open(img)

        img = torch.Tensor(img)
        target = torch.Tensor(target)
        target[target > NUM_CLASSES] = target - NUM_CLASSES + 1

        return img, target.long()


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, num_epochs, path, save_freq=100, train=True):
    lr = args.lr
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    worst = 1

    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0
        scores_list = ['prec', 'rec', 'f1', 'acc', 'iou']

        total_scores = np.zeros((NUM_CLASSES, len(scores_list)))

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
            outputs = net(inputs)

            loss = eval(args.seg_lf)(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            outputs = torch.softmax(outputs, dim=1)

            print('{} ({}) Epoch: {}/{} Batch: {}/{} Batch Loss {} LR {}'.format(params_id, mode, epoch + 1, num_epochs, i, len(dataloader), loss.item(), lr))

            results = {
                '{}/loss'.format(mode): loss.item(), '{}/mean_loss'.format(mode): total_loss / (i + 1), '{}/lr'.format(mode): lr, '{}/batch'.format(mode): i + 1, '{}/epoch'.format(mode): epoch + 1
                }

            batch_scores = scores(outputs, targets)

            total_scores += batch_scores

            print(scores_list)
            print(batch_scores)

            results.update({'{}/avg_f1'.format(mode): (np.sum(total_scores[1:NUM_CLASSES, 2]) / NUM_CLASSES) / (i + 1)})

            for cls_num in range(NUM_CLASSES):
                results.update(dict(zip(['{}/{}/'.format(mode, CLASS_LIST[cls_num]) + s for s in scores_list], total_scores[cls_num] / (i + 1))))

            wandb.log(results)
            if (np.mean(batch_scores[:, 2]) <= worst) and args.find_bad:
                worst = np.mean(batch_scores[:, 2])
                print('Worst: ', worst)
                save_bad = True
            else:
                save_bad = False

            if ((i + 1) % save_freq == 0) or save_bad:
                results["{}/inputs".format(mode)] = [wandb.Image(im) for im in inputs[:args.num_wb_img].cpu()]

                results["{}/targets".format(mode)] = [wandb.Image(im) for im in targets[:args.num_wb_img].float().cpu()]

                for cls_im in range(NUM_CLASSES):
                    results["{}/{}/output".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in outputs[:args.num_wb_img, cls_im].float().cpu()]

                    results["{}/{}/targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(separate_masks(im)[cls_im]) for im in targets[:args.num_wb_img].float().cpu()]

                if args.hipe:

                    for cls_im in range(1, NUM_CLASSES):
                        hbs = []
                        for hb in range(args.batch_size):

                            if cls_im in targets[hb] and (torch.sum(targets[hb] > 0) < (dim * dim) / 10):
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


samples = glob.glob("newer_exported_tiles/*/*.tif")

random.shuffle(samples)
if args.subset < 1:
    samples = samples[:int(len(samples) * args.subset)]

wandb.log({'samples': len(samples)})

train_size = int(0.8 * len(samples))
val_test_size = len(samples) - train_size
val_size = int(0.5 * val_test_size)

train_dataset = lc_data(samples[:train_size], augment=args.augment)
val_dataset = lc_data(samples[train_size:train_size + val_size], augment=False)
test_dataset = lc_data(samples[train_size + val_size:], augment=False)

if args.stats:
    all_dataset = lc_data(samples, augment=False)
    data_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)
    stats = get_stats(data_loader)
    print(stats)
    exit()

print('{} training samples, {} validation samples, {} test samples...'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)

net = eval(args.model)(encoder_name=args.encoder, in_channels=1, classes=NUM_CLASSES)

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
