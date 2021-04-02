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
from torchvision.utils import save_image

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
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--conv_thresh', type=float, default=0)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--subset', default=100, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--model', default='smp.Unet')
parser.add_argument('--encoder', default='resnet34')
parser.add_argument('--seg_lf', default='focal_loss')
parser.add_argument('--hipe_cells', type=int, default=4)
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--lr_decay', default=True, action='store_true')
parser.add_argument('--stats', default=False, action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "immune_seg_multi"

wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

dim = 256


################################################ HELPERS


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def equalise(img):
    return np.sort(img.ravel()).searchsorted(img)


def get_stats(loader):
    mean = 0
    std = 0
    var = 0
    minimum = 9999999
    maximum = 0
    cls_count = torch.zeros(5)
    cls_covg = torch.zeros(5)
    for i, data in enumerate(loader):
        print(i, '/', len(loader))
        input, target = data
        mean += input.mean()
        std += input.std()
        var += input.var()
        maximum = max(input.max(), maximum)
        minimum = min(input.min(), minimum)
        classes = torch.unique(target.reshape(-1)).to(torch.int64)
        cls_count[classes] += 1
        for c in range(5):
            cls_covg[c] += len(target[target == c].reshape(-1)) / len(target.reshape(-1))

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    var /= len(loader.dataset)
    cls_count = {'{}_count'.format(i): cls_count[i] / len(loader.dataset) for i in range(5)}
    cls_covg = {'{}_covg'.format(i): cls_covg[i] / len(loader.dataset) for i in range(5)}

    stats = {'minimum': minimum, 'maximum': maximum, 'mean': mean, 'std': std, 'var': var}
    stats.update(cls_count)
    stats.update(cls_covg)
    return stats


def separate_masks(masks):
    with torch.no_grad():
        sep_masks = []
        for cls_num in range(5):
            mask = masks.clone()
            mask[mask != cls_num] = -1
            mask[mask > -1] = 0
            mask += 1
            sep_masks.append(mask)
    return sep_masks


################################################ METRICS


def scores(o, t):
    score_arr = np.zeros((5, 5))

    for cls_num in range(5):
        output = o[:, cls_num]
        target = t.float()
        target[target != cls_num] = -1
        target[target > -1] = 0
        target += 1

        with torch.no_grad():
            tp = torch.sum(target * output)
            tn = torch.sum((1 - target) * (1 - output))
            fp = torch.sum((1 - target) * output)
            fn = torch.sum(target * (1 - output))

            p = tp / (tp + fp + 0.0001)
            r = tp / (tp + fn + 0.0001)

            f1 = 2 * p * r / (p + r + 0.0001)
            f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
            acc = (tp + tn) / (tp + tn + fp + fn)

            iou = tp / ((torch.sum(output + target) - tp) + 0.0001)
        score_arr[cls_num] = np.array([iou.item(), acc.item(), p.item(), r.item(), f1.item()])

    return score_arr


def focal_loss(outputs, targets, alpha=0.8, gamma=2):
    CE = F.cross_entropy(outputs, targets, reduction='mean')
    CE_EXP = torch.exp(-CE)
    focal_loss = alpha * (1 - CE_EXP) ** gamma * CE

    return focal_loss


################################################ SETTING UP DATASET


if not os.path.exists('sample_paths.csv'):
    print('Building data paths...')

    data = glob.glob("data/new_exported_tiles/*/*")

    samples = {}
    for d in data:
        dn = d.split(']')[0] + ']'
        print(dn)
        sample = samples[dn] if dn in samples else {}
        if '-labelled' in d:
            sample['Mask'] = d
            for s in range(5):
                sample['Class_{}'.format(s)] = 1 if s in np.array(Image.open(d)) else 0
        else:
            sample['Img'] = d
        sample['Slide'] = dn.split('/')[2]
        samples[dn] = sample

    samples = pd.DataFrame.from_dict(samples, orient='index')
    samples = samples.dropna()
    samples.to_csv('sample_paths.csv', index=False)
    print('\n', len(samples))
    print('DONE!')


class lc_data(Dataset):

    def __init__(self, samples, augment=False):

        self.samples = samples
        if augment:
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

        target = np.array(Image.open(s['Mask']))[:dim, :dim]
        target = np.pad(target, ((0, dim - target.shape[0]), (0, dim - target.shape[1])), 'minimum')

        img, target = torch.Tensor(img.astype(np.float32)).unsqueeze(0), torch.Tensor(target.astype(np.float32)).unsqueeze(0)

        if args.augment:
            hflip, vflip, angle = s['Aug']
            img = TF.rotate(img, angle)
            target = TF.rotate(target, angle)

            if hflip == 1:
                img = TF.hflip(img)
                target = TF.hflip(target)

            if vflip == 1:
                img = TF.vflip(img)
                target = TF.vflip(target)

        img = normalise(img)

        return img, target[0].long()


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, seg_criterion, num_epochs, path, save_freq=100, train=True):
    for epoch in range(num_epochs):
        total_loss = 0
        total_scores = np.zeros((5, 5))

        if args.lr_decay:
            lr = args.lr / (epoch + 1)

        else:
            lr = args.lr

        if train:
            print('Training...')
            mode = 'train'
            net.train()
            dataloader = train_loader
            optimizer = optim.Adam(net.parameters(), lr=lr)

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
            loss = focal_loss(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            output_masks = F.softmax(outputs, dim=1)

            total_loss += loss.item()

            print('({}) Epoch: {}/{} Batch: {}/{} Batch Loss {}'.format(mode, epoch, num_epochs, i, len(dataloader), loss.item()))

            results = {
                '{}/batch'.format(mode): i, '{}/epoch'.format(mode): epoch,
                '{}/loss'.format(mode):  loss.item(), '{}/mean_loss'.format(mode): total_loss / (i + 1), '{}/lr'.format(mode): lr
                }

            scores_list = ['IOU', 'acc', 'prec', 'rec', 'f1']

            batch_scores = scores(output_masks, targets)
            total_scores += batch_scores
            for cls_num in range(5):
                results.update(dict(zip(['{}/cls_{}/'.format(mode, cls_num) + s for s in scores_list], total_scores[cls_num] / (i + 1))))
                results.update(dict(zip(['{}/cls_{}/batch_'.format(mode, cls_num) + s for s in scores_list], batch_scores[cls_num])))

            if (i + 1) % save_freq == 0:

                results["{}/inputs".format(mode)] = [wandb.Image(im) for im in inputs.cpu()]
                results["{}/orig_targets".format(mode)] = [wandb.Image(im) for im in targets.float().cpu()]

                for cls_im in range(5):
                    results["{}/cls_{}/output".format(mode, cls_im)] = [wandb.Image(im) for im in output_masks[:, cls_im].cpu()]
                    results["{}/cls_{}/target_masks".format(mode, cls_im)] = [wandb.Image(im) for im in separate_masks(targets.float().cpu())[cls_im]]

                if args.hipe_cells > 0:
                    for cls_im in range(1, 5):
                        print('HiPe for class {}'.format(cls_im))
                        hipe, depth_hipe, hipe_masks = hierarchical_perturbation(net, inputs[0].unsqueeze(0).detach(), target=cls_im, batch_size=1, num_cells=args.hipe_cells, perturbation_type='fade')
                        results["{}/cls_{}/hipe".format(mode, cls_im)] = wandb.Image(hipe.cpu())
                        results["{}/cls_{}/hipe_masks".format(mode, cls_im)] = hipe_masks
                        for ds in range(depth_hipe.shape[0]):
                            results["{}/cls_{}/hipe_depth_{}".format(mode, cls_im, ds)] = wandb.Image(depth_hipe.cpu()[ds])

            wandb.log(results)

        if train:
            print('Saving model...')
            torch.save(net.state_dict(), path)
            with torch.no_grad():
                run_epochs(net, None, eval_loader, seg_criterion, 1, None, train=False, save_freq=args.save_freq)

    return


samples = pd.read_csv('sample_paths.csv').sample(frac=args.subset / 100, random_state=0).reset_index(drop=True)
wandb.log({'samples': len(samples)})

train_size = int(0.8 * len(samples))
val_test_size = len(samples) - train_size
val_size = int(0.5 * val_test_size)

print('{} training samples, {} validation samples...'.format(train_size, val_size))

all_dataset = lc_data(samples, augment=False)

if args.stats:
    data_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)

    print(get_stats(data_loader))
    exit()

train_dataset = lc_data(samples[:train_size], augment=args.augment)
val_dataset = lc_data(samples[train_size:train_size+val_size], augment=False)
test_dataset = lc_data(samples[train_size+val_size:], augment=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)
eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)


net = eval(args.model)(encoder_name=args.encoder, in_channels=1, classes=5)

seg_criterion = eval(args.seg_lf)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)

num_epochs = args.epochs

if args.load is not '':
    net.load_state_dict(torch.load('params/' + args.load + ".pth", map_location=torch.device(device)))

net.to(device)
wandb.watch(net)

if args.train:
    print('Training network...')
    run_epochs(net, train_loader, eval_loader, seg_criterion, num_epochs, 'params/' + params_id + '.pth', save_freq=args.save_freq)

elif args.test:
    with torch.no_grad():
        run_epochs(net, None, test_loader, seg_criterion, 1, None, train=False, save_freq=args.save_freq)

else:
    with torch.no_grad():
        run_epochs(net, None, eval_loader, seg_criterion, 1, None, train=False, save_freq=args.save_freq)
