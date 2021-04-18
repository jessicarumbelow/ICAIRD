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
parser.add_argument('--batch_size', type=int, default=48)
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
parser.add_argument('--seg_lf', default='bce_loss')
parser.add_argument('--large', default=False, action='store_true')
parser.add_argument('--hipe_cells', type=int, default=4)
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--stats', default=False, action='store_true')
parser.add_argument('--centroids', default=False, action='store_true')
parser.add_argument('--weighted_loss', default=False, action='store_true')
parser.add_argument('--dynamic_lr', default=False, action='store_true')
parser.add_argument('--num_wb_img', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--cr', type=int, default=0)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "immune_seg_multi"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

dim = 256

CLASS_LIST = ['0', 'CD8', 'CD3', 'CD20', 'CD8: CD3', 'CD20: CD8', 'CD20: CD8: CD3', 'CD20: CD3']
print(CLASS_LIST)


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
    cls_count = dict(zip(CLASS_LIST, [0] * len(CLASS_LIST)))
    cls_covg = torch.zeros(4)
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
            cls_covg[c] += torch.sum(target[c]) / len(target[c].reshape(-1))

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    var /= len(loader.dataset)
    cls_count = {'{}_count'.format(i): cls_count[i] / len(loader.dataset) for i in range(4)}
    cls_covg = {'{}_covg'.format(i): cls_covg[i] / len(loader.dataset) for i in range(4)}

    stats = {'minimum': minimum, 'maximum': maximum, 'mean': mean, 'std': std, 'var': var}
    stats.update(cls_count)
    stats.update(cls_covg)
    return stats


################################################ METRICS


def scores(o, t):
    score_arr = np.zeros((4, 3))

    for cls_num in range(4):
        output = o[:, cls_num]
        target = t[:, cls_num]

        with torch.no_grad():
            tp = torch.sum(target * output)
            tn = torch.sum((1 - target) * (1 - output))
            fp = torch.sum((1 - target) * output)
            fn = torch.sum(target * (1 - output))

            p = tp / (tp + fp + 0.0001)
            r = tp / (tp + fn + 0.0001)
            f1 = 2 * p * r / (p + r + 0.0001)
            # acc = (tp + tn) / (tp + tn + fp + fn)
            # iou = tp / ((torch.sum(output + target) - tp) + 0.0001)

        score_arr[cls_num] = np.array([p.item(), r.item(), f1.item()])
    return score_arr


def f1_loss(o, t):
    f1 = torch.zeros(1, device=device)

    for cls_num in range(4):
        output = o[:, cls_num]
        target = t[:, cls_num]
        target[target != cls_num] = -1
        target[target > -1] = 0
        target += 1

        tp = torch.sum(target * output)
        fp = torch.sum((1 - target) * output)
        fn = torch.sum(target * (1 - output))

        p = tp / (tp + fp + 0.0001)
        r = tp / (tp + fn + 0.0001)

        f1 += 2 * p * r / (p + r + 0.0001)

    return 1 - f1 / 4


def get_class_weights(targets):
    if args.weighted_loss:
        weights = torch.zeros(4, device=device)
        for cls_num in range(4):
            weights[cls_num] += 1 - torch.sum(targets[:, cls_num]) / torch.sum(targets)
    else:
        weights = torch.ones(4, device=device)
    return weights


def bce_focal_loss(outputs, targets, alpha=0.8, gamma=2):
    bce = F.binary_cross_entropy(outputs, targets)
    bce_exp = torch.exp(-bce)
    loss = alpha * (1 - bce_exp) ** gamma * bce

    return loss


def bce_loss(outputs, targets):
    bce = F.binary_cross_entropy(outputs, targets)

    return bce  # * get_class_weights(targets)


def mse_loss(outputs, targets):
    return F.mse_loss(outputs, targets)


def get_centroids(outputs, coord_img):
    mask = (torch.sum(coord_img, dim=1) > 0)
    outputs = torch.cat([outputs[:, c][mask].unsqueeze(0) for c in range(4)]).unsqueeze(0)
    coord_img = torch.cat([coord_img[:, c][mask].unsqueeze(0) for c in range(4)]).unsqueeze(0)
    return outputs, coord_img


def centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return bce_loss(out, coord)


def focal_centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return bce_focal_loss(out, coord)


def centroid_scores(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return scores(out, coord)


################################################ SETTING UP DATASET


if not os.path.exists('sample_paths.csv'):
    print('Building data paths...')

    data = glob.glob("data/exported_tiles/*/*")

    samples = {}
    for d in data:
        dn = d.split(']')[0] + ']'
        sample = samples[dn] if dn in samples else {}
        if '-labelled' in d:
            sample['Mask'] = d
            coords = d.split('[')[-1].split(']')[0].split(',')
            x, y, dimx, dimy = (int(c.split('=')[-1]) for c in coords)

            slide_centroids = pd.read_csv(glob.glob("data/new_exported_coords/{}*".format(dn.split('/')[2]))[0], sep='\t')

            slide_centroids['Centroid X µm'] = slide_centroids['Centroid X µm'] / 0.227
            slide_centroids['Centroid Y µm'] = slide_centroids['Centroid Y µm'] / 0.227

            centroids = slide_centroids.copy()
            centroids['Class'] = centroids['Class'].fillna('0')
            centroids = centroids[centroids['Centroid X µm'] >= x]
            centroids = centroids[centroids['Centroid X µm'] < x + dimx]
            centroids = centroids[centroids['Centroid Y µm'] >= y]
            centroids = centroids[centroids['Centroid Y µm'] < y + dimy]

            for c in range(4):
                sample[CLASS_LIST[c]] = 1 if c in np.array(Image.open(d)) else 0
                coords = list(zip(centroids[centroids['Class'] == CLASS_LIST[c]]['Centroid X µm'].values - x, centroids[centroids['Class'] == CLASS_LIST[c]]['Centroid Y µm'].values - y))
                sample['{}_coords'.format(CLASS_LIST[c])] = coords
        else:
            sample['Img'] = d
        sample['Slide'] = dn.split('/')[2]
        samples[dn] = sample

    samples = pd.DataFrame.from_dict(samples, orient='index')
    samples.to_csv('sample_paths.csv', index=False)
    print('\n', len(samples))
    print('Finished building dataset paths.')
    exit()


class lc_data(Dataset):

    def __init__(self, samples, augment=False):

        self.samples = pd.DataFrame()
        for c in range(1, 4):
            self.samples = pd.concat([self.samples, samples[samples[CLASS_LIST[c]] == 1]])
        self.samples.drop_duplicates(subset=['Img'], inplace=True)

        self.augment = augment

        if augment:
            mirror = [0, 1]
            rots = [0, 90, 180, 270]
            augs = []
            for r in rots:
                for m in mirror:
                    augs.append([m, r])

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

        coord = np.ones_like(img) * - 1
        cr = args.cr
        for c in range(4):
            class_coords = eval(s['{}_coords'.format(CLASS_LIST[c])])
            for x, y in class_coords:
                x1, x2 = max(0, int(x - cr)), min(dim, int(x + cr) + 1)
                y1, y2 = max(0, int(y - cr)), min(dim, int(y + cr) + 1)

                coord[y1:y2, x1:x2] = c

        target = np.array(Image.open(s['Mask']))[:dim, :dim]

        target = np.pad(target, ((0, dim - target.shape[0]), (0, dim - target.shape[1])), 'minimum')

        img, target, coord = torch.Tensor(img.astype(np.float32)).unsqueeze(0), torch.Tensor(target.astype(np.float32)).unsqueeze(0), torch.Tensor(coord.astype(np.float32)).unsqueeze(0)

        if self.augment:
            mir, rot = s['Aug']
            img = TF.rotate(img, rot)
            target = TF.rotate(target, rot)
            coord = TF.rotate(coord, rot)

            if mir == 1:
                img = TF.hflip(img)
                target = TF.hflip(target)
                coord = TF.hflip(coord)

        targets = torch.cat([target] * len(CLASS_LIST), dim=0)
        coords = torch.cat([coord] * len(CLASS_LIST), dim=0)

        for c in range(len(CLASS_LIST)):
            targets[c] = (targets[c] == c).float()
            coords[c] = (coord == c).float()

        # Coallate CD8 cells
        targets[1] += targets[4] + targets[5] + targets[6]
        coords[1] += coords[4] + coords[5] + coords[6]

        # Coallate CD3 cells
        targets[2] += targets[1] + targets[4] + targets[6] + targets[7]
        coords[2] += coords[1] + coords[4] + coords[6] + coords[7]

        # Coallate CD20 cells
        targets[3] += targets[5] + targets[6] + targets[7]
        coords[3] += coords[5] + coords[6] + coords[7]

        targets[targets > 1] = 1
        coords[coords > 1] = 1

        img = normalise(img)
        return img, targets[:4], coords[:4]


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, num_epochs, path, save_freq=100, train=True):
    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0
        scores_list = ['prec', 'rec', 'f1']

        total_scores = np.zeros((4, len(scores_list)))

        if args.centroids:
            total_scores = np.zeros((4, len(scores_list) * 2))
            scores_list.extend(['centroid_{}'.format(s) for s in scores_list])

        if args.lr_decay and epoch > 0:
            print('Decaying learning rate...')
            lr = args.lr / (epoch + 1)

        else:
            lr = args.lr

        if train:
            print('Training...')
            mode = 'train'
            net.train()
            dataloader = train_loader
            optimizer = optim.AdamW(net.parameters(), lr=lr)

        else:
            print('Evaluating...')
            mode = 'val'
            net.eval()
            dataloader = eval_loader
            lr = 0

        save_freq = max(min(save_freq, len(dataloader) - 1), 1)

        for i, data in enumerate(dataloader):

            inputs, targets, coords = data
            inputs, targets, coords = inputs.to(device), targets.to(device), coords.to(device)

            outputs = torch.sigmoid(net(inputs))
            if 'centroid' in args.seg_lf:
                loss = eval(args.seg_lf)(outputs, coords)
            else:
                loss = eval(args.seg_lf)(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            print('({}) Epoch: {}/{} Batch: {}/{} Batch Loss {} LR {}'.format(mode, epoch + 1, num_epochs, i, len(dataloader), loss.item(), lr))

            results = {
                '{}/loss'.format(mode): loss.item(), '{}/mean_loss'.format(mode): total_loss / (i + 1), '{}/lr'.format(mode): lr, '{}/batch'.format(mode): i + 1, '{}/epoch'.format(mode): epoch + 1
                }

            if args.centroids:
                batch_scores = np.zeros_like(total_scores)
                batch_scores[:, :len(scores_list) // 2] = scores(outputs, targets)
                batch_scores[:, len(scores_list) // 2:] = centroid_scores(outputs, coords)
            else:
                batch_scores = scores(outputs, targets)

            total_scores += batch_scores

            print(scores_list)
            print(batch_scores)

            for cls_num in range(4):
                results.update(dict(zip(['{}/{}/'.format(mode, CLASS_LIST[cls_num]) + s for s in scores_list], total_scores[cls_num] / (i + 1))))

            wandb.log(results)

            if (i + 1) % save_freq == 0:
                results["{}/inputs".format(mode)] = [wandb.Image(im) for im in inputs.cpu()[:args.num_wb_img]]

                for cls_im in range(4):
                    results["{}/{}/output".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in outputs[:args.num_wb_img, cls_im].cpu()]
                    results["{}/{}/targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in targets[:args.num_wb_img, cls_im].cpu()]
                    results["{}/{}/coord_targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in coords[:args.num_wb_img, cls_im].cpu()]
                if args.hipe_cells > 0:
                    for cls_im in range(1, 4):
                        print('HiPe for {}'.format(CLASS_LIST[cls_im]))
                        hipe, depth_hipe, hipe_masks = hierarchical_perturbation(net, inputs[0].unsqueeze(0).detach(), target=cls_im, batch_size=1, num_cells=args.hipe_cells, perturbation_type='fade')
                        if torch.sum(hipe) > 0:
                            results["{}/{}/hipe".format(mode, CLASS_LIST[cls_im])] = wandb.Image(hipe.cpu())
                            results["{}/{}/hipe_masks".format(mode, CLASS_LIST[cls_im])] = hipe_masks
                            for ds in range(depth_hipe.shape[0]):
                                results["{}/{}/hipe_depth_{}".format(mode, CLASS_LIST[cls_im], ds)] = wandb.Image(depth_hipe.cpu()[ds])

                wandb.log(results)

                if train:
                    torch.save(net.state_dict(), path)
                    print('Saving model...')
                    with torch.no_grad():
                        run_epochs(net, None, eval_loader, 1, None, train=False, save_freq=args.save_freq)
                    net.train()

    return


if args.large:
    samples = pd.read_csv('large_sample_paths.csv').sample(frac=args.subset / 100, random_state=0).reset_index(drop=True)
else:
    samples = pd.read_csv('sample_paths.csv').sample(frac=args.subset / 100, random_state=0).reset_index(drop=True)

wandb.log({'samples': len(samples)})

train_size = int(0.8 * len(samples))
val_test_size = len(samples) - train_size
val_size = int(0.5 * val_test_size)

if args.stats:
    all_dataset = lc_data(samples, augment=False)
    data_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)
    print(get_stats(data_loader))
    exit()

train_dataset = lc_data(samples[:train_size], augment=args.augment)
val_dataset = lc_data(samples[train_size:train_size + val_size], augment=False)
test_dataset = lc_data(samples[train_size + val_size:], augment=False)

print('{} training samples, {} validation samples, {} test samples...'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)

net = eval(args.model)(encoder_name=args.encoder, in_channels=1, classes=4)

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

print(params_id)
run.finish()
exit()
