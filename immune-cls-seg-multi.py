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
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--conv_thresh', type=float, default=0)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--subset', default=100, type=float)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--model', default='smp.Unet')
parser.add_argument('--encoder', default='resnet34')
parser.add_argument('--seg_lf', default='bce_loss')
parser.add_argument('--large', default=False, action='store_true')
parser.add_argument('--hipe', default=False, action='store_true')
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--stats', default=False, action='store_true')
parser.add_argument('--weighted_loss', default=False, action='store_true')
parser.add_argument('--dynamic_lr', default=False, action='store_true')
parser.add_argument('--strict_classes', default=False, action='store_true')
parser.add_argument('--num_wb_img', type=int, default=12)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--cr', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--interim_val', default=True, action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "immune_seg_multi"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

dim = 256

CLASS_LIST = ['0', 'CD8', 'CD3', 'CD20', 'CD8: CD3', 'CD20: CD3', 'CD20: CD8', 'CD20: CD8: CD3']
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
    coord_count = np.zeros(len(CLASS_LIST))
    class_presence = np.zeros(len(CLASS_LIST))

    for i, data in enumerate(loader):
        # print(i, '/', len(loader))
        input, target, coords = data
        mean += input.mean()
        std += input.std()
        var += input.var()
        maximum = max(input.max(), maximum)
        minimum = min(input.min(), minimum)
        cc = [torch.sum(coords[:, c, :, :]).item() for c in range(len(CLASS_LIST))]
        # print(cc)
        coord_count += cc
        class_presence += (np.array(cc) > 0).astype(float)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    var /= len(loader.dataset)

    coord_perc = coord_count / np.sum(coord_count)

    coord_count = dict(zip(['{}_count'.format(c) for c in CLASS_LIST], coord_count))
    class_presence = dict(zip(['{}_presence'.format(c) for c in CLASS_LIST], class_presence))
    coord_perc = dict(zip(['{}_percent'.format(c) for c in CLASS_LIST], coord_perc))

    stats = {'minimum': minimum, 'maximum': maximum, 'mean': mean, 'std': std, 'var': var}
    stats.update(coord_count)
    stats.update(coord_perc)
    stats.update(class_presence)
    return stats


def separate_masks(masks):
    with torch.no_grad():
        sep_masks = []
        for cls_num in range(len(CLASS_LIST)):
            sep_masks.append((masks == cls_num).float())
    return sep_masks


################################################ METRICS


def scores(o, t):
    score_arr = np.zeros((args.num_classes, 3))

    for cls_num in range(args.num_classes):
        output = o[:, cls_num]
        if args.strict_classes:
            target = (t == cls_num).float()
        else:
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

    for cls_num in range(args.num_classes):
        output = o[:, cls_num]
        if args.strict_classes:
            target = (t == cls_num).float()
        else:
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

    return 1 - f1 / args.num_classes


def get_class_weights(coords):
    if args.weighted_loss:
        weights = torch.zeros(args.num_classes, device=device)
        for cls_num in range(args.num_classes):
            weights[cls_num] += 1 - torch.sum(coords[:, cls_num]) / torch.sum(coords)
    else:
        weights = torch.ones(args.num_classes, device=device)
    return weights


def ce_focal_loss(outputs, targets, alpha=0.8, gamma=2):
    if args.strict_classes:
        ce = F.cross_entropy(outputs, targets)
        ce_exp = torch.exp(-ce)
        loss = alpha * (1 - ce_exp) ** gamma * ce
    else:
        bce = F.binary_cross_entropy(outputs, targets)
        bce_exp = torch.exp(-bce)
        loss = alpha * (1 - bce_exp) ** gamma * bce
    return loss


def ce_loss(outputs, targets):
    if args.strict_classes:
        return F.cross_entropy(outputs, targets)
    return F.binary_cross_entropy(outputs, targets)


def mse_loss(outputs, targets):
    return F.mse_loss(outputs, targets)


def get_centroids(outputs, coord_img):
    if args.strict_classes:
        mask = (coord_img >= 0)
        outputs = torch.cat([outputs[:, c][mask].unsqueeze(0) for c in range(args.num_classes)]).unsqueeze(0)
        coord_img = coord_img[mask]

    else:
        mask = (torch.sum(coord_img, dim=1) > 0)
        outputs = torch.cat([outputs[:, c][mask].unsqueeze(0) for c in range(args.num_classes)]).unsqueeze(0)
        coord_img = torch.cat([coord_img[:, c][mask].unsqueeze(0) for c in range(args.num_classes)]).unsqueeze(0)

    return outputs, coord_img


def centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return ce_loss(out, coord)


def focal_centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return ce_focal_loss(out, coord)


def centroid_scores(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return scores(out, coord)


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
            coords = d.split('[')[-1].split(']')[0].split(',')
            x, y, dimx, dimy = (int(c.split('=')[-1]) for c in coords)
            slide_centroids = pd.read_csv(glob.glob("data/new_exported_coords/{}*".format(dn.split('/')[2]))[0], sep='\t')
            class_list = slide_centroids['Class'].fillna('0').unique()

            slide_centroids['Centroid X µm'] = slide_centroids['Centroid X µm'] / 0.227
            slide_centroids['Centroid Y µm'] = slide_centroids['Centroid Y µm'] / 0.227

            centroids = slide_centroids.copy()
            centroids['Class'] = centroids['Class'].fillna('0')
            centroids = centroids[centroids['Centroid X µm'] >= x]
            centroids = centroids[centroids['Centroid X µm'] < x + dimx]
            centroids = centroids[centroids['Centroid Y µm'] >= y]
            centroids = centroids[centroids['Centroid Y µm'] < y + dimy]

            for c in range(len(class_list)):
                coords = list(zip(centroids[centroids['Class'] == class_list[c]]['Centroid X µm'].values - x, centroids[centroids['Class'] == class_list[c]]['Centroid Y µm'].values - y))
                sample[class_list[c]] = coords
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

        self.samples = samples.fillna('[]')
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
        for c in range(len(CLASS_LIST)):
            class_coords = eval(s[CLASS_LIST[c]])
            for x, y in class_coords:
                if args.cr > 1:
                    rr, cc = disk((int(y), int(x)), args.cr, shape=(dim, dim))
                    coord[rr, cc] = c
                else:
                    coord[int(y), int(x)] = c

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

        if not args.strict_classes:

            targets = torch.cat([target] * len(CLASS_LIST), dim=0)
            coords = torch.cat([coord] * len(CLASS_LIST), dim=0)

            for c in range(len(CLASS_LIST)):
                targets[c] = (target == c).float()
                coords[c] = (coord == c).float()

            # Coallate CD8 cells
            targets[1] += targets[4] + targets[6] + targets[7]
            coords[1] += coords[4] + coords[6] + coords[7]

            # Coallate CD3 cells
            targets[2] += targets[4] + targets[5] + targets[7]
            coords[2] += coords[4] + coords[5] + coords[7]

            # Coallate CD20 cells
            targets[3] += targets[5] + targets[6] + targets[7]
            coords[3] += coords[5] + coords[6] + coords[7]

            targets[targets > 1] = 1
            coords[coords > 1] = 1

            target, coord = targets[:args.num_classes], coords[:args.num_classes]

        else:

            target[target >= args.num_classes] = 0
            coord[coord >= args.num_classes] = 0
            target, coord = target.squeeze(0).long(), coord.squeeze(0)

        img = normalise(img)

        return img, target, coord


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, num_epochs, path, save_freq=100, train=True):
    lr = args.lr
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0
        scores_list = ['prec', 'rec', 'f1']

        total_scores = np.zeros((args.num_classes, len(scores_list)))

        total_scores = np.zeros((args.num_classes, len(scores_list) * 2))
        scores_list.extend(['centroid_{}'.format(s) for s in scores_list])

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

            inputs, targets, coords = data
            inputs, targets, coords = inputs.to(device), targets.to(device), coords.to(device)

            if args.strict_classes:
                outputs = torch.softmax(net(inputs), dim=1)
            else:
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

            print('{} ({}) Epoch: {}/{} Batch: {}/{} Batch Loss {} LR {}'.format(params_id, mode, epoch + 1, num_epochs, i, len(dataloader), loss.item(), lr))

            results = {
                '{}/loss'.format(mode): loss.item(), '{}/mean_loss'.format(mode): total_loss / (i + 1), '{}/lr'.format(mode): lr, '{}/batch'.format(mode): i + 1, '{}/epoch'.format(mode): epoch + 1
                }

            batch_scores = np.zeros_like(total_scores)
            batch_scores[:, :len(scores_list) // 2] = scores(outputs, targets)
            batch_scores[:, len(scores_list) // 2:] = centroid_scores(outputs, coords)

            total_scores += batch_scores

            print(scores_list)
            print(batch_scores)

            results.update({'{}/avg_f1'.format(mode): (np.sum(total_scores[:, 2]) / args.num_classes) / (i + 1)})

            results.update({'{}/avg_centroid_f1'.format(mode): (np.sum(total_scores[:, 5]) / args.num_classes) / (i + 1)})

            for cls_num in range(args.num_classes):
                results.update(dict(zip(['{}/{}/'.format(mode, CLASS_LIST[cls_num]) + s for s in scores_list], total_scores[cls_num] / (i + 1))))

            wandb.log(results)

            if (i + 1) % save_freq == 0:
                results["{}/inputs".format(mode)] = [wandb.Image(im) for im in inputs.cpu()[:args.num_wb_img]]

                for cls_im in range(args.num_classes):
                    results["{}/{}/output".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in outputs[:args.num_wb_img, cls_im].cpu()]
                    if args.strict_classes:
                        results["{}/{}/targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(separate_masks(im)) for im in targets[:args.num_wb_img, cls_im].cpu()]
                        results["{}/{}/coord_targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(separate_masks(im)) for im in coords[:args.num_wb_img, cls_im].cpu()]
                    else:

                        results["{}/{}/targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in targets[:args.num_wb_img, cls_im].cpu()]
                        results["{}/{}/coord_targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in coords[:args.num_wb_img, cls_im].cpu()]
                if args.hipe:
                    for cls_im in range(1, args.num_classes):
                        print('HiPe for {}'.format(CLASS_LIST[cls_im]))
                        hipe, depth_hipe, hipe_masks = hierarchical_perturbation(net, inputs[0].unsqueeze(0).detach(), target=cls_im, batch_size=1, num_cells=4, perturbation_type='fade')
                        if torch.sum(hipe) > 0:
                            results["{}/{}/hipe".format(mode, CLASS_LIST[cls_im])] = wandb.Image(hipe.cpu())
                            results["{}/{}/hipe_masks".format(mode, CLASS_LIST[cls_im])] = hipe_masks
                            for ds in range(depth_hipe.shape[0]):
                                results["{}/{}/hipe_depth_{}".format(mode, CLASS_LIST[cls_im], ds)] = wandb.Image(depth_hipe.cpu()[ds])

                wandb.log(results)

                if train and args.interim_val:
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

net = eval(args.model)(encoder_name=args.encoder, in_channels=1, classes=args.num_classes)

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
