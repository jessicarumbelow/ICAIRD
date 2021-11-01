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
import random
from pathlib import Path

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
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--save_freq', default=200, type=int)
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
parser.add_argument('--num_wb_img', type=int, default=12)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--rot_aug', type=int, default=10)
parser.add_argument('--shear_aug', type=int, default=10)
parser.add_argument('--scale_aug', type=float, default=0)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--sanity', default=False, action='store_true')
parser.add_argument('--find_bad', default=False, action='store_true')
parser.add_argument('--online_augment', default=False, action='store_true')
parser.add_argument('--random_split', default=False, action='store_true')
parser.add_argument('--evaluate_slides', default=False, action='store_true')
parser.add_argument('--fold', default=False, action='store_true')
parser.add_argument('--ds', default=False, action='store_true')
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--binary_class', default='')
parser.add_argument('--cd3', default=False, action='store_true')
parser.add_argument('--centroids', default=False, action='store_true')
parser.add_argument('--weighted_loss', default=False, action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "immune_seg_multi"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args, reinit=True)
args = wandb.config
params_id = wandb.run.name

print(params_id, args)
DIM = 256

ALL_CLASS_LIST = ['0', 'CD8', 'CD3', 'CD20']

CLASS_LIST = ALL_CLASS_LIST

if args.cd3:
    CLASS_LIST = ['0', 'CD3', 'CD20']

if args.binary_class != '':
    CLASS_LIST = ['0', args.binary_class]

ALL_SLIDES = ['L730', 'L749', 'L135', 'L149', 'L70', 'L722', 'L102', 'L111', 'L74', 'L93', 'L114', 'L47']

NUM_CLASSES = len(CLASS_LIST)

print(CLASS_LIST, NUM_CLASSES)
wandb.log({'Class list': CLASS_LIST})


################################################ HELPERS


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def separate_masks(masks):
    with torch.no_grad():
        sep_masks = []
        for cls_num in range(NUM_CLASSES):
            m = (masks == cls_num).float()
            sep_masks.append(m)

    return sep_masks


def save_stats(loader, note='lc'):
    sum_pix = 0
    sum_sq = 0
    num_pix = len(loader) * DIM * DIM
    minimum = 9999999
    maximum = 0

    for i, input in enumerate(loader):
        sum_pix += input.sum()
        maximum = max(input.max(), maximum)
        minimum = min(input.min(), minimum)
    mean = sum_pix / num_pix

    for i, input in enumerate(loader):
        sum_sq += ((input - mean).pow(2)).sum()
    std = torch.sqrt(sum_sq / num_pix)

    st = np.array([mean, std, minimum, maximum])
    np.savetxt('{}_stats.csv'.format(note), st, delimiter=',')

    print('{} stats: {}'.format(note, st))
    return


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


def get_centroids(outputs, coord_img):
    mask = (coord_img >= 0)
    outputs = torch.cat([outputs[:, c][mask].unsqueeze(0) for c in range(NUM_CLASSES)]).unsqueeze(0)
    coord_img = coord_img[mask].unsqueeze(0)

    return outputs, coord_img


def correct_pixels(outputs, targets):
    return torch.sum(torch.argmax(outputs, dim=1) == targets) / (DIM * DIM * outputs.shape[0])


def correct_centroids(outputs, coord_img):
    outputs, coord_img = get_centroids(outputs, coord_img)
    return torch.sum(torch.argmax(outputs, dim=1) == coord_img) / (coord_img[coord_img >= 0]).reshape(-1).shape[0]


def centroid_scores(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return scores(out, coord)


def f1_loss(o, t):
    o = torch.softmax(o, dim=1)
    f1 = torch.zeros(1, device=device)

    for cls_num in range(NUM_CLASSES):
        output = o[:, cls_num]
        target = (t == cls_num).float()

        tp = torch.sum(target * output)
        fp = torch.sum((1 - target) * output)
        fn = torch.sum(target * (1 - output))

        p = tp / (tp + fp + 0.0001)
        r = tp / (tp + fn + 0.0001)

        f1 += (2 * p * r) / (p + r + 0.0001)

    return 1 - (f1 / NUM_CLASSES)


def f1_centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return f1_loss(out, coord)


def ce_focal_loss(outputs, targets, alpha=0.8, gamma=2):
    ce = F.cross_entropy(outputs, targets)
    ce_exp = torch.exp(-ce)
    loss = alpha * (1 - ce_exp) ** gamma * ce
    return loss


def ce_loss(outputs, targets):
    weight = [1] * NUM_CLASSES
    if args.weighted_loss:
        weight[0] = 0
    return F.cross_entropy(outputs, targets, weight=torch.Tensor(weight).float().to(device))


def ce_centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return ce_loss(out, coord)


def mse_loss(outputs, targets):
    return F.mse_loss(outputs, targets)


################################################ SETTING UP DATASET


class lc_data(Dataset):

    def __init__(self, samples, cell_data, augment=False, online_augment=False):
        self.samples = samples
        self.augment = augment
        if args.centroids:
            self.cell_data = cell_data
            self.pixel_size = 0.227
            self.cell_data['Centroid X µm'] = pd.to_numeric(self.cell_data['Centroid X µm'])
            self.cell_data['Centroid Y µm'] = pd.to_numeric(self.cell_data['Centroid Y µm'])

            if args.cd3:
                self.cell_data.loc[self.cell_data['Class'] == 'CD8', 'Class'] = 'CD3'

            if args.binary_class != '':
                self.cell_data.drop(self.cell_data[self.cell_data['Class'] != args.binary_class].index, inplace=True)

        self.online_augment = online_augment
        if self.augment:
            self.augs = list(range(0, 360, 90))
            num_augs = len(self.augs)
            self.augs *= len(self.samples)
            self.samples *= num_augs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name = self.samples[index]
        target = Image.open(img_name.split('.tif')[0] + '-labelled.png')
        target = torch.Tensor(np.array(target))

        id = img_name.split('/')[1].split('_')[0]

        img = np.array(Image.open(img_name)).astype(np.float)
        img = torch.Tensor(img).unsqueeze(0)
        img = normalise(img)
        img = TF.normalize(img, torch.mean(img), torch.std(img))

        coord = np.zeros_like(target) - 1
        if args.centroids:
            coords = img_name.split('[')[-1].split(']')[0].split(',')
            x, y, dimx, dimy = (int(int(c.split('=')[-1]) * self.pixel_size) for c in coords)
            centroids = self.cell_data.copy()
            centroids = centroids[centroids['Centroid X µm'] >= x]
            centroids = centroids[centroids['Centroid X µm'] < x + dimx]
            centroids = centroids[centroids['Centroid Y µm'] >= y]
            centroids = centroids[centroids['Centroid Y µm'] < y + dimy]

            for i, c in centroids.iterrows():
                cls = CLASS_LIST.index(c['Class'])
                cx, cy = int(c['Centroid X µm']) - x, int(c['Centroid Y µm']) - y
                coord[int(cy / self.pixel_size), int(cx / self.pixel_size)] = cls

        coord = torch.Tensor(coord)

        if self.augment:
            rot = self.augs[index]
            img = TF.rotate(img, rot)
            coord = TF.rotate(coord.unsqueeze(0), rot)[0]
            target = TF.rotate(target.unsqueeze(0), rot)[0]

        if self.online_augment:

            angle, scale, shear = random.randint(-args.rot_aug, args.rot_aug), 1 + (np.random.randn() * args.scale_aug), [random.randint(-args.shear_aug, args.shear_aug),
                                                                                                                          random.randint(-args.shear_aug, args.shear_aug)]
            img = TF.affine(img, angle=angle, translate=[0, 0], scale=scale, shear=shear, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR)
            coord = TF.affine(coord.unsqueeze(0), angle=angle, translate=[0, 0], scale=scale, shear=shear, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)[0]
            target = TF.affine(target.unsqueeze(0), angle=angle, translate=[0, 0], scale=scale, shear=shear, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)[0]

        if args.cd3:
            target[target == ALL_CLASS_LIST.index('CD8')] = ALL_CLASS_LIST.index('CD3')

        if args.binary_class != '':
            target[target != ALL_CLASS_LIST.index(args.binary_class)] = 0
            target[target != 0] = 1

        return img, target.long(), coord, id


################################################ TRAINING


def run_epochs(net, train_loader, val_loader, num_epochs, path, save_freq=100, train=True, lv=999999, esc=args.early_stopping):
    lr = args.lr
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=args.decay)
    worst = 1
    best_val_loss = lv
    es_countdown = esc

    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        mean_loss = 0
        scores_list = ['prec', 'rec', 'f1', 'acc', 'iou']

        total_scores = np.zeros((NUM_CLASSES, len(scores_list)))
        total_centroid_scores = np.zeros((NUM_CLASSES, len(scores_list)))

        if args.lr_decay and epoch > 0:
            print('Decaying learning rate...')
            lr = args.lr / (epoch + 1)
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=args.decay)

        if train:
            print('Training...')
            mode = 'train'
            net.train()
            dataloader = train_loader

        else:
            print('Evaluating...')
            mode = 'val'
            net.eval()
            dataloader = val_loader
            lr = 0

        save_freq = max(min(save_freq, len(dataloader) - 1), 1)

        for i, data in enumerate(dataloader):

            inputs, targets, coords, slide_ids = data
            inputs, targets, coords = inputs.to(device), targets.to(device), coords.to(device)
            outputs = net(inputs)

            if 'centroid' in args.seg_lf:
                targets = coords

            if args.seg_lf == 'combined_ce':
                loss = (ce_loss(outputs, targets) + ce_centroid_loss(outputs, coords)) / 2
            elif args.seg_lf == 'combined_f1':
                loss = (f1_loss(outputs, targets) + f1_centroid_loss(outputs, coords)) / 2
            else:
                loss = eval(args.seg_lf)(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss.item() / len(dataloader)
            correct_pix = correct_pixels(outputs, targets)

            slide_f1 = dict(zip(['{}/{}'.format(mode, s) for s in ALL_SLIDES], [0] * len(ALL_SLIDES)))
            for s in range(inputs.shape[0]):
                s_id = slide_ids[s]
                s_id_count = slide_ids.count(s_id)
                slide_loss = 1 - f1_loss(outputs[s].detach().unsqueeze(0), targets[s].detach().unsqueeze(0)).cpu()
                slide_f1['{}/{}'.format(mode, s_id)] += (slide_loss / s_id_count)

            outputs = torch.softmax(outputs, dim=1)

            print('{} ({}) Epoch: {}/{} Batch: {}/{} Batch Loss {} Correct Pixels: {} LR {}'.format(params_id, mode, epoch + 1, num_epochs, i, len(dataloader), loss.item(), correct_pix,
                                                                                                    lr))

            results = {
                '{}/loss'.format(mode):  loss.item(), '{}/correct_pixels'.format(mode): correct_pix, '{}/mean_loss'.format(mode): mean_loss / (i + 1), '{}/lr'.format(mode): lr,
                '{}/batch'.format(mode): i + 1, '{} / epoch'.format(mode): epoch + 1
                }

            results.update(slide_f1)
            batch_scores = scores(outputs, targets)
            batch_centroid_scores = centroid_scores(outputs, coords)
            total_scores += batch_scores
            total_centroid_scores += batch_centroid_scores

            print(scores_list)
            print(batch_scores)
            if args.centroids:
                print(batch_centroid_scores)

            for s in range(len(scores_list)):
                results.update({'{}/avg_{}'.format(mode, scores_list[s]): (np.sum(total_scores[:, s]) / NUM_CLASSES) / (i + 1)})
                results.update({'{}/avg_centroid_{}'.format(mode, scores_list[s]): (np.sum(total_centroid_scores[:, s]) / NUM_CLASSES) / (i + 1)})

            for cls_num in range(NUM_CLASSES):
                results.update(dict(zip(['{}/{}/'.format(mode, CLASS_LIST[cls_num]) + s for s in scores_list], total_scores[cls_num] / (i + 1))))
                results.update(dict(zip(['{}/centroid_{}/'.format(mode, CLASS_LIST[cls_num]) + s for s in scores_list], total_centroid_scores[cls_num] / (i + 1))))

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
                    results["{}/{}/target_coords".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(separate_masks(im)[cls_im]) for im in coords[:args.num_wb_img].float().cpu()]

                if args.hipe:

                    for cls_im in range(1, NUM_CLASSES + 1):
                        hbs = []
                        for hb in range(args.batch_size):

                            if cls_im in targets[hb]:
                                print('HiPe for {}'.format(CLASS_LIST[cls_im]))
                                hipe, hipe_depth, hipe_masks = hierarchical_perturbation(net, inputs[hb].unsqueeze(0).detach(), target=cls_im, batch_size=32, perturbation_type='fade')
                                hbs.append(wandb.Image(hipe_depth.cpu() * (targets[hb] > 0).cpu()))

                            else:
                                hbs.append(wandb.Image(torch.zeros_like(inputs[hb].unsqueeze(0))))

                        results["{}/{}/hipe".format(mode, CLASS_LIST[cls_im])] = hbs

            wandb.log(results, commit=True)

        if train:
            with torch.no_grad():
                best_val_loss, es_countdown = run_epochs(net, None, val_loader, 1, path, train=False, save_freq=args.save_freq, lv=best_val_loss, esc=es_countdown)
            net.train()

        else:
            print('Last val loss: {} Current val_loss: {}'.format(best_val_loss, mean_loss))
            'Early stopping countdown: {}'.format(es_countdown)
            wandb.log({'es_countdown': es_countdown, 'val/mean_loss': mean_loss})
            if mean_loss > best_val_loss:
                es_countdown -= 1
                if es_countdown == 0:
                    print('Early stopping - val loss did not improve after {} epochs'.format(args.early_stopping))
                    exit()
            else:
                best_val_loss = mean_loss
                wandb.log({'val/best_loss': best_val_loss})
                es_countdown = args.early_stopping
                print('Saving model...')
                torch.save(net.state_dict(), path)

    return best_val_loss, es_countdown


if args.ds:
    samples = glob.glob("ds2_exported_tiles/*/*.tif")

else:
    samples = glob.glob("newer_exported_tiles/*/*.tif")

cell_data = pd.read_csv("cell_detections.csv")

samples = [s for s in samples if s.split('/')[1].split('_')[0] in ALL_SLIDES]

random.shuffle(samples)
if args.subset < 1:
    samples = samples[:int(len(samples) * args.subset)]

wandb.log({'samples': len(samples)})

if args.random_split:
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))
    n_test = len(samples) - n_train - n_val
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

else:
    VAL_SLIDES = ['L93', 'L114', 'L47']
    TEST_SLIDES = ['L102', 'L111', 'L74', ]
    TRAIN_SLIDES = ['L730', 'L749', 'L135', 'L149', 'L70', 'L722']

    train_samples = [s for s in samples if s.split('/')[1].split('_')[0] in TRAIN_SLIDES]
    val_samples = [s for s in samples if s.split('/')[1].split('_')[0] in VAL_SLIDES]
    test_samples = [s for s in samples if s.split('/')[1].split('_')[0] in TEST_SLIDES]

"""
for slide in ALL_SLIDES:
    slide_dataset = lc_data([s for s in samples if s.split('/')[1].split('_')[0] == slide])
    slide_loader = torch.utils.data.DataLoader(slide_dataset, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), drop_last=False)
    save_stats(slide_loader, slide)
exit()
"""

train_dataset = lc_data(train_samples, cell_data, augment=args.augment, online_augment=args.online_augment)
val_dataset = lc_data(val_samples, cell_data)
test_dataset = lc_data(test_samples, cell_data)

print({'train_slides': len(train_samples), 'val_slides': len(val_samples), 'test_slides': len(test_samples)})
wandb.log({'train_slides': len(train_samples), 'val_slides': len(val_samples), 'test_slides': len(test_samples)})

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), drop_last=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), drop_last=False)

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
    _, _ = run_epochs(net, train_loader, val_loader, num_epochs, 'params/' + params_id + '.pth', save_freq=args.save_freq)

elif args.test:
    with torch.no_grad():
        _, _ = run_epochs(net, None, test_loader, 1, None, train=False, save_freq=args.save_freq)

elif args.evaluate_slides:
    for slide in ALL_SLIDES:
        slide_samples = lc_data([s for s in samples if s.split('/')[1].split('_')[0] == slide], cell_data)
        print('Slide: {}, {} samples.'.format(slide, len(slide_samples)))
        slide_test_loader = torch.utils.data.DataLoader(slide_samples, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), drop_last=False)
        _, _ = run_epochs(net, None, slide_test_loader, 1, None, train=False, save_freq=args.save_freq)

elif args.fold:
    run.finish()
    for slide in ALL_SLIDES:
        run = wandb.init(project=proj, entity="jessicamarycooper", config=args, reinit=True)
        net = eval(args.model)(encoder_name=args.encoder, in_channels=1, classes=NUM_CLASSES)
        net.to(device)
        wandb.watch(net)
        wandb.log({'Fold Slide': slide})
        print('Fold Slide: {}'.format(slide))
        train_samples = lc_data([s for s in samples if s.split('/')[1].split('_')[0] != slide])
        val_samples = lc_data([s for s in samples if s.split('/')[1].split('_')[0] == slide])
        train_loader = torch.utils.data.DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_samples, batch_size=args.batch_size, shuffle=True, worker_init_fn=np.random.seed(0), drop_last=False)
        _, _ = run_epochs(net, train_loader, val_loader, num_epochs, 'params/' + params_id + '.pth', save_freq=args.save_freq)

        run.finish()
else:
    with torch.no_grad():
        _, _ = run_epochs(net, None, val_loader, 1, None, train=False, save_freq=args.save_freq)

run.finish()
exit()
