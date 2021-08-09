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
parser.add_argument('--conv_thresh', type=float, default=0)
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
parser.add_argument('--weighted_loss', default=False, action='store_true')
parser.add_argument('--dynamic_lr', default=False, action='store_true')
parser.add_argument('--num_wb_img', type=int, default=12)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--cr', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--overlap38', default=False, action='store_true')
parser.add_argument('--sanity', default=False, action='store_true')
parser.add_argument('--large', default=False, action='store_true')
parser.add_argument('--reg_centroids', default=False, action='store_true')
parser.add_argument('--find_bad', default=False, action='store_true')
parser.add_argument('--cells', default=False, action='store_true')
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

CLASS_LIST = ['0', 'CD8', 'CD3', 'CD20', 'CD8: CD3']
if args.cells:
    CLASS_LIST = ['OTHER', 'CD8_CD3LO', 'CD3', 'CD20', 'CD8_CD3HI']
SLIDES = ['L135', 'L149', 'L74', 'L111', 'L93', 'L730']

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
    coord_count = np.zeros(len(CLASS_LIST))
    class_presence = np.zeros(len(CLASS_LIST))
    mean_img = torch.zeros((1, dim, dim))

    for i, data in enumerate(loader):
        print(i, '/', len(loader))
        input, target, coords = data
        mean_img += input[0]
        mean += input.mean()
        std += input.std()
        var += input.var()
        maximum = max(input.max(), maximum)
        minimum = min(input.min(), minimum)
        cc = [torch.sum((coords == c).float()) for c in range(len(CLASS_LIST))]
        coord_count += cc
        class_presence += (np.array(cc) > 0).astype(float)

    mean_img = normalise(mean_img)

    if args.large:
        img_name = 'large_mean_img'
    else:
        img_name = 'mean_img'
    save_im(mean_img, img_name)
    wandb.log({img_name: wandb.Image(mean_img[0])})
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
    score_arr = np.zeros((args.num_classes, 5))

    for cls_num in range(args.num_classes):
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

    for cls_num in range(args.num_classes):
        output = o[:, cls_num]
        target = (t == cls_num).float()

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
    ce = F.cross_entropy(outputs, targets)
    ce_exp = torch.exp(-ce)
    loss = alpha * (1 - ce_exp) ** gamma * ce
    return loss


def ce_loss(outputs, targets):
    return F.cross_entropy(outputs, targets)


def mse_loss(outputs, targets):
    return F.mse_loss(outputs, targets)


def get_centroids(outputs, coord_img):
    mask = (coord_img >= 0)
    outputs = torch.cat([outputs[:, c][mask].unsqueeze(0) for c in range(args.num_classes)]).unsqueeze(0)
    coord_img = coord_img[mask].unsqueeze(0)

    return outputs, coord_img


def centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    if args.reg_centroids:
        return ce_loss(out, coord) + torch.sum(torch.abs(outputs))
    return ce_loss(out, coord)


def focal_centroid_loss(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return ce_focal_loss(out, coord)


def centroid_scores(outputs, coord_img):
    out, coord = get_centroids(outputs, coord_img)
    return scores(out, coord)


################################################ SETTING UP DATASET


if args.large:
    sp = 'large_sample_paths.csv'
    tile_path = "data/new_large_exported_tiles/*/*"
else:
    sp = 'sample_paths.csv'
    tile_path = "data/new_exported_tiles/*/*"
if not args.cells and not os.path.exists(sp):
    print('Building data paths...')

    data = glob.glob(tile_path)

    samples = {}
    for d in data:
        print(d)
        dn = d.split(']')[0] + ']'
        sample = samples[dn] if dn in samples else {}
        if '-labelled' in d:
            sample['Mask'] = d
            coords = d.split('[')[-1].split(']')[0].split(',')
            x, y, dimx, dimy = (int(c.split('=')[-1]) for c in coords)
            slide_centroids = pd.read_csv(glob.glob("data/new_exported_coords/{}*".format(dn.split('/')[2]))[0], sep='\t')

            pixel_size = 0.227

            slide_centroids['Centroid X µm'] = slide_centroids['Centroid X µm'] / pixel_size
            slide_centroids['Centroid Y µm'] = slide_centroids['Centroid Y µm'] / pixel_size

            centroids = slide_centroids.copy()
            centroids = centroids[centroids['Centroid X µm'] >= x]
            centroids = centroids[centroids['Centroid X µm'] < x + dimx]
            centroids = centroids[centroids['Centroid Y µm'] >= y]
            centroids = centroids[centroids['Centroid Y µm'] < y + dimy]
            centroids['Class'].fillna('0', inplace=True)

            for c in range(len(CLASS_LIST)):
                sample[CLASS_LIST[c]] = list(
                        zip(centroids[centroids['Class'] == CLASS_LIST[c]]['Centroid X µm'].values - x, centroids[centroids['Class'] == CLASS_LIST[c]]['Centroid Y µm'].values - y))
        else:
            sample['Img'] = d
        sample['Slide'] = dn.split('/')[2]
        samples[dn] = sample

    samples = pd.DataFrame.from_dict(samples, orient='index')
    samples.fillna('[]', inplace=True)
    if args.large:
        samples.to_csv('large_sample_paths.csv', index=False)
    else:
        samples.to_csv('sample_paths.csv', index=False)
    print('\n', len(samples))
    print('Finished building dataset paths.')
    print(samples.info())
    exit()


class lc_data(Dataset):

    def __init__(self, samples, augment=False):

        # self.samples = samples[samples['CD8: CD3'] != '[]']
        # print(len(self.samples))
        self.samples = samples.drop_duplicates()

        self.augment = augment
        self.sanity_img, self.sanity_target, self.sanity_coord = None, None, None

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
                if args.large:
                    x = x * 2
                    y = y * 2
                if args.cr > 1:
                    rr, cc = disk((int(y), int(x)), args.cr, shape=(dim, dim))
                    coord[rr, cc] = c
                else:
                    coord[int(y), int(x)] = c

        target = np.array(Image.open(s['Mask']))[:dim, :dim]

        target = np.pad(target, ((0, dim - target.shape[0]), (0, dim - target.shape[1])), 'minimum')

        img, target, coord = torch.Tensor(img.astype(np.float32)).unsqueeze(0), torch.Tensor(target.astype(np.float32)).unsqueeze(0), torch.Tensor(coord.astype(np.float32)).unsqueeze(0)

        if (len(torch.unique(target)) == args.num_classes) and (self.sanity_img is None):
            self.sanity_img, self.sanity_target, self.sanity_coord = img, target, coord

        if self.augment:
            mir, rot = s['Aug']
            img = TF.rotate(img, rot)
            target = TF.rotate(target, rot)
            coord = TF.rotate(coord, rot)

            if mir == 1:
                img = TF.hflip(img)
                target = TF.hflip(target)
                coord = TF.hflip(coord)

        if args.sanity and self.sanity_img is not None:
            img, target, coord = self.sanity_img, self.sanity_target, self.sanity_coord

        target[target >= args.num_classes] = 0
        coord[coord >= args.num_classes] = 0
        target, coord = target[0], coord[0]

        img = normalise(img)
        return img, target.long(), coord.long()


class cell_data(Dataset):

    def __init__(self, samples):
        self.samples = samples
        self.dim = dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index]
        mask = normalise(np.array(Image.open(img.split('.tif')[0] + '-mask.png')))
        img = np.array(Image.open(img))
        img = img * gaussian_filter(mask, sigma=1)

        w, h = img.shape
        if h > self.dim:
            ch1 = (h - self.dim) // 2
            ch2 = self.dim - ch1
            img = img[:, ch1:ch2]
            mask = mask[:, ch1:ch2]

        if w > self.dim:
            cw1 = (w - self.dim) // 2
            cw2 = self.dim - cw1
            img = img[cw1:cw2, :]
            mask = mask[cw1:cw2, :]

        w, h = img.shape

        pw1, ph1 = (self.dim - w) // 2, (self.dim - h) // 2
        pw2, ph2 = self.dim - w - pw1, self.dim - h - ph1

        img = np.pad(img, ((pw1, pw2), (ph1, ph2)), 'constant')
        mask = np.pad(mask, ((pw1, pw2), (ph1, ph2)), 'constant')

        img = torch.Tensor(normalise(img)).unsqueeze(0)
        cls = self.samples[index].split('-')[1]
        tc = CLASS_LIST.index(cls)
        target = torch.Tensor(mask) * tc
        coord = torch.ones(self.dim,self.dim) * - 1
        coord[dim // 2, dim // 2] = tc
        return img, target.long(), coord.long()


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, num_epochs, path, save_freq=100, train=True):
    lr = args.lr
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    worst = 1

    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0
        scores_list = ['prec', 'rec', 'f1', 'acc', 'iou']

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
            outputs = net(inputs)

            if 'centroid' in args.seg_lf:
                loss = eval(args.seg_lf)(outputs, coords)
            else:
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

            batch_scores = np.zeros_like(total_scores)
            batch_scores[:, :len(scores_list) // 2] = scores(outputs, targets)
            batch_scores[:, len(scores_list) // 2:] = centroid_scores(outputs, coords)

            total_scores += batch_scores

            print(scores_list)
            print(batch_scores)

            results.update({'{}/avg_f1'.format(mode): (np.sum(total_scores[1:args.num_classes, 2]) / args.num_classes) / (i + 1)})

            results.update({'{}/avg_centroid_f1'.format(mode): (np.sum(total_scores[1:args.num_classes, 5]) / args.num_classes) / (i + 1)})

            for cls_num in range(args.num_classes):
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
                results["{}/coord_targets".format(mode)] = [wandb.Image(im) for im in coords[:args.num_wb_img].float().cpu()]

                for cls_im in range(args.num_classes):
                    results["{}/{}/output".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in outputs[:args.num_wb_img, cls_im].float().cpu()]

                    results["{}/{}/targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(separate_masks(im)[cls_im]) for im in targets[:args.num_wb_img].float().cpu()]
                    results["{}/{}/coord_targets".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(separate_masks(im)[cls_im]) for im in coords[:args.num_wb_img].float().cpu()]

                if args.hipe:

                    for cls_im in range(1, args.num_classes):
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


if not args.cells:
    if args.large:
        samples = pd.read_csv('large_sample_paths.csv').sample(frac=args.subset / 100, random_state=0).reset_index(drop=True)
    else:
        samples = pd.read_csv('sample_paths.csv').sample(frac=args.subset / 100, random_state=0).reset_index(drop=True)

    samples = samples[samples['Slide'].isin(SLIDES)]

    wandb.log({'samples': len(samples), 'slides': len(SLIDES)})

    train_size = int(0.8 * len(samples))
    val_test_size = len(samples) - train_size
    val_size = int(0.5 * val_test_size)

    if args.stats:
        all_dataset = lc_data(samples, augment=False)
        data_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=True, worker_init_fn=np.random.seed(0), num_workers=0)
        stats = get_stats(data_loader)
        print(stats)
        stats = pd.DataFrame.from_dict(stats)
        stats.to_csv('stats.csv', mode='a+')
        exit()

    train_dataset = lc_data(samples[:train_size], augment=args.augment)
    val_dataset = lc_data(samples[train_size:train_size + val_size], augment=False)
    test_dataset = lc_data(samples[train_size + val_size:], augment=False)

else:
    samples = glob.glob("exported_cells/*/*.tif")
    if args.im_only:
        samples = [s for s in samples if 'OTHER' not in s]
    random.shuffle(samples)
    if args.subset < 1:
        samples = samples[:int(len(samples) * args.subset)]

    if args.stats:

        class_count = [len([s for s in samples if c in s]) for c in CLASS_LIST]
        print(dict(zip(CLASS_LIST, class_count)))
        exit()

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

run.finish()
exit()
