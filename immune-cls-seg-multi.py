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
parser.add_argument('--large', default=False, action='store_true')
parser.add_argument('--hipe_cells', type=int, default=4)
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--stats', default=False, action='store_true')
parser.add_argument('--centroids', default=False, action='store_true')
parser.add_argument('--weighted_loss', default=False, action='store_true')
parser.add_argument('--dynamic_lr', default=False, action='store_true')
parser.add_argument('--classes', type=int, default=-1)
parser.add_argument('--num_wb_img', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "immune_seg_multi"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
print(params_id, args)

dim = 256

CLASS_LIST = ['0', 'CD8', 'CD3', 'CD20', 'CD8: CD3', 'CD20: CD8', 'CD20: CD8: CD3', 'CD20: CD3'][:args.classes]
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
    cls_count = torch.zeros(len(CLASS_LIST))
    cls_covg = torch.zeros(len(CLASS_LIST))
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
    cls_count = {'{}_count'.format(i): cls_count[i] / len(loader.dataset) for i in range(len(CLASS_LIST))}
    cls_covg = {'{}_covg'.format(i): cls_covg[i] / len(loader.dataset) for i in range(len(CLASS_LIST))}

    stats = {'minimum': minimum, 'maximum': maximum, 'mean': mean, 'std': std, 'var': var}
    stats.update(cls_count)
    stats.update(cls_covg)
    return stats


def separate_masks(masks):
    with torch.no_grad():
        sep_masks = []
        for cls_num in range(len(CLASS_LIST)):
            mask = masks.clone()
            mask[mask != cls_num] = -1
            mask[mask > -1] = 0
            mask += 1
            sep_masks.append(mask)
    return sep_masks


################################################ METRICS


def scores(o, t):
    score_arr = np.zeros((len(CLASS_LIST), 3))

    for cls_num in range(len(CLASS_LIST)):
        output = o[:, cls_num]
        target = t.float()
        flat_ix = target.reshape(-1) > -1
        target[target != cls_num] = -1
        target[target > -1] = 0
        target += 1

        output = torch.masked_select(output.reshape(-1), flat_ix)
        target = torch.masked_select(target.reshape(-1), flat_ix)

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

    for cls_num in range(len(CLASS_LIST)):
        output = o[:, cls_num]
        target = t.float()
        target[target != cls_num] = -1
        target[target > -1] = 0
        target += 1

        tp = torch.sum(target * output)
        fp = torch.sum((1 - target) * output)
        fn = torch.sum(target * (1 - output))

        p = tp / (tp + fp + 0.0001)
        r = tp / (tp + fn + 0.0001)

        f1 += 2 * p * r / (p + r + 0.0001)

    return 1 - f1 / len(CLASS_LIST)


def get_class_weights(targets):
    if args.weighted_loss:
        weights = torch.zeros(len(CLASS_LIST), device=device)
        for cls_num in range(len(CLASS_LIST)):
            weights[cls_num] += 1 - targets[targets == cls_num].reshape(-1).shape[0] / targets.reshape(-1).shape[0]
    else:
        weights = torch.ones(len(CLASS_LIST), device=device)
    return weights


def focal_loss(outputs, targets, alpha=0.8, gamma=2):
    CE = F.cross_entropy(outputs, targets, reduction='mean', weight=get_class_weights(targets.detach()))
    CE_EXP = torch.exp(-CE)
    focal_loss = alpha * (1 - CE_EXP) ** gamma * CE

    return focal_loss


def centroid_loss(outputs, coord_img):
    coord_img_z = coord_img.clone()
    coord_img_z[coord_img_z >= 0] = 1
    coord_img_z[coord_img_z < 0] = 0
    outputs_z = outputs * coord_img_z.unsqueeze(1)
    return focal_loss(outputs_z, coord_img)


def centroid_scores(outputs, coord_img):
    coord_img_z = coord_img.clone()
    coord_img_z[coord_img_z >= 0] = 1
    coord_img_z[coord_img_z < 0] = 0
    outputs_z = outputs * coord_img_z.unsqueeze(1)
    return scores(outputs_z, coord_img)


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

            for c in range(len(CLASS_LIST)):
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
        for c in range(1, len(CLASS_LIST)):
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

        coords_img = np.ones_like(img) * - 1
        for c in range(len(CLASS_LIST)):
            class_coords = eval(s['{}_coords'.format(CLASS_LIST[c])])
            for x, y in class_coords:
                coords_img[int(y), int(x)] = c

        target = np.array(Image.open(s['Mask']))[:dim, :dim]

        target = np.pad(target, ((0, dim - target.shape[0]), (0, dim - target.shape[1])), 'minimum')

        img, target, coords_img = torch.Tensor(img.astype(np.float32)).unsqueeze(0), torch.Tensor(target.astype(np.float32)).unsqueeze(0), torch.Tensor(coords_img.astype(np.float32)).unsqueeze(0)

        target[target >= len(CLASS_LIST)] = 0

        if self.augment:
            mir, rot = s['Aug']
            img = TF.rotate(img, rot)
            target = TF.rotate(target, rot)
            coords_img = TF.rotate(coords_img, rot)

            if mir == 1:
                img = TF.hflip(img)
                target = TF.hflip(target)
                coords_img = TF.hflip(coords_img)

        img = normalise(img)

        return img, target[0].long(), coords_img[0].long()


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, seg_criterion, num_epochs, path, save_freq=100, train=True):
    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0
        scores_list = ['prec', 'rec', 'f1']

        total_scores = np.zeros((len(CLASS_LIST), len(scores_list)))

        if args.centroids:
            total_scores = np.zeros((len(CLASS_LIST), len(scores_list) * 2))
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

            outputs = net(inputs)
            loss = seg_criterion(outputs, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            print('({}) Epoch: {}/{} Batch: {}/{} Batch Loss {} LR {}'.format(mode, epoch + 1, num_epochs, i, len(dataloader), loss.item(), lr))

            results = {
                '{}/loss'.format(mode): loss.item(), '{}/mean_loss'.format(mode): total_loss / (i + 1), '{}/lr'.format(mode): lr, '{}/batch'.format(mode): i + 1, '{}/epoch'.format(mode): epoch + 1
                }

            outputs = F.softmax(outputs, dim=1)

            if args.centroids:
                batch_scores = np.zeros_like(total_scores)
                batch_scores[:, :len(scores_list) // 2] = scores(outputs, targets)
                batch_scores[:, len(scores_list) // 2:] = centroid_scores(outputs, coords)
            else:
                batch_scores = scores(outputs, targets)

            total_scores += batch_scores

            print(scores_list)
            print(batch_scores)

            for cls_num in range(len(CLASS_LIST)):
                results.update(dict(zip(['{}/{}/'.format(mode, CLASS_LIST[cls_num]) + s for s in scores_list], total_scores[cls_num] / (i + 1))))

            wandb.log(results)

            if (i + 1) % save_freq == 0:
                results["{}/inputs".format(mode)] = [wandb.Image(im) for im in inputs.cpu()[:args.num_wb_img]]
                results["{}/orig_targets".format(mode)] = [wandb.Image(im) for im in targets[:args.num_wb_img].float().cpu()]
                results["{}/coord_targets".format(mode)] = [wandb.Image(im) for im in coords[:args.num_wb_img].float().cpu()]

                for cls_im in range(len(CLASS_LIST)):
                    results["{}/{}/output".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in outputs[:args.num_wb_img, cls_im].cpu()]
                    results["{}/{}/target_masks".format(mode, CLASS_LIST[cls_im])] = [wandb.Image(im) for im in separate_masks(targets[:args.num_wb_img].float().cpu())[cls_im]]

                if args.hipe_cells > 0:
                    for cls_im in range(1, len(CLASS_LIST)):
                        print('HiPe for {}'.format(CLASS_LIST[cls_im]))
                        hipe, depth_hipe, hipe_masks = hierarchical_perturbation(net, inputs[0].unsqueeze(0).detach(), target=cls_im, batch_size=1, num_cells=args.hipe_cells, perturbation_type='fade')
                        results["{}/{}/hipe".format(mode, CLASS_LIST[cls_im])] = wandb.Image(hipe.cpu())
                        results["{}/{}/hipe_masks".format(mode, CLASS_LIST[cls_im])] = hipe_masks
                        for ds in range(depth_hipe.shape[0]):
                            results["{}/{}/hipe_depth_{}".format(mode, CLASS_LIST[cls_im], ds)] = wandb.Image(depth_hipe.cpu()[ds])

                wandb.log(results)

                if train:
                    torch.save(net.state_dict(), path)
                    print('Saving model...')
                    with torch.no_grad():
                        run_epochs(net, None, eval_loader, seg_criterion, 1, None, train=False, save_freq=args.save_freq)
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

net = eval(args.model)(encoder_name=args.encoder, in_channels=1, classes=len(CLASS_LIST))

seg_criterion = eval(args.seg_lf)

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
    run_epochs(net, train_loader, eval_loader, seg_criterion, num_epochs, 'params/' + params_id + '.pth', save_freq=args.save_freq)

elif args.test:
    with torch.no_grad():
        run_epochs(net, None, test_loader, seg_criterion, 1, None, train=False, save_freq=args.save_freq)

else:
    with torch.no_grad():
        run_epochs(net, None, eval_loader, seg_criterion, 1, None, train=False, save_freq=args.save_freq)

print(params_id)
run.finish()
exit()
