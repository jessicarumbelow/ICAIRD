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
from all_cell_cleanup import *

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
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--subset', default=1, type=float)
parser.add_argument('--num_nodes', default=4, type=float)
parser.add_argument('--num_layers', default=1, type=float)
parser.add_argument('--kernel_size', default=3, type=float)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--combined', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--binary_class', default='')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--lf', default='weighted_ ce_loss')
parser.add_argument('--hipe', default=False, action='store_true')
parser.add_argument('--lr_decay', default=False, action='store_true')
parser.add_argument('--im_only', default=False, action='store_true')
parser.add_argument('--augment', default=False, action='store_true')
parser.add_argument('--standardise', default=False, action='store_true')
parser.add_argument('--normalise', default=False, action='store_true')
parser.add_argument('--tabular', default=False, action='store_true')
parser.add_argument('--pretrained', default=False, action='store_true')
parser.add_argument('--slide_losses', default=False, action='store_true')
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

ALL_CLASS_LIST = ['OTHER', 'CD8', 'CD3']

if args.binary_class != '':
    CLASS_LIST = ['OTHER', args.binary_class]
elif args.combined:
    CLASS_LIST = ['OTHER', 'CD3']
elif args.im_only:
    CLASS_LIST = ['CD8', 'CD3']
else:
    CLASS_LIST = ALL_CLASS_LIST

LUNG_SLIDES = ['Lung{}'.format(l) for l in [47, 70, 74, 102, 111, 114, 135, 149, 730, 93]]
COLON_SLIDES = ['Colon{}'.format(c) for c in [188, 189, 233, 276, 372, 468, 504, 514, 553, 569]]
ALL_SLIDES = LUNG_SLIDES + COLON_SLIDES
VAL_SLIDES = random.choices(ALL_SLIDES, k=4)
TRAIN_SLIDES = list(set(ALL_SLIDES) - set(VAL_SLIDES))
TEST_SLIDES = random.choices(ALL_SLIDES, k=4)
TRAIN_SLIDES = list(set(TRAIN_SLIDES) - set(TEST_SLIDES))

NUM_CLASSES = len(CLASS_LIST)

print('TRAIN_SLIDES {}'.format(TRAIN_SLIDES))
print('VAL_SLIDES {}'.format(VAL_SLIDES))
print('TEST_SLIDES {}'.format(TEST_SLIDES))

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


def weighted_ce_loss(outputs, targets):
    weights = 1 - torch.Tensor([torch.sum((targets == t).float()) for t in range(len(CLASS_LIST))]).to(device) / \
              targets.shape[0]
    return F.cross_entropy(outputs, targets, weights)


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


def bce_loss(outputs, targets):
    return F.binary_cross_entropy_with_logits(outputs.reshape(-1).float(), targets.float())


def scores(o, t):
    score_arr = np.zeros((NUM_CLASSES, 4))

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

        score_arr[cls_num] = np.array([p.item(), r.item(), f1.item(), acc.item()])
    return score_arr


class custom_convnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, args.num_nodes, kernel_size=args.kernel_size)
        self.bn1 = nn.BatchNorm2d(args.num_nodes)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(args.num_nodes, args.num_nodes, kernel_size=args.kernel_size)
        self.bn2 = nn.BatchNorm2d(args.num_nodes)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(args.num_nodes*14*14, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class cell_net(nn.Module):

    def __init__(self):
        super().__init__()

        if args.model == 'linear':
            input = [torch.nn.Linear(args.dim ** 2, args.num_nodes), torch.nn.ReLU()]
            layers = [torch.nn.Linear(args.num_nodes, args.num_nodes), torch.nn.ReLU()] * args.num_layers
            output = [torch.nn.Linear(args.num_nodes, NUM_CLASSES)]
            net = input + layers + output
            self.net = torch.nn.Sequential(*net)
        elif 'wide_resnet' in args.model:
            self.net = eval('models.{}(pretrained={})'.format(args.model, args.pretrained))

            num_features = self.net.fc.in_features
            self.net.fc = nn.Linear(num_features, NUM_CLASSES)

        elif 'custom' in args.model:
            self.net = custom_convnet()
        else:
            self.net = make_model(args.model, num_classes=NUM_CLASSES, pretrained=args.pretrained,dropout_p=args.dropout)

    def forward(self, x):
        if args.model == 'linear':
            x = x.reshape(-1, dim*dim)
            return self.net(x)
        elif 'custom' in args.model:
            return self.net(x)
        else:
            return self.net(torch.cat([x] * 3, dim=1))


class tab_cell_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(7, 1000)
        self.l2 = torch.nn.Linear(1000, NUM_CLASSES)

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
        self.samples = self.samples[
            ['Detection probability', 'Nucleus: Area µm^2', 'Nucleus: Length µm', 'Nucleus: Circularity',
             'Nucleus: Solidity', 'Nucleus: Max diameter µm', 'Nucleus: Min diameter µm']]

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

        target = ALL_CLASS_LIST.index(cls)

        if args.binary_class != '':
            target = 1 if cls == args.binary_class else 0
        elif args.combined:
            target = 1 if target > 0 else 0
        elif args.im_only:
            target -=1

        return img, target, id


################################################ TRAINING


def run_epochs(net, train_loader, eval_loader, num_epochs, path, save_freq=100, train=True, lv=0,
               esc=args.early_stopping):
    lr = args.lr
    optimizer = optim.Adam(net.parameters(), lr=lr)
    best_val_correct = lv
    es_countdown = esc

    for epoch in range(args.start_epoch, args.start_epoch + num_epochs):
        total_loss = 0
        total_correct = 0
        scores_list = ['prec', 'rec', 'f1', 'acc']
        total_scores = np.zeros((NUM_CLASSES, len(scores_list)))

        if args.lr_decay and epoch > 0:
            print('Decaying learning rate...')
            lr = args.lr / (epoch + 1)
            optimizer = optim.Adam(net.parameters(), lr=lr)

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

            if args.slide_losses:
                slide_losses = dict(zip(['{}/{}'.format(mode, s) for s in ALL_SLIDES], [0] * len(ALL_SLIDES)))
                for s in range(inputs.shape[0]):
                    s_id = slide_ids[s]
                    s_id_count = slide_ids.count(s_id)
                    slide_loss = eval(args.lf)(outputs[s].detach().unsqueeze(0), targets[s].detach().unsqueeze(0)).cpu()
                    slide_losses['{}/{}'.format(mode, s_id)] += (slide_loss / s_id_count)

            outputs = torch.softmax(outputs, dim=1)

            batch_scores = scores(outputs, targets)
            total_scores += batch_scores

            correct = torch.sum(torch.argmax(outputs, dim=1) == targets)
            total_correct += correct / outputs.shape[0] / len(dataloader)

            print('{} ({}) Epoch: {}/{} Correct: {}/{} Batch: {}/{} Batch Loss {} LR {}'.format(params_id, mode,
                                                                                                epoch + 1, num_epochs,
                                                                                                correct,
                                                                                                outputs.shape[0], i,
                                                                                                len(dataloader),
                                                                                                loss.item(), lr))
            print(batch_scores)

            results = {
                '{}/loss'.format(mode)   : loss.item(), '{}/mean_loss'.format(mode): total_loss / (i + 1),
                '{}/lr'.format(mode)     : lr, '{}/batch'.format(mode): i + 1, '{}/epoch'.format(mode): epoch + 1,
                '{}/correct'.format(mode): correct / outputs.shape[0]
                }

            class_correct = {}
            for c in range(len(CLASS_LIST)):
                c_mask = (targets == c)
                class_correct['{}/'.format(mode) + CLASS_LIST[c] + '_correct'] = torch.sum(
                        torch.argmax(outputs, dim=1)[c_mask] == targets[c_mask]) / torch.sum(c_mask)

            results.update(class_correct)
            if args.slide_losses:
                results.update(slide_losses)

            for s in range(len(scores_list)):
                results.update({
                    '{}/avg_{}'.format(mode, scores_list[s]): (np.sum(total_scores[:, s]) / NUM_CLASSES) / (i + 1)
                    })

            for cls_num in range(NUM_CLASSES):
                results.update(dict(zip(['{}/{}/'.format(mode, CLASS_LIST[cls_num]) + s for s in scores_list],
                                        total_scores[cls_num] / (i + 1))))

            wandb.log(results)

            if (i + 1) % save_freq == 0 and not args.tabular:
                results["{}/imgs".format(mode)] = [wandb.Image(inputs[b].detach().cpu(),
                                                               caption='pred: {}\ntrue: {}'.format(
                                                                       CLASS_LIST[torch.argmax(outputs, dim=1)[b]],
                                                                       CLASS_LIST[targets[b]])) for b in
                                                   range(min(outputs.shape[0], args.num_wb_img))]
                results["{}/confusion".format(mode)] = wandb.plot.confusion_matrix(
                        preds=torch.argmax(outputs.detach(), dim=1).cpu().reshape(-1).numpy(),
                        y_true=targets.detach().cpu().reshape(-1).numpy(), class_names=CLASS_LIST)

                if args.hipe:
                    for cls_im in range(1, len(CLASS_LIST)):
                        hbs = []
                        for hb in range(args.batch_size):
                            if cls_im in targets[hb]:
                                print('HiPe for {}'.format(CLASS_LIST[cls_im]))
                                hipe, hipe_depth, hipe_masks = hierarchical_perturbation(net, inputs[hb].unsqueeze(
                                        0).detach(), target=cls_im, batch_size=32, perturbation_type='fade')
                                hbs.append(wandb.Image(hipe_depth.cpu() * (targets[hb] > 0).cpu()))
                            else:
                                hbs.append(wandb.Image(torch.zeros_like(inputs[hb].unsqueeze(0))))

                        results["{}/{}/hipe".format(mode, CLASS_LIST[cls_im])] = hbs

                wandb.log(results)

        if train:
            with torch.no_grad():
                best_val_correct, es_countdown = run_epochs(net, None, eval_loader, 1, path, train=False,
                                                            save_freq=args.save_freq, lv=best_val_correct,
                                                            esc=es_countdown)
            net.train()

        else:
            print('Last num correct: {} Current num correct: {}'.format(best_val_correct, total_correct))
            'Early stopping countdown: {}'.format(es_countdown)
            wandb.log({'es_countdown': es_countdown, 'val/total_num_correct': total_correct})
            if total_correct < best_val_correct:
                es_countdown -= 1
                if es_countdown == 0:
                    print('Early stopping - val correct rate did not improve after {} epochs'.format(
                        args.early_stopping))
                    exit()
            else:
                es_countdown = args.early_stopping
                best_val_correct = total_correct
                wandb.log({'val/best_correct': best_val_correct})

                print('Saving model...')
                torch.save(net.state_dict(), path)

    return best_val_correct, es_countdown


if args.tabular:
    samples = pd.read_csv("cell_detections.csv").sample(frac=1)

else:
    samples = glob.glob("LC38_CELLS/*/*.tif")
    random.shuffle(samples)

if args.subset < 1:
    samples = samples[:int(len(samples) * args.subset)]

if args.im_only:
    samples = [s for s in samples if 'OTHER' not in s]

train_samples = [s for s in samples if s.split('/')[1].split('_')[0] in TRAIN_SLIDES]
val_samples = [s for s in samples if s.split('/')[1].split('_')[0] in VAL_SLIDES]
test_samples = [s for s in samples if s.split('/')[1].split('_')[0] in TEST_SLIDES]

if args.tabular:
    train_dataset = tab_cell_data(train_samples)
    val_dataset = tab_cell_data(val_samples)
    test_dataset = tab_cell_data(test_samples)
else:
    train_dataset = cell_data(train_samples)
    val_dataset = cell_data(val_samples)
    test_dataset = cell_data(test_samples)

wandb.log({'train samples': len(train_dataset), 'val samples': len(val_dataset), 'test_samples': len(test_dataset)})

print('{} training samples, {} validation samples, {} test samples...'.format(len(train_dataset), len(val_dataset),
                                                                              len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                          worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                          worker_init_fn=np.random.seed(0), num_workers=0, drop_last=False)

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
    _, _ = run_epochs(net, train_loader, eval_loader, num_epochs, 'params/' + params_id + '.pth',
                      save_freq=args.save_freq)

elif args.test:
    with torch.no_grad():
        _, _ = run_epochs(net, None, test_loader, 1, None, train=False, save_freq=args.save_freq)

else:
    with torch.no_grad():
        _, _ = run_epochs(net, None, eval_loader, 1, None, train=False, save_freq=args.save_freq)

run.finish()
exit()
