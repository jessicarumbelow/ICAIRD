"""
Defines experiments as methods This allows easy chaining of process calls and iteration over hyperparameters etc. Call from command line specifying the name of the experiment to run. Results
logging is handled by run.py.

"""

import argparse
import subprocess
import pandas as pd
import copy
import random

parser = argparse.ArgumentParser()
parser.add_argument("exp", help="Specify which experiment to run")
args = parser.parse_args()


def run_exp(exp_args):
    run_args = ["python3", "run.py"] + exp_args
    try:
        subprocess.call(run_args)
    except Exception as e:
        print(e)
        exit()


def grid_search():
    lrs = [0.1, 0.01, 0.001]
    dropouts = [0.0, 0.3, 0.5]
    depths = [2, 8, 16]
    wfs = [2, 4, 8]
    batch_sizes = [1, 8, 24]
    layer_channels = [8, 16, 24]

    for bs in batch_sizes:
        for lr in lrs:
            for dr in dropouts:
                for d in depths:
                    for w in wfs:
                        for lc in layer_channels:
                            run_exp(["--lc", str(lc), "--batch_size", str(bs), "--lr", str(lr), "--dropout", str(dr), "--depth", str(d), "--wf", str(w), "--random_augment", "True"])


def exps():
    run_exp(["--lc", str(8), "--batch_size", str(1), "--lr", str(0.0001), "--dropout", str(0.0), "--depth", str(2), "--wf", str(2), "--random_augment", "False", "--st", str(10)])
    run_exp(["--lc", str(16), "--batch_size", str(1), "--lr", str(0.0001), "--dropout", str(0.0), "--depth", str(5), "--wf", str(2), "--random_augment", "False", "--st", str(10)])




def get_last_result():
    res = pd.read_csv("results.csv").iloc[-1]
    return res.to_dict()


def simple_search(hps=None, prev_hps=None, ix=0):
    print('Previous hps: {}'.format(prev_hps))
    print('Current hps: {}'.format(hps))

    train_acc_thresh = 0.7

    if not hps:
        hps = {
            'lr':             0.0001,
            'lc':             16,
            'batch_size':     1,
            'dropout':        0.2,
            'depth':          3,
            'wf':             2,
            'random_augment': False,
            }
        prev_hps = hps.copy()

        prev_tc = 999
        prev_vc = 999
        prev_run = 999
    else:
        base = get_last_result()
        prev_tc = base['TRAIN_CORRECT']
        prev_vc = base['VAL_CORRECT']
        prev_run = base['RUN_ID']

    run_exp(["--lr", str(hps['lr']),
             "--lc", str(hps['lc']),
             "--batch_size", str(hps['batch_size']),
             "--dropout", str(hps['dropout']),
             "--depth", str(hps['depth']),
             "--wf", str(hps['wf']),
             "--random_augment", str(hps['random_augment']),
             ])


    base = get_last_result()
    tc = base['TRAIN_CORRECT']
    vc = base['VAL_CORRECT']
    run = base['RUN_ID']

    if run == prev_run:
        print('Something went wrong...')
        exit()

    underfit = {
        'lr':             max(hps['lr'] / 10, 0.00001),
        'dropout':        max(hps['dropout'] - 0.1, 0.0),
        'depth':          hps['depth'] + 1,
        'wf':             hps['wf'] + 1,
        'batch_size':     min(hps['batch_size'] * 2, 96),
        'lc':             hps['lc'] * 2,
        'random_augment': False,
        }

    overfit = {
        'lr':             min(hps['lr'] * 10, 1),
        'dropout':        min(hps['dropout'] + 0.1, 0.5),
        'depth':          max(hps['depth'] - 1, 1),
        'wf':             max(hps['wf'] - 1, 1),
        'batch_size':     max(hps['batch_size'] // 2, 1),
        'lc':             max(hps['lc'] // 2, 1),
        'random_augment': True,
        }

    def get_new_hps(tc, prev_tc, vc, prev_vc, hps, prev_hps, ix):

        hparams = list(hps.keys())
        hp = hparams[ix]
        ix = ix + 1 if ix < len(hparams) - 1 else 0
        improvement = (vc >= prev_vc) and (tc >= train_acc_thresh)

        if not improvement:
            print('No improvement. Rolling back hyperparameters...')
            tc = prev_tc
            vc = prev_vc
            hps = prev_hps

        if tc < train_acc_thresh:
            print('Underfit...')
            adj = {hp: underfit[hp]}
        else:
            adj = {hp: overfit[hp]}

        print('Simple search hyperparameter permutation {}...'.format(adj))

        new_hps = copy.deepcopy(hps)
        new_hps.update(adj)

        if str(new_hps) == str(hps):
            print('Duplicate hyperparameters...')
            new_hps, ix = get_new_hps(tc, prev_tc, vc, prev_vc, hps, prev_hps, ix)

        return new_hps, ix

    new_hps, ix = get_new_hps(tc, prev_tc, vc, prev_vc, hps, prev_hps, ix)

    try:
        simple_search(new_hps, hps, ix)
    except Exception as e:
        print(e)
        exit()


if __name__ == '__main__':
    print('Running experiment {}'.format(args.exp))
    exp = eval(args.exp)
    exp()
