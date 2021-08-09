import numpy as np
import torch
import scipy.ndimage
import pandas as pd
from binary_occlusion import *
from datasets import *


torch.set_printoptions(precision=1, linewidth=150)
np.set_printoptions(precision=1, linewidth=150, suppress=True)

import matplotlib.pyplot as plt


def normalise(x):
    return (x - x.min()) / (x.max() - x.min())


if __name__ == '__main__':

    run_id = '1592307211'
    model = torch.load('models/' + run_id)
    model.eval()

    dataset = MNIST()

    results = pd.DataFrame()


    def get_model_output(x, im_dim=dataset.im_dim):
        x = x[:im_dim[1], :im_dim[2]]
        x = torch.from_numpy(x.reshape(1, 1, x.shape[0], x.shape[1])).to('cpu', dtype=torch.float)
        out = torch.sum(torch.abs(model(x))).item()
        return out


    def sum_abs(im, hm):
        return np.sum(abs(im - hm))

    def mean_abs(im, hm):
        return np.mean(abs(im - hm))

    def mse(im, hm):
        return np.mean((im - hm)**2)

    samples = []
    vals = []

    while len(samples) < 10:
        for d in dataset:
            if d['Y'] not in vals:
                vals.append(d['Y'])
                samples.append(d)

    num_samples = 1000
    random_samples = np.random.random_integers(0, len(dataset), num_samples)

    kernels = [1, 2, 4]

    count = 0
    # for data in samples:
    for ix in random_samples:

        print('{}/{}'.format(count + 1, num_samples))
        data = dataset[ix]
        im = data['X']
        y = data['Y']
        im = im.reshape(dataset.im_dim[1], dataset.im_dim[2])


        model_out = get_model_output(im)

        hm_n, n_occs = binary_occlude_nr(im, get_model_output)


        results = results.append({'ALGORITHM': 'NEW', 'IM': y, 'IX':ix, 'K_SIZE': 1, 'NUM_OCCS': n_occs, 'SAMPLE':im, 'HM':hm_n},
                                 ignore_index=True)

        for k_size in kernels:

            hm_o, o_occs = old_occlude(im, get_model_output, k_size)

            hm_n = normalise(hm_o)

            results = results.append({'ALGORITHM': 'OLD', 'IM': y, 'IX':ix, 'K_SIZE': k_size, 'NUM_OCCS': o_occs, 'SAMPLE':im, 'HM':hm_o},
                                     ignore_index=True)

        count += 1

    print(results)
    results.to_csv('mnist_quantification_results.csv', index=False, encoding='utf-8')
