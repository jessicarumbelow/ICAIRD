import numpy as np
import timeit
import scipy
from scipy.sparse import random
import scipy.ndimage

import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=1, linewidth=150, suppress=True)


def feature_dims(n, max_x, max_y):
    result = []
    for i in range(1, n + 1):
        div, mod = divmod(n, i)
        if mod == 0 and i <= max_x and div <= max_y:
            result.append((i, div))
    return result


def save_test_im(im, occ_dim, name='_'):
    y1, yd, x1, xd = occ_dim
    mask = np.ones_like(im)
    mask[y1:y1 + yd, x1:x1 + xd] = np.zeros(1)
    im = (im + 10) * mask
    fig = plt.figure('Occlusion')
    plt.clf()
    plt.imshow(im + 20, cmap="gray")
    ax = fig.add_subplot(111)
    plt.imshow(mask, cmap="gray", alpha=0.5)
    plt.axis('off')
    plt.savefig('imgs/occlusion_examples/occlusion_{}'.format(name))


def calculate_binary_occlusions(old_xd, old_yd, fxd, fx, fyd, fy):
    num_occs = 0
    x, y = 0, 0
    xd = 2 ** int(np.ceil(np.log2(old_xd)))
    yd = 2 ** int(np.ceil(np.log2(old_yd)))
    branches = [(xd, x, yd, y)]
    while len(branches) > 0:
        xd, x, yd, y = branches.pop(0)
        if x < old_xd and y < old_yd:
            f = set(range(int(x), int(x + xd))).intersection(set(range(int(fx), int(fx + fxd)))) and set(range(int(y), int(y + yd))).intersection(set(range(int(fy), int(fy + fyd))))
            num_occs += 1
            x2, y2 = x, y
            if xd > yd:
                xd = xd // 2
                x2 = x + xd
            else:
                yd = yd // 2
                y2 = y + yd
            if xd * yd >= 1 and f:
                branches.extend([(xd, x, yd, y), (xd, x2, yd, y2)])
    return num_occs



def binary_occlude_nr(original_input, model, threshold=0.0):
    unoccluded_output = model(original_input)
    input_y_dim, input_x_dim = original_input.shape
    y_ix, x_ix = 0, 0
    y_dim = 2 ** int(np.ceil(np.log2(input_y_dim)))
    x_dim = 2 ** int(np.ceil(np.log2(input_x_dim)))
    input = np.zeros((y_dim, x_dim))
    input[:input_y_dim, :input_x_dim] = original_input
    saliency_map = np.zeros((y_dim, x_dim))
    branches = [(x_dim, x_ix, y_dim, y_ix)]
    num_occs = 0
    while len(branches) > 0:
        x_dim, x_ix, y_dim, y_ix = branches.pop()
        if x_ix < input_x_dim and y_ix < input_y_dim:
            occluded_input = input.copy()
            occluded_input[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] = np.zeros(1)
            output_difference = abs(unoccluded_output - model(occluded_input[:input_y_dim, :input_x_dim]))
            num_occs+=1

            if output_difference > threshold:
                saliency_map[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] += (output_difference / (y_dim + x_dim))

                next_x_ix, next_y_ix = x_ix, y_ix
                if x_dim > y_dim:
                    x_dim = x_dim // 2
                    next_x_ix = x_ix + x_dim
                else:
                    y_dim = y_dim // 2
                    next_y_ix = y_ix + y_dim

                if x_dim * y_dim >= 1:
                    branches.extend([(x_dim, x_ix, y_dim, y_ix), (x_dim, next_x_ix, y_dim, next_y_ix)])

            else:
                saliency_map[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] /= (y_dim + x_dim)


    return saliency_map[:input_y_dim, :input_y_dim], num_occs


def binary_occlude(input, input_x_dim, input_y_dim, model, unoccluded_output, branch, saliency_map, threshold=0.0):
    x_dim, x_ix, y_dim, y_ix = branch
    if x_ix < input_x_dim and y_ix < input_y_dim:
        occluded_input = input.copy()
        occluded_input[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] = np.zeros(1)
        salience = abs(unoccluded_output - model(occluded_input[:input_y_dim, :input_x_dim]))

        if salience > threshold:
            saliency_map[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] += (salience / (y_dim + x_dim))
            x_ix_2, y_ix_2 = x_ix, y_ix
            if x_dim > y_dim:
                x_dim = x_dim // 2
                x_ix_2 = x_ix + x_dim
            else:
                y_dim = y_dim // 2
                y_ix_2 = y_ix + y_dim

            if x_dim * y_dim >= 1:
                saliency_map = binary_occlude(input, input_x_dim, input_y_dim, model, unoccluded_output, (x_dim, x_ix, y_dim, y_ix), saliency_map)
                saliency_map = binary_occlude(input, input_x_dim, input_y_dim, model, unoccluded_output, (x_dim, x_ix_2, y_dim, y_ix_2), saliency_map)
        else:
            saliency_map[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] /= (y_dim + x_dim)

    return saliency_map


def old_occlude(input, model, k_size):
    yd, xd = input.shape
    unocc_pred = model(input)
    x_steps = range(0, xd - k_size + 1)
    y_steps = range(0, yd - k_size + 1)
    heatmap = np.zeros((len(y_steps), len(x_steps)))
    num_occs = 0

    hx = 0
    for x in x_steps:
        hy = 0
        for y in y_steps:
            occ_im = input.copy()
            occ_im[y: y + k_size, x: x + k_size] = np.zeros(1)
            heatmap[hy, hx] = abs(unocc_pred - model(occ_im))
            num_occs += 1
            hy += 1
        hx += 1

    z = xd / len(x_steps)
    return scipy.ndimage.zoom(heatmap, z, order=0), num_occs


