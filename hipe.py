"""
Implementation of hierarchical perturbation for saliency map generation adapted from:

Believe The HiPe: Hierarchical Perturbation for Fast and Robust Explanation of Black Box Models
https://arxiv.org/abs/2103.05108

"""

import torch
import numpy as np
from torchray.utils import imsc
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import time
import inspect
from IPython.display import clear_output

"""
https://arxiv.org/abs/2103.05108
"""


def hierarchical_perturbation(model,
                              input,
                              target,
                              vis=False,
                              interp_mode='nearest',
                              resize=None,
                              batch_size=32,
                              perturbation_type='fade',
                              num_cells=4):
    with torch.no_grad():
        # Get device of input (i.e., GPU).
        dev = input.device
        if dev == 'cpu':
            batch_size = 1
        bn, channels, input_y_dim, input_x_dim = input.shape
        dim = min(input_x_dim, input_y_dim)
        total_masks = 0
        depth = 0
        max_depth = int(np.log2(dim / num_cells))
        saliency = torch.zeros((1, 1, input_y_dim, input_x_dim), device=dev)
        depth_saliencies = torch.zeros((max_depth+1, 1, 1, input_y_dim, input_x_dim), device=dev)
        max_batch = batch_size

        output = F.softmax(model(input), dim=1)[:, target]

        if perturbation_type == 'blur':
            pre_b_image = blur(input.clone().cpu()).to(dev)

        while depth < max_depth:

            masks_list = []
            b_list = []
            num_cells *= 2
            depth += 1
            threshold = torch.min(saliency) + ((torch.max(saliency) - torch.min(saliency)) / 2)

            print('Depth: {}, {} x {} Cell'.format(depth, input_y_dim // num_cells, input_x_dim // num_cells))
            print('Threshold: {:.1f}'.format(threshold))
            print('Range {:.1f} to {:.1f}'.format(saliency.min(), saliency.max()))

            y_ixs = range(-1, num_cells)
            x_ixs = range(-1, num_cells)
            x_cell_dim = input_x_dim // num_cells
            y_cell_dim = input_y_dim // num_cells

            pos_masks = 0

            for x in x_ixs:
                for y in y_ixs:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(x + 2, num_cells), min(y + 2, num_cells)
                    pos_masks += 1

                    mask = torch.zeros((1, 1, num_cells, num_cells), device=dev)
                    mask[:, :, y1:y2, x1:x2] = 1.0
                    local_saliency = F.interpolate(mask, (input_y_dim, input_x_dim), mode=interp_mode) * saliency

                    if depth > 1:
                        local_saliency = torch.max(local_saliency)
                    else:
                        local_saliency = 0

                    # If salience of region is greater than the average, generate higher resolution mask
                    if local_saliency >= threshold:

                        masks_list.append(abs(mask - 1))

                        if perturbation_type == 'blur':

                            b_image = input.clone()
                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = pre_b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim]
                            b_list.append(b_image)

                        if perturbation_type == 'mean':
                            b_image = input.clone()
                            mean = torch.mean(b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim],
                                              axis=(-1, -2), keepdims=True)

                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = mean
                            b_list.append(b_image)

            num_masks = len(masks_list)
            print('Selected {}/{} masks at depth {}'.format(num_masks, pos_masks, depth))

            if num_masks == 0:
                depth -= 1
                break
            total_masks += num_masks

            while len(masks_list) > 0:
                m_ix = min(len(masks_list), max_batch)
                if perturbation_type != 'fade':
                    b_imgs = torch.cat(b_list[:m_ix])
                    del b_list[:m_ix]
                masks = torch.cat(masks_list[:m_ix])
                del masks_list[:m_ix]

                # resize low-res masks to input size
                masks = F.interpolate(masks, (input_y_dim, input_x_dim), mode=interp_mode)

                if perturbation_type == 'fade':
                    perturbed_outputs = torch.relu(output - F.softmax(model(input * masks), dim=1)[:, target])
                else:
                    perturbed_outputs = torch.relu(output - F.softmax(model(b_imgs), dim=1)[:, target])

                sal = perturbed_outputs * torch.abs(masks.transpose(0, 1) - 1)
                saliency += torch.sum(sal, dim=(0, 1))
                depth_saliencies[depth] += torch.sum(sal, dim=(0, 1))

                if vis:
                    clear_output(wait=True)
                    print('Saving image...')
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.title('Depth: {}, {} x {} Mask\nThreshold: {:.1f}'.format(depth, num_cells, num_cells, threshold))
                    if perturbation_type == 'fade':
                        imsc((input * masks)[0])
                    else:
                        imsc(b_imgs[0])
                    plt.subplot(1, 2, 2)
                    imsc(saliency[0])
                    plt.show()
                    plt.savefig('data/attribution_benchmarks/preview')
                    plt.close()

        print('Used {} masks in total.'.format(total_masks))
        if resize is not None:
            saliency = F.interpolate(saliency, (resize[1], resize[0]), mode=interp_mode)
        return saliency, depth_saliencies, total_masks


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def blur(x, klen=11, ksig=5):
    kern = gkern(klen, ksig)
    return F.conv2d(x, kern, padding=klen // 2)
