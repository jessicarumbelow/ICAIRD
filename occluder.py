import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import torch
import glob
import os


from PIL import Image

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

device = torch.device("cuda")

KERNEL_SIZES = [96]#,4,16,32]
STEP_SIZE = 1
THRESHOLD = 0.1


def binary_occlude_nr(original_input, model, threshold=0.0):

    unoccluded_output = model(original_input)
    print(unoccluded_output)
    input_y_dim, input_x_dim, channels = original_input.shape
    y_ix, x_ix = 0, 0
    y_dim = 2 ** int(np.ceil(np.log2(input_y_dim)))
    x_dim = 2 ** int(np.ceil(np.log2(input_x_dim)))
    input = np.zeros((y_dim, x_dim, channels))
    input[:input_y_dim, :input_x_dim] = original_input
    saliency_map = np.zeros((y_dim, x_dim))
    branches = [(x_dim, x_ix, y_dim, y_ix)]
    num_occs = 0
    """
    while len(branches) > 0:
        x_dim, x_ix, y_dim, y_ix = branches.pop()
        if x_ix < input_x_dim and y_ix < input_y_dim:
            occluded_input = input.copy()
            occluded_input[y_ix:y_ix + y_dim, x_ix:x_ix + x_dim] = np.zeros(1)

            output_difference = abs(unoccluded_output - model(occluded_input[:input_y_dim, :input_x_dim]))
            num_occs += 1
            #print(output_difference)
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
        """
    return saliency_map[:input_y_dim, :input_y_dim], num_occs


def old_occlude(input, model, k_size=1):
    yd, xd, _ = input.shape
    unocc_pred = model(input)
    x_steps = range(0, xd - k_size + 1, STEP_SIZE)
    y_steps = range(0, yd - k_size + 1, STEP_SIZE)
    heatmap = np.zeros((len(y_steps), len(x_steps)))
    num_occs = 0

    hx = 0
    for x in x_steps:
        hy = 0
        for y in y_steps:
            occ_im = input.copy()
            occ_im[y: y + k_size, x: x + k_size] = np.zeros(1)
            diff = abs(unocc_pred - model(occ_im))
            print(diff)
            heatmap[hy, hx] = diff
            num_occs += 1
            hy += 1
        hx += 1


    z = input.shape[0] / heatmap.shape[0]
    return scipy.ndimage.zoom(heatmap, z, order=0), num_occs



def stitch(img_arrs, img_names):
    print(np.array(img_arrs).shape, img_names)
    img_coords = [tuple([int(i) for i in co.split('.')[0].split('_')[-2:]]) for co in img_names]
    im_dict = dict(zip(img_coords, img_arrs))
    last_dims = sorted(img_coords)[-1]
    shape = img_arrs[0].shape
    img_dim = shape[0]
    if len(shape) > 2:
        channels = shape[-1]
        im = np.zeros((last_dims[0] + img_dim, last_dims[1] + img_dim, channels))
    else:
        im = np.zeros((last_dims[0] + img_dim, last_dims[1] + img_dim))
    for k, v in im_dict.items():
        im[k[0]:k[0] + img_dim, k[1]:k[1] + img_dim] = v
    return im


def normalise(x):

    return (x - np.min(x)) / (np.max(x) - np.min(x))

def run_occlusion(img_dir, run_id, algo='new', save_patch=False):

    model = torch.load('models/' + run_id)
    model.eval()

    img_files = glob.glob(img_dir)
    clean_imgs = normalise(np.asarray([np.asarray(Image.open(i)) for i in img_files]))
    channels = clean_imgs[0].shape[-1]
    img_dim = clean_imgs[0].shape[0]

    def get_model_prediction(x):
        pred = model(torch.from_numpy(np.asarray([x])).to(device, dtype=torch.float)).item()

        return pred

    for KERNEL_SIZE in KERNEL_SIZES:
        print('Kernel size: {}'.format(KERNEL_SIZE))

        heatmaps = []
        img_names = []
        for img_ix in range(len(clean_imgs)):

            img_name = img_files[img_ix].split('/')[-1]
            print('Image: {}'.format(img_name))
            im = clean_imgs[img_ix]


            if algo == 'new':
                heatmap, occs = binary_occlude_nr(im, get_model_prediction, threshold=THRESHOLD)
            else:
                heatmap, occs = old_occlude(im, get_model_prediction, k_size=KERNEL_SIZE)

            heatmaps.append(heatmap)
            img_names.append(img_name)

            if save_patch:
                fig = plt.figure('Heatmap')
                plt.clf()
                if channels == 1:
                    plt.imshow(im, cmap="gray")
                else:
                    plt.imshow(im)

                ax = fig.add_subplot(111)
                ax.imshow(heatmap, cmap=plt.cm.cividis, alpha=0.7)
                plt.axis('off')
                plt.savefig('imgs/HE_L{}_{}_{}_{}_occ_patch{}.png'.format(slide_id, SERIES, KERNEL_SIZE, algo, img_ix))


        heat = normalise(stitch(heatmaps, img_names))
        clean = stitch(clean_imgs, img_names)

        fig = plt.figure('Heatmap')
        plt.clf()
        print('Plotting image...')
        if channels == 1:
            plt.imshow(clean, cmap="gray")
        else:
            plt.imshow(clean)

        ax = fig.add_subplot(111)
        print('Plotting heatmap...')

        ax.imshow(heat, cmap=plt.cm.cividis, alpha=0.7)
        plt.axis('off')
        plt.savefig('imgs/HE_L{}_{}_{}_{}_occ.png'.format(slide_id, SERIES, KERNEL_SIZE, algo))



if __name__ == '__main__':
    slide_ids = ['110']#, '721', '114']


    SERIES = 4
    for slide_id in slide_ids:
        img_dir = '/raid/datasets/LC/tiles/*4.41,42826,29302*'.format(slide_id, SERIES)
        run_occlusion(img_dir, '1597253525', 'new')
        run_occlusion(img_dir, '1597253525', 'old')

