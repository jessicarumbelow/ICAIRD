"""
Generates jpg patches from .czi format whole slide images.
Run python czi_to_jpg.py --help to see possible arguments.

Requires javabridge, bioformats and PIL.

Script by Jessica Cooper (jmc31@st-andrews.ac.uk)
"""

import argparse
import os

import bioformats
import javabridge
from PIL import Image

from onedrive_access import *

parser = argparse.ArgumentParser()
parser.add_argument('--patch_dim', type=int, default=256, help='Patch dimension. Default is 256.')
parser.add_argument('--overlap', type=int, default=0, help='By how many pixels patches should overlap. Default is 0.')
parser.add_argument('--series', type=int, default=2, help='Czi series/zoom level. Lower number = higher resolution. Typically runs from 1 to ~7. Lower numbers may cause memory issues. Default is 2.)')
parser.add_argument('--czi_dir', default='imgs/czis', help='Location of czi files. Default is "imgs/czis"')
parser.add_argument('--jpg_dir', default='imgs/jpgs', help='Where to save entire slide jpgs. Default is "imgs/jpgs"')
parser.add_argument('--save_blank', action='store_true', help='Whether or not to save blank patches (i.e. no pixel variation whatsoever, such as at edge of slides)')
parser.add_argument('--no_patch', action='store_true', help='If you just want a jpg of the czi file without patches, use this. I suggest you set the series value high to avoid creating a giant '
                                                            'monster jpg.')
parser.add_argument('--resize', default="0,0", help='Optionally provide desired image dimensions separated by a comma, e.g. h,w, which will be used to size the entire slide jpg (with '
                                                    'rotation '
                                                    'and padding if '
                                                    'necessary). If not provided, the jpg dimensions will reflect the czi dimensions and series.')
parser.add_argument('--save_locally', action='store_true', help='Whether or not to save czi file and patches locally when downloaded. Default is False')
parser.add_argument('--force_regen', action='store_true', help='Whether to force re-patching if folder exists. Default is False')

args = parser.parse_args()

PATCH_DIM = args.patch_dim
SERIES = args.series

STORAGE_PATH = '/raid/datasets/LC'


def normalise(x):
    print('Normalising pixel values...')
    return (x - x.min()) / (x.max() - x.min())


def resize(im, target_height, target_width):
    old_width, old_height = im.size
    target_orientation = 'P' if target_height > target_width else 'L'
    old_orientation = 'P' if old_height > old_width else 'L'

    if old_orientation != target_orientation:
        print('Rotating image...')
        im = im.rotate(90, expand=True)
        old_width, old_height = im.size

    print('Resizing image...')
    if target_height == target_width:
        ratio = target_width / max(old_width, old_height)
    else:
        ratio = min(target_width, target_height) / min(old_width, old_height)
    new_width = int(old_width * ratio)
    new_height = int(old_height * ratio)
    im = im.resize((new_width, new_height), resample=Image.LANCZOS)
    new_im = Image.new(im.mode, (target_width, target_height))
    new_im.paste(im, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return new_im


completed = []
czi_folder = my_drive.get_item_by_path('/Paired_DAPI_HE')
drive_czis = list(czi_folder.get_items())
jpgs_folder = my_drive.get_item_by_path('/jpgs')
patches = my_drive.get_item_by_path('/patches')
patched_czis = str(list(patches.get_items()))

if not args.force_regen:
    drive_czis = [f for f in drive_czis if str(f.name).split('.')[0] + '_{}_patches'.format(SERIES) not in patched_czis]
drive_czis = [f for f in drive_czis if str(f.name).split('.')[-1] == 'czi']

czis_todo = len(drive_czis)
print('Found {} CZIs...'.format(czis_todo))
czis_done = 0

generated_patches = []
javabridge.start_vm(class_path=bioformats.JARS)

for file in drive_czis:
    slide_id = file.name.split('.')[0]
    if not os.path.exists(STORAGE_PATH + '/czis/{}'.format(file.name)):
        direct_download(file, STORAGE_PATH + '/czis')
    try:
        img = bioformats.load_image('{}/{}'.format(STORAGE_PATH + '/czis', file.name), series=SERIES)
    except Exception as e:
        print(e)
        print('Retrying...')
        direct_download(file, STORAGE_PATH + '/czis')
        img = bioformats.load_image('{}/{}'.format(STORAGE_PATH + '/czis', file.name), series=SERIES)

    img = normalise(img) * 255

    if not args.save_locally:
        os.remove('{}/{}'.format(STORAGE_PATH + '/czis', file.name))

    completed.append(file.name)
    x_dim, y_dim = img.shape[0], img.shape[1]
    print('Image dimensions: {},{}'.format(x_dim, y_dim))

    if not args.no_patch:
        print('Generating patches...')
        for x in range(0, x_dim - PATCH_DIM, PATCH_DIM):
            for y in range(0, y_dim - PATCH_DIM, PATCH_DIM):
                print(x, y)
                patch = img[x:x + PATCH_DIM, y:y + PATCH_DIM]
                if (patch.max() - patch.min() > 0) or args.save_blank:
                    patch = Image.fromarray(patch.astype('uint8'))
                    if not os.path.exists(STORAGE_PATH + '/patches/{}_{}_patches'.format(slide_id, SERIES)):
                        os.makedirs(STORAGE_PATH + '/patches/{}_{}_patches'.format(slide_id, SERIES))
                    patch_path = STORAGE_PATH + '/patches/{}_{}_patches/{}_{}_{}_{}.jpg'.format(slide_id, SERIES, slide_id, SERIES, x, y)
                    patch.save(patch_path)
                    try:
                        patches_folder = my_drive.get_item_by_path("/patches/{}_{}_patches".format(slide_id, SERIES))
                    except:
                        imgs_folder = my_drive.get_item_by_path("/patches")
                        patches_folder = imgs_folder.create_child_folder("{}_{}_patches".format(slide_id, SERIES))

                    direct_upload(patch_path, patches_folder)
                    if not args.save_locally:
                        os.remove(patch_path)

    img = Image.fromarray(img.astype('uint8'))
    if not os.path.exists(STORAGE_PATH + '/imgs/jpgs'):
        os.makedirs(STORAGE_PATH + '/imgs/jpgs'.format(slide_id))
    img_name = STORAGE_PATH + '/imgs/jpgs/{}_{}.jpg'.format(slide_id, SERIES)
    if args.resize != "0,0":
        img = resize(img, int(args.resize.split(',')[0]), int(args.resize.split(',')[1]))
        resized_name = '{}_RESIZED_{}.jpg'.format(img_name.split('.')[0], args.resize)
        img.save(resized_name)
        direct_upload(resized_name, jpgs_folder)
        if not args.save_locally:
            os.remove(resized_name)
    else:
        img.save(img_name)
        direct_upload(img_name, jpgs_folder)
        if not args.save_locally:
            os.remove(img_name)
    czis_done += 1
    print('Completed {}/{}'.format(czis_done, czis_todo))

print('Completed files:')
[print(p) for p in completed]
javabridge.kill_vm()
