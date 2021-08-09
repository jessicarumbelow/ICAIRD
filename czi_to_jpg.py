"""
Generates jpg patches from .czi format whole slide images.
Run python czi_to_jpg.py --help to see possible arguments.

Requires javabridge, bioformats and PIL.

Script by Jessica Cooper (jmc31@st-andrews.ac.uk)
"""

import argparse
import glob

import bioformats
import javabridge
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--patch_dim', type=int, default=256, help='Patch dimension. Default is 512.')
parser.add_argument('--overlap', type=int, default=0, help='By how many pixels patches should overlap. Default is 0.')
parser.add_argument('--series', type=int, default=2, help='Czi series/zoom level. Lower number = higher resolution. Typically runs from 1 to ~7. Lower numbers may cause memory issues. Default is 2.)')
parser.add_argument('--czi_dir', default='imgs/czis', help='Location of czi files. Default is "imgs/czis"')
parser.add_argument('--patch_dir', default='imgs/patches', help='Where to save generated patches. Default is "imgs/patches"')
parser.add_argument('--jpg_dir', default='imgs/jpgs', help='Where to save entire slide jpgs. Default is "imgs/jpgs"')
parser.add_argument('--save_blank', action='store_true', help='Whether or not to save blank patches (i.e. no pixel variation whatsoever, such as at edge of slides)')
parser.add_argument('--no_patch', action='store_true', help='If you just want a jpg of the czi file without patches, use this. I suggest you set the series value high to avoid creating a giant '
                                                            'monster jpg.')
parser.add_argument('--resize', default="0,0", help='Optionally provide desired image dimensions separated by a comma, e.g. h,w, which will be used to size the entire slide jpg (with '
                                                    'rotation '
                                                    'and padding if '
                                                    'necessary). If not provided, the jpg dimensions will reflect the czi dimensions and series.')

args = parser.parse_args()

javabridge.start_vm(class_path=bioformats.JARS, max_heap_size="2G")

PATCH_DIM = args.patch_dim
SERIES = args.series

im_names = glob.glob("{}/*.czi".format(args.czi_dir))


def normalise(x):
    if x.max() - x.min() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min())


def resize(im, target_height, target_width):
    print('Image dimensions:', im.size)
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

generated_patches = []

for i in im_names:
    with bioformats.ImageReader(i) as reader:
        reader.rdr.setSeries(SERIES)
        x_dim, y_dim = reader.rdr.getSizeX(), reader.rdr.getSizeY()

        if not args.no_patch:
            print('Generating patches...')
            for x in range(0, x_dim - PATCH_DIM, PATCH_DIM - args.overlap):
                for y in range(0, y_dim - PATCH_DIM, PATCH_DIM - args.overlap):
                    patch = normalise(reader.read(XYWH=(x, y, PATCH_DIM, PATCH_DIM))) * 255

                    if (patch.max() - patch.min() > 0) or args.save_blank:
                        patch_name = '{}/{}_{}_{}_{}.jpg'.format(args.patch_dir, i.split('/')[-1].split('.')[0], SERIES, x, y)
                        patch = Image.fromarray(patch.astype('uint8'))
                        patch.save(patch_name)
                        generated_patches.append(patch_name)
                        print(patch_name)

    try:
        img = normalise(bioformats.load_image(i, series=SERIES)) * 255
        img = Image.fromarray(img.astype('uint8'))
        img.save('{}/{}_{}.jpg'.format(args.jpg_dir, i.split('/')[-1], SERIES))
        if args.resize != "0,0":
            img = resize(img, int(args.resize.split(',')[0]), int(args.resize.split(',')[1]))
            img.save('{}/{}_{}_RESIZED.jpg'.format(args.jpg_dir, i.split('/')[-1], SERIES))
    except Exception:
        print("Could not save entire czi as jpg - it's probably too big. Try using a higher series value.")

javabridge.kill_vm()
print(generated_patches)
print('Successfully generated {} patches!'.format(len(generated_patches)))
