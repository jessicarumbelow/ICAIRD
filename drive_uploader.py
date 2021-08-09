from O365 import Account
from PIL import Image
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_dir', default='./', help='Local directory to upload files from.')
parser.add_argument('--drive_dir', default='Paired_DAPI_HE', help='Drive directory to upload files to.')
args = parser.parse_args()

client_secret = 'uzTrTxL5HxBkD=n]PkBg9SQf4N?Lmn5='

client_id = '291b2960-9a18-4859-8315-6b099b9ee87a'

scopes = ['basic', 'onedrive_all']

credentials = (client_id, client_secret)

account = Account(credentials)

def authenticate():
    print('Authenticating...')
    if not account.is_authenticated:
        account.authenticate(scopes=scopes)
    print('Authenticated...')


def upload(filepath, folder_name):
    try:
        print('Uploading {} to {}...'.format(filepath, folder_name))
        folder_name.upload_file(item=filepath)
    except:
        authenticate()
        upload(filepath, folder_name)


authenticate()

storage = account.storage()

my_drive = storage.get_default_drive()
root_folder = my_drive.get_root_folder()

DRIVE_DIR = args.drive_dir
FILE_DIR = args.file_dir

completed = []
file_drive_folder = list(my_drive.search(DRIVE_DIR))[0]
uploaded_files = [file.name for file in list(file_drive_folder.get_items())]

files_to_upload = [f for f in listdir(FILE_DIR) if (f not in uploaded_files) and (isfile(join(FILE_DIR, f)))]

for file in files_to_upload:
    upload('{}/{}'.format(FILE_DIR, file), file_drive_folder)

