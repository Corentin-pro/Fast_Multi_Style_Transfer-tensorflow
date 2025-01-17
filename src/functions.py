import os
import numpy as np
from glob import glob

import numpy as np
from PIL import Image

def get_image(img_path, size=None):
    img = Image.open(img_path).convert('RGB')
    if size:
        img = img.resize((size, size))
    img = np.asarray(img)
    h, w, c = np.shape(img)

    img = img[:h, :w, ::-1]  # rgb to bgr
    return img


def inverse_image(img):
    img[img > 255] = 255
    img[img < 0] = 0
    img = img[:, :, ::-1]  # bgr to rgb
    return img


def make_project_dir(project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir,'models'))
        os.makedirs(os.path.join(project_dir,'train_result'))
        os.makedirs(os.path.join(project_dir,'test_result'))


def data_loader(dataset):
    print('images Load ....')
    data_path = dataset

    if os.path.exists(data_path + '.npy'):
        data = np.load(data_path + '.npy')
    else:
        data = glob(os.path.join(data_path, "*.*"))
        np.save(data_path + '.npy', data)
    print('images Load Done')

    return data
