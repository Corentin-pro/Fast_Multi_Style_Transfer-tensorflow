import os
import cv2
import numpy as np


def main():
    dataset = 'new_style'
    saveset = 'new_style_crop'
    size = 512

    if not os.path.exists(saveset):
        os.makedirs(saveset)
    image_index = 0
    for fn in os.listdir(dataset):
        img = cv2.imread(dataset + '/' + fn)
        h,w,c = np.shape(img)
        # print(fn, w, h, c)

        if w >= h:
            ratio = float(h)/float(w)
            resize_factor = (int(size/ratio), size)
            img_resize = cv2.resize(img, resize_factor)
        else:
            ratio = float(w)/float(h)
            resize_factor = (size, int(size/ratio))
            img_resize = cv2.resize(img, resize_factor)

        w,h,c = np.shape(img_resize)
        crop_w = int((w-size) * 0.5)
        crop_h = int((h-size) * 0.5)

        img_crop = img_resize[crop_w:crop_w+size,crop_h:crop_h+size,:]
        cv2.imwrite(os.path.join(saveset, '{}-'.format(image_index) + fn), img_crop)
        image_index += 1


if __name__ == '__main__':
    main()
