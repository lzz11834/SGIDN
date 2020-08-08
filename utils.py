import gc
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import random


class train_data():
    def __init__(self, filepath):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath):
    return train_data(filepath=filepath)


def load_images(filelist):
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')


def tf_psnr(im1, im2):
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    psnr = 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))
    return psnr


def cal_psnr(im1, im2):
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def np_psnr(im1, im2):
    mse = (((im1.astype(np.float))*255.0 - (im2.astype(np.float))*255.0) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def np_mpsnr(img1, img2):
    mse = np.zeros(img1.shape[2])
    psnr = np.zeros(img1.shape[2])
    for i in range(img1.shape[2]):
        im1 = img1[:,:,i]
        im2 = img2[:,:,i]
        mse[i]= (((im1.astype(np.float))*255.0 - (im2.astype(np.float))*255.0) ** 2).mean()
        psnr[i] = 10 * np.log10(255 ** 2 / mse[i])
    return np.mean(psnr)


def salt_and_pepper_noise(img, proportion):
    noise_img =img
    height,width =noise_img.shape[0],noise_img.shape[1]
    num = int(height*width*proportion)
    for i in range(num):
        w = random.randint(0,width-1)
        h = random.randint(0,height-1)
        if random.randint(0,1) ==0:
            noise_img[h,w] = 0
        else:
            noise_img[h,w] = 1
    return noise_img