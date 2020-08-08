import argparse
import scipy.io as sio
from model_SGIDN_hybrid import denoiser
import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir',default='./data/checkpoint')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/', help='test sample are saved here')
parser.add_argument('--group_size', dest='group_size', type=int, default=10, help='grouped band numbers')
args = parser.parse_args()


def denoiser_test(denoiser):
    test_files = sio.loadmat(args.test_dir + 'DC.mat')
    test_img = test_files['data'][:,:,0:30]

    test_img = test_img.astype(np.float32)
    [test_h, test_w, test_d] = np.shape(test_img)

    output_clean_image = np.zeros((test_h, test_w, test_d), dtype="float32")

    noisy_files = sio.loadmat(args.test_dir + 'DC_noise.mat')
    noisy_img = noisy_files['data_noise'][:,:,0:30]
    output_clean_image = denoiser.test(test_img, noisy_img, ckpt_dir=args.ckpt_dir)
    output = np.reshape(output_clean_image,
                        [output_clean_image.shape[1], output_clean_image.shape[2], output_clean_image.shape[3]])

    # ori = test_img[:, :, 0]
    # im1 = Image.fromarray(np.uint8((ori - np.max(ori)) / (np.max(ori) - np.min(ori)) * 255))
    # im1.save('./data/result/ori.png')
    #
    # noisy = noisy_img[:, :, 0]
    # im2 = Image.fromarray(np.uint8((noisy - np.max(noisy)) / (np.max(noisy) - np.min(noisy)) * 255))
    # im2.save('./data/result/noisy.png')
    #
    # result = output[:, :, 0]
    # im3 = Image.fromarray(np.uint8((result - np.max(result)) / (np.max(result) - np.min(result)) * 255))
    # im3.save('./data/result/result.png')

    sio.savemat('./data/result/DC_denoised.mat', {'denoised': output})


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if args.use_gpu:
        print("GPU\n")
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_test(model)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess)
            denoiser_test(model)


if __name__ == '__main__':
    tf.app.run()
