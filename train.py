
import argparse
from model_SGIDN_hybrid import denoiser
from utils import *
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=300)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)
parser.add_argument('--lr', dest='lr', type=float, default=0.001)
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1)
parser.add_argument('--checkpoint_dir', dest='ckpt_dir',
                    default='./data/checkpoint')

parser.add_argument('--eval_set', dest='eval_set', default='set1', help='dataset for eval in training')
args = parser.parse_args()


def denoiser_train(denoiser, lr):
    with load_data(filepath='./data/patches/img_train.npy') as data:
        data = data.astype(np.float32)
    with load_data(filepath='./data/patches/img_eval.npy') as eval_data:
        eval_data = eval_data.astype(np.float32)
        # eval_data = eval_data[:, :, :, 10:20]
        denoiser.train(data, eval_data, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr)

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    lr = args.lr * np.ones([args.epoch])
    if args.use_gpu:
        print("GPU\n")
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_train(model, lr=lr)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess)
            denoiser_train(model, lr=lr)

if __name__ == '__main__':
    tf.app.run()
