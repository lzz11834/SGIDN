import argparse
import glob
import os
import numpy as np
import h5py



parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/raw', help='training data')
parser.add_argument('--save_dir', dest='save_dir', default='./data/patches', help='save patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=40, help='patch size')
parser.add_argument('--patch_depth', dest='pat_depth', type=int, default=30, help='patch depth')
parser.add_argument('--stride', dest='stride', type=int, default=40, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
args = parser.parse_args()


def get_from_mats(key, mat_list):
    v_list = [np.array(e[key]) for e in mat_list]
    return v_list


def generate_patches(isDebug= False):
    count = 0
    file_paths = glob.glob(args.src_dir + '/*.mat')
    if isDebug:
        file_paths = file_paths[:10]
    print("number of training data %d" % len(file_paths))

    # load mat Data
    mat_list = []
    for i in range(len(file_paths)):
        mat_data = h5py.File(file_paths[i], 'r+')
        mat_list.append(mat_data)
    data_list = get_from_mats('data', mat_list)
    ret = np.stack(data_list, axis=0)
    ret = np.swapaxes(ret, 1, 2)
    ret = np.swapaxes(ret, 2, 3)
    ret = ret[:, :, :, :]
    [im_n, im_h, im_w, im_d] = np.shape(ret)
    print(im_n, im_h, im_w, im_d)
    for i in range(len(file_paths)):
        for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
            for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                count += 1
    print("calculate patches = %d" % count)
    origin_patch_num = count

    if origin_patch_num % args.bat_size != 0:
        num_patches = (origin_patch_num // args.bat_size + 1) * args.bat_size
    else:
        num_patches = origin_patch_num
    print("total patches = %d, batch size = %d, total batches = %d" %
          (num_patches, args.bat_size, num_patches / args.bat_size))
    inputs = np.zeros((num_patches, args.pat_size, args.pat_size, args.pat_depth), dtype="float32")
    count = 0
    ret_s = np.array(ret)
    for i in range(len(file_paths)):
        img_s = ret_s[i, :, :, :]
        for x in range(0 + args.step, im_h - args.pat_size, args.stride):
            for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                inputs[count, :, :, :, ] = img_s[x:x + args.pat_size, y:y + args.pat_size, :]
                for band in range(im_d):
                    inputs[count, :, :, band] = (inputs[count, :, :, band] - np.min(inputs[count, :, :, band])) / (
                            np.max(inputs[count, :, :, band]) - np.min(inputs[count, :, :, band]))
                count += 1

    if count < num_patches:
        to_pad = num_patches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    inputs = inputs[:, :, :, 10:20]
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, "img_train"), inputs)
    print("size of inputs tensor = " + str(inputs.shape))

if __name__ == '__main__':
    generate_patches()
