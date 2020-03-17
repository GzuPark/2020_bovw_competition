import argparse
import itertools
import os

import numpy as np
import pandas as pd

from bovw import train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force-train', action='store_true', help='Force training even if a codebook have')
    parser.add_argument('--image-size', type=int, nargs='+', default=[0, 0], help='Resize image size | (0,0) means that do not resize')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose level of scikit-learn')
    parser.add_argument('--ratio', type=float, default=0.25, help='Split ratio of train/validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--voc-size', type=int, default=200, help='Size of vocabulary')
    parser.add_argument('-C', '--regularization', type=float, default=0.1, help='Regularization parameter in LinearSVC')
    parser.add_argument('--result-log', type=str, default='parameter_accuracy.log', help='Log file name')
    parser.add_argument('--train', action='store_true', help='Train and evaluate BoVW model')
    parser.add_argument('--show', action='store_true', help='Show accuracy in the log file')
    parser.add_argument('--top-n', type=int, default=10, help='Show top N accuracy')
    args = parser.parse_args()

    return args


def find_top_n(args):
    real_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(real_path, 'result', args.result_log)

    df = pd.read_csv(log_path, sep='\t', header=None, names=['time', 'acc', 'params'])
    max_index = df['acc'].idxmax()
    result = df.nlargest(args.top_n, 'acc')
    print(result)


def train_grid(args):
    img_size = [[251, 281], [260, 300], [0, 0]]
    voc_size = [200, 400]
    regularization = np.arange(0.0001, 0.0101, 0.0001)

    knob_params = [img_size, voc_size, regularization]
    comb_params = list(itertools.product(*knob_params))
    total_comb = len(comb_params)

    prev_img_size = [1, 1]
    prev_voc_size = 0
    print('[ INFO ] Total combination: {}'.format(total_comb))

    for i, params in enumerate(comb_params):
        if prev_img_size == params[0] and prev_voc_size == params[1]:
            args.force_train = False
        else:
            prev_img_size = params[0]
            prev_voc_size = params[1]
            args.force_train = True
        args.image_size = params[0]
        args.voc_size = params[1]
        args.regularization = params[2]
        print('Start to train [ {}/{} ]:\n Vocab:\t{}\n Seed:\t{}\n C:\t{}\n'.format(
            i+1, total_comb, args.voc_size, args.seed, args.regularization ))
        train(args)


def main():
    args = get_args()
    if args.train:
        train_grid(args)
    if args.show:
        find_top_n(args)


if __name__ == '__main__':
    main()
