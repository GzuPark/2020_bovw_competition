import argparse
import itertools

import numpy as np

import bovw

from utils import logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force-train', action='store_true', help='Force training even if a codebook have')
    parser.add_argument('--config', action='store_true', help='Configuration hyperparameter candidates')
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--ratio', type=float, default=0.25, help='Split ratio of train/validation')
    parser.add_argument('--image-size', type=int, nargs='+', default=[0, 0], help='Resize image size | (0,0) means that do not resize')
    parser.add_argument('--feat-step-size', type=int, default=16, help='Step size when extracting features')
    parser.add_argument('--dense-step-size', type=int, default=2, help='Step size when Dense SIFT')
    parser.add_argument('-L', '--level', type=int, default=2, help='Levels of SPM')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose level of scikit-learn')
    parser.add_argument('--voc-size', type=int, default=200, help='Size of vocabulary')
    parser.add_argument('-C', '--regularization', type=float, default=0.1, help='Regularization parameter in LinearSVC')
    parser.add_argument('--result-log', type=str, default='parameter_accuracy.rank', help='Log file name')
    parser.add_argument('--train', action='store_true', help='Train and evaluate BoVW model')
    parser.add_argument('--show', action='store_true', help='Show accuracy in the log file')
    parser.add_argument('--top-n', type=int, default=10, help='Show top N accuracy')
    args = parser.parse_args()

    return args


def notification(args, cnt, total_comb):
    output = '\n{}\n'.format('-'*40) 
    output += 'Start to train [ {}/{} ]:\n'
    output += ' Seed:\t{}\n'
    output += ' Image:\t{}\n'
    output += ' Vocab:\t{}\n'
    output += ' Fstep:\t{}\n'
    output += ' Dstep:\t{}\n'
    output += ' Level:\t{}\n'
    output += ' C:\t{}\n'
    output = output.format(
        cnt+1, total_comb,
        args.seed,
        args.image_size,
        args.voc_size,
        args.feat_step_size,
        args.dense_step_size,
        args.level,
        args.regularization,
    )
    logger.info(output)


def train_grid(args):
    if args.config:
        import config

        hp = config.Params()
        knob_params = [hp.img_size, hp.voc_size, hp.feat_step_size, hp.dense_step_size, hp.level, hp.regularization]
        comb_params = list(itertools.product(*knob_params))
        total_comb = len(comb_params)

        prev_img_size = [1, 1]
        prev_voc_size = 0
        prev_feat_step_size = 0
        logger.info('Total combination: {}'.format(total_comb))

        for i, params in enumerate(comb_params):
            if prev_img_size == params[0] and prev_voc_size == params[1] and prev_feat_step_size == params[2]:
                args.force_train = False
            else:
                prev_img_size = params[0]
                prev_voc_size = params[1]
                prev_feat_step_size = params[2]
                args.force_train = True
            args.image_size = params[0]
            args.voc_size = params[1]
            args.feat_step_size = params[2]
            args.dense_step_size = params[3]
            args.level = params[4]
            args.regularization = params[5]

            notification(args, i, total_comb)
            bovw.train(args)
    else:
        args.force_train = True
        notification(args, 0, 1)
        bovw.train(args)


def main():
    args = get_args()
    if args.train:
        train_grid(args)
    if args.show:
        bovw.find_top_n(args)


if __name__ == '__main__':
    main()
