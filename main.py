import argparse
import itertools


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force-train', action='store_true', help='Force training even if a codebook have')
    parser.add_argument('--image-size', type=int, nargs='+', default=[0, 0], help='Resize image size | (0,0) means that do not resize')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose level of scikit-learn')
    parser.add_argument('--ratio', type=float, default=0.25, help='Split ratio of train / validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--voc-size', type=int, default=200, help='Size of vocabulary')
    parser.add_argument('-C', '--regularization', type=float, default=0.1, help='Regularization parameter in LinearSVC/SVC')
    parser.add_argument('--result-log', type=str, default='parameter_accuracy.log', help='Log file name')
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    voc_size = [200, 400]
    regularization = np.arange(0.0001, 0.0101, 0.0001)

    knob_params = [voc_size, regularization]
    comb_params = list(itertools.product(*knob_params))
    total_comb = len(comb_params)

    prev_voc_size = 0
    print('[ INFO ] Total combination: {}'.format(total_comb))

    for i, params in enumerate(comb_params):
        if prev_voc_size == params[0]:
            args.force_train = False
        else:
            prev_voc_size = params[0]
            args.force_train = True
        args.voc_size = params[0]
        args.regularization = params[1]
        print('Start to train [ {}/{} ]:\n Vocab:\t{}\n Seed:\t{}\n C:\t{}\n'.format(
            i+1, total_comb, args.voc_size, args.seed, args.regularization ))
        run(args)


if __name__ == '__main__':
    main()
