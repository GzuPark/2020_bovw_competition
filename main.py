import argparse
import copy
import os

import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force-train', action='store_true', help='Force training even if a codebook have')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose level of scikit-learn')
    parser.add_argument('--ratio', type=float, default=0.25, help='Split ratio of train / validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--voc-size', type=int, default=200, help='Size of vocabulary')
    parser.add_argument('-C', '--regularization', type=float, default=0.1, help='Regularization parameter in LinearSVC')
    args = parser.parse_args()

    return args


@utils.timer
def get_data():
    real_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(real_path, 'data')
    train_path = os.path.join(data_path, 'traindata')
    test_path = os.path.join(data_path, 'testdata')
    
    data = {}

    # ---------- train ----------
    data['train'] = {}
    y_train_list = os.listdir(train_path)
    
    for label in y_train_list:
        if label not in data['train'].keys():
            _y_train_path = os.path.join(train_path, label)
            data['train'][label] = \
                [os.path.join(_y_train_path, img_path) for img_path in os.listdir(_y_train_path)]
    
    # ---------- test ----------
    data['test'] = [os.path.join(test_path, img_path) for img_path in os.listdir(test_path)]

    return data, real_path


@utils.timer
def load_data(data):
    X_train = []
    y_train = []
    X_test = []

    # ---------- train ----------
    for img_label, img_path_list in data['train'].items():
        for img_path in img_path_list:
            X_train.append(cv2.imread(img_path, 0))
            y_train.append(img_label)

    # ---------- test ----------
    for img_path in data['test']:
        X_test.append(cv2.imread(img_path, 0))

    return X_train, y_train, X_test
        

@utils.timer
def extract_features(data):
    descriptors = []
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create()
        _, desc = sift.detectAndCompute(img, None)
        descriptors.append(desc)

    return descriptors


@utils.timer
def train_clustering(descriptors, voc_size, seed, verbose=0):
    features = np.vstack([desc for desc in descriptors])
    kmeans = KMeans(n_clusters=voc_size, n_jobs=-2, random_state=seed, verbose=verbose).fit(features)
    # codebook = kmeans.cluster_centers_.squeeze()

    return kmeans


@utils.timer
def represent_histogram(feature, codebook, minlength):
    train_hist = []
    for feat in feature:
        pred = codebook.predict(feat)
        train_hist.append(np.bincount(pred, minlength=minlength).reshape(1, -1).ravel())

    return np.array(train_hist)


@utils.timer
def predict(hist_train, hist_val, y_train, y_val, seed, C):
    clf = LinearSVC(random_state=seed, C=C, max_iter=3000).fit(hist_train, y_train)
    pred = clf.predict(hist_val)
    score = accuracy_score(y_val, pred)

    return pred, score


def main():
    args = get_args()
    data, real_path = get_data()
    
    train_data, train_label, X_test = load_data(data)

    # Split train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_label, 
        test_size=args.ratio,
        random_state=args.seed,
        shuffle=True,
    )
    print('[ INFO ] Split dataset:\n\tRatio: {}\n\t# of Train: {}\n\t# of Validation: {}'.format(args.ratio, len(y_train), len(y_val)))

    # Extract features
    feature_train = extract_features(X_train)
    feature_val = extract_features(X_val)
    
    # Generate codebook
    result_path = os.path.join(real_path, 'result')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    codebook_filename = 'codebook_{}.p'.format(str(args.voc_size))
    codebook_path = os.path.join(result_path, codebook_filename)

    if (not os.path.exists(codebook_path)) or (args.force_train):
        codebook = train_clustering(feature_train, args.voc_size, args.seed, args.verbose)
        utils.safe_pickle_dump(codebook, codebook_path)
        print('[ INFO ] Complete to train')
    else:
        codebook = utils.pickle_load(codebook_path)
        print('[ INFO ] Complete to load \' {} \''.format(codebook_filename))
    
    # Represent image using codebook
    hist_train = represent_histogram(feature_train, codebook, args.voc_size)
    hist_val = represent_histogram(feature_val, codebook, args.voc_size)

    scaler = StandardScaler().fit(hist_train)
    hist_train = scaler.transform(hist_train)
    hist_val = scaler.transform(hist_val)

    # Predict
    pred, score = predict(hist_train, hist_val, y_train, y_val, args.seed, args.regularization)
    print('[ INFO ] Accuracy: {:.3f} %'.format(score * 100))


if __name__ == '__main__':
    main()
