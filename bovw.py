import datetime
import os

import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import utils

from utils import logger


@utils.timer
def get_data(ratio, seed):
    real_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(real_path, 'data')
    train_path = os.path.join(data_path, 'traindata')
    test_path = os.path.join(data_path, 'testdata')
    label_path = os.path.join(data_path, 'Label2Names.csv')
    
    data = {}

    # ---------- train, validation ----------
    data['train'] = {}
    data['validation'] = {}

    y_train_list = os.listdir(train_path)
    
    for label in y_train_list:
        if label not in data['train'].keys():
            _y_train_path = os.path.join(train_path, label)
            _all_file_path = [os.path.join(_y_train_path, img_path) for img_path in os.listdir(_y_train_path)]

            _train_img_path, _val_img_path = train_test_split(
                _all_file_path,
                test_size=ratio,
                random_state=seed,
                shuffle=True,
            )

            data['train'][label] = [os.path.join(_y_train_path, img_path) for img_path in _train_img_path]
            data['validation'][label] = [os.path.join(_y_train_path, img_path) for img_path in _val_img_path]

    # ---------- test ----------
    data['test_list'] = os.listdir(test_path)
    data['test'] = [os.path.join(test_path, img_path) for img_path in data['test_list']]
    data['label'] = pd.read_csv(label_path, header=None, names=['Category', 'Names'])

    # Add omitted labels
    omitted_label = [k for k in data['train'].keys() if k not in list(data['label']['Names'])]
    for i, omitted in enumerate(omitted_label):
        data['label'].loc[data['label'].index.max()+i+1] = [data['label']['Category'].max()+i+1, omitted]

    return data, real_path


@utils.timer
def load_data(data, size):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []

    # ---------- train ----------
    for img_label, img_path_list in data['train'].items():
        for img_path in img_path_list:
            img = cv2.imread(img_path, 0)
            if (size[0] != 0) and (size[1] != 0):
                img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
            X_train.append(cv2.imread(img_path, 0))
            y_train.append(img_label)

    # ---------- validation ----------
    for img_label, img_path_list in data['validation'].items():
        for img_path in img_path_list:
            X_val.append(cv2.imread(img_path, 0))
            y_val.append(img_label)

    # ---------- test ----------
    for img_path in data['test']:
        X_test.append(cv2.imread(img_path, 0))

    return X_train, y_train, X_val, y_val, X_test
        

@utils.timer
def extract_features(data, step_size):
    descriptors = []
    for img in data:
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = []
        for h in range(0, img.shape[0], step_size):
            for w in range(0, img.shape[1], step_size):
                keypoints.append(cv2.KeyPoint(h, w, step_size))
        dense_feature = sift.compute(img, keypoints)
        descriptors.append(dense_feature[1])

    return descriptors


@utils.timer
def train_clustering(descriptors, voc_size, seed, verbose=0):
    features = np.vstack([desc for desc in descriptors])
    kmeans = KMeans(n_clusters=voc_size, n_jobs=-2, random_state=seed, verbose=verbose).fit(features)

    return kmeans


class SPM(object):
    def __init__(self, step_size, L, kmeans, voc_size):
        self.step_size = step_size
        self.L = L
        self.kmeans = kmeans
        self.voc_size = voc_size

    def extract_dense_sift(self, img):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints = []
        for h in range(0, img.shape[0], self.step_size):
            for w in range(0, img.shape[1], self.step_size):
                keypoints.append(cv2.KeyPoint(h, w, self.step_size))
        return sift.compute(img, keypoints)[1]

    def get_features_spm(self, img):
        histogram = []
        for _l in range(self.L+1):
            h_step = int(np.floor(img.shape[0]/(2**_l)))
            w_step = int(np.floor(img.shape[1]/(2**_l)))
            i = 0
            j = 0
            for _ in range(1, 2**_l+1):
                j = 0
                for _ in range(1, 2**_l+1):
                    desc = self.extract_dense_sift(img[i:i+h_step, j:j+w_step])
                    pred = self.kmeans.predict(desc)
                    hist = np.bincount(pred, minlength=self.voc_size).reshape(1, -1).ravel()
                    histogram.append(hist*(2**(_l-self.L)))
                    j += w_step
                i += h_step

        histogram = np.array(histogram).ravel()
        histogram = (histogram - np.mean(histogram)) / np.std(histogram)
        
        return histogram

    @utils.timer
    def get_histogram(self, data):
        histogram = []
        for img in data:
            hist = self.get_features_spm(img)
            histogram.append(hist)
        
        return np.array(histogram)


@utils.timer
def predict(hist_train, y_train, hist_val, hist_test, seed, C):
    clf = LinearSVC(random_state=seed, C=C).fit(hist_train, y_train)
    pred_val = clf.predict(hist_val)
    pred_test = clf.predict(hist_test)

    return pred_val, pred_test


@utils.timer
def get_accuracy(y_test, pred):
    score = accuracy_score(y_test, pred)
    return score


@utils.timer
def train(args):
    # Load and split data
    data, real_path = get_data(args.ratio, args.seed)
    X_train, y_train, X_val, y_val, X_test = load_data(data, args.image_size)
    
    result_path = os.path.join(real_path, 'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    _tmp_path = os.path.join(result_path, 'tmp')
    if not os.path.exists(_tmp_path):
        os.makedirs(_tmp_path)

    # Extract features
    feat_train_filename = 'feat_train_{}_{}.p'.format(str(args.image_size[0]), str(args.image_size[1]))
    feat_train_path = os.path.join(_tmp_path, feat_train_filename)

    if (not os.path.exists(feat_train_path)) or (not os.path.exists(feat_val_path)) or (args.force_train):
        feature_train = extract_features(X_train, args.feat_step_size)
        utils.safe_pickle_dump(feature_train, feat_train_path)
        logger.info('Save caches of features')
    else:
        feature_train = utils.pickle_load(feat_train_path)
        logger.info('Complete to load features')
    
    descriptors = []
    for i in range(len(feature_train)):
        for j in range(feature_train[i].shape[0]):
            descriptors.append(feature_train[i][j,:])
    
    # Generate codebook
    codebook_filename = 'codebook_{}_{}_{}.p'.format(str(args.ratio), str(args.seed), str(args.voc_size))
    codebook_path = os.path.join(_tmp_path, codebook_filename)

    if (not os.path.exists(codebook_path)) or (args.force_train):
        codebook = train_clustering(feature_train, args.voc_size, args.seed, args.verbose)
        utils.safe_pickle_dump(codebook, codebook_path)
        logger.info('Complete to train')
    else:
        codebook = utils.pickle_load(codebook_path)
        logger.info('Complete to load \' {} \''.format(codebook_filename))
    
    # Represent image using codebook
    spm = SPM(args.dense_step_size, args.level, codebook, args.voc_size)
    hist_train = spm.get_histogram(X_train)
    hist_val = spm.get_histogram(X_val)
    hist_test = spm.get_histogram(X_test)

    # Predict
    pred_val, pred_test = predict(hist_train, y_train, hist_val, hist_test, args.seed, args.regularization)
    score = get_accuracy(y_val, pred_val)
    logger.info('Accuracy: {:.3f} %'.format(score * 100))

    # Logging
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST).strftime('%Y-%m-%d-%H-%M-%S')

    log_path = os.path.join(result_path, args.result_log)
    with open(log_path, 'a') as f:
        f.write('{}\t{:.5f}\t{}\n'.format(now, score, args))

    # Make a csv
    submit_path = os.path.join(result_path, 'submit')
    if not os.path.exists(submit_path):
        os.makedirs(submit_path)

    submit_filename = 'submit_{:.5f}_{}.csv'.format(score, now)
    submit_file_path = os.path.join(submit_path, submit_filename)
    make_submission_csv(pred_test, data['test_list'], data['label'], submit_file_path)


def make_submission_csv(pred, submit_list, label, csv_path):
    result = []
    for i, p in enumerate(pred):
        res = label[label['Names'].isin([p])]['Category']
        result.append(res.iloc[0])
    df = pd.DataFrame(list(zip(submit_list, result)), columns=['Id', 'Category'])
    df.sort_values(by=['Id'])
    df.to_csv(csv_path, index=False)


def find_top_n(args):
    real_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(real_path, 'result', args.result_log)

    df = pd.read_csv(log_path, sep='\t', header=None, names=['time', 'acc', 'params'])
    max_index = df['acc'].idxmax()
    result = df.nlargest(args.top_n, 'acc')
    print(result)
