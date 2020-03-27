import argparse
import datetime
import gc
import glob
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from skimage import color, exposure, io, transform
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import utils

from models import backbone, spatial_pyramid_pool
from utils import logger
from utils import REAL_PATH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='store_true', help='Force run everything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--ratio', type=float, default=0.25, help='Split ratio of train/validation')
    parser.add_argument('--image-size', type=int, nargs='+', default=[224, 224], help='Resize image size')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--network', type=str, default='sppnet', help='Network')
    parser.add_argument('--spp', action='store_true', help='Apply Spatial Pyramid Pooling')
    parser.add_argument('--result-log', type=str, default='default.rank', help='Log file name')
    parser.add_argument('--train', action='store_true', help='Train and evaluate BoVW model')
    parser.add_argument('--show', action='store_true', help='Show accuracy in the log file')
    parser.add_argument('--top-n', type=int, default=10, help='Show top N accuracy')
    args = parser.parse_args()

    return args


@utils.timer
def get_data():
    data_path = os.path.join(REAL_PATH, 'data')
    train_path = os.path.join(data_path, 'traindata')
    test_path = os.path.join(data_path, 'testdata')
    test_label_path = os.path.join(data_path, 'Label2Names.csv')
    
    data = {}

    data['train'] = {}

    label_path_list = sorted(glob.glob(os.path.join(train_path, '*')))
    
    for label_path in label_path_list:
        _label_files_path = sorted(glob.glob(os.path.join(label_path, '*')))
        label = label_path.split(os.path.sep)[-1]
        data['train'][label] = sorted(_label_files_path)

    data['label_list'] = sorted(os.listdir(train_path))

    data['test'] = sorted(glob.glob(os.path.join(test_path, '*')))
    data['test_label'] = pd.read_csv(test_label_path, header=None, names=['Category', 'Names'])

    # # Add omitted labels
    omitted_label = [k for k in data['label_list'] if k not in list(data['test_label']['Names'])]
    for i, omitted in enumerate(omitted_label):
        data['test_label'].loc[data['test_label'].index.max()+i+1] = \
            [data['test_label']['Category'].max()+i+1, omitted]

    return data


def load_data(args):
    preprocessed_filename = 'preprocessed_{}_{}.p'.format(args.image_size[0], args.image_size[1])
    preprocessed_path = os.path.join(REAL_PATH, 'result', 'tmp', preprocessed_filename)

    if (not os.path.exists(preprocessed_path)) or (args.force):
        db = build_database(args, preprocessed_path)
    else:
        db = utils.pickle_load(preprocessed_path)

    return db


@utils.timer
def build_database(args, filepath):
    db = {}

    data = get_data()
    db['data'] = data

    db['X_train'] = []
    db['y_train'] = []

    for i, label in enumerate(data['label_list']):
        for f_train in data['train'][label]:
            img = preprocess_img(io.imread(f_train), args.image_size)
            db['X_train'].append(img)
            db['y_train'].append(data['label_list'].index(label))

    utils.safe_pickle_dump(db, filepath)

    return db


def preprocess_img(img, resolution):
    try:
        hsv = color.rgb2hsv(img)
    except:
        rbg_img = color.gray2rgb(img)
        hsv = color.rgb2hsv(rgb_img)
        
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    
    min_side = min(img.shape[:-1])
    center = img.shape[0] // 2, img.shape[1] // 2
    w1 = center[0] - min_side // 2
    w2 = center[0] + min_side // 2
    h1 = center[1] - min_side // 2
    h2 = center[1] + min_side // 2
    img = img[w1:w2, h1:h2, :]
    
    img = transform.resize(img, (resolution[0], resolution[1]))
    
    return img


def get_train_val_data(args, db):
    X = np.array(db['X_train'], dtype='float32')
    y = to_categorical(db['y_train'], len(db['data']['label_list']))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.ratio, random_state=args.seed, shuffle=True)

    return X_train, X_val, y_train, y_val


@utils.timer
def train(args, n_classes, X_train, y_train, X_val, y_val):
    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(3)

    _backbone = backbone(args.network)
    base_model = _backbone.pretrained(input_shape=(args.image_size[0], args.image_size[1], 3))

    for layer in base_model.layers:
        layer.trainable = False

    base_model.summary()
    model = Sequential()
    model.add(base_model)

    if args.spp:
        spp_input_shape = base_model.layers[-1].output_shape
        out_pool_size = [4, 2, 1]
        spp = spatial_pyramid_pool(
            base_model.layers[-1],
            spp_input_shape[0],
            [spp_input_shape[1], spp_input_shape[2]],
            out_pool_size
        )
        model.add(spp)
    else:
        model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.RMSprop(lr=args.lr)
    # optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    net_path = os.path.join(REAL_PATH, 'result', args.network)
    utils.check_path(net_path)

    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST).strftime('%Y%m%d-%H%M%S')
    ckpt_filename = '{}_{}.h5'.format(args.network, str(now))
    ckpt_path = os.path.join(net_path, ckpt_filename)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.epochs//10)
    save_checkpoint = ModelCheckpoint(ckpt_path, monitor='val_accuracy', mode='min', save_best_only=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, save_checkpoint]
    )

    return history, ckpt_path


def make_submission_csv(pred, submit_list, label, csv_path):
    result = []
    for i, p in enumerate(list(pred)):
        res = label[label['Names'].isin([p])]['Category']
        result.append(res.iloc[0])
    df = pd.DataFrame(list(zip(submit_list, result)), columns=['Id', 'Category'])
    df.sort_values(by=['Id'])
    df.to_csv(csv_path, index=False)


def main():
    args = get_args()
    
    result_path = os.path.join(REAL_PATH, 'result')
    utils.check_path(result_path)

    tmp_path = os.path.join(result_path, 'tmp')
    utils.check_path(tmp_path)

    db =  load_data(args)
    X_train, X_val, y_train, y_val = get_train_val_data(args, db)
    train(args, len(db['data']['label_list']), X_train, y_train, X_val, y_val)


if __name__ == '__main__':
    main()
