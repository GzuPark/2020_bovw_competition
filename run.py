import argparse
import datetime
import gc
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import utils

from models import backbone
from utils import logger
from utils import REAL_PATH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--ratio', type=float, default=0.25, help='Split ratio of train/validation')
    parser.add_argument('--image-size', type=int, nargs='+', default=[224, 224], help='Resize image size')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--result-log', type=str, default='default.rank', help='Log file name')
    parser.add_argument('--train', action='store_true', help='Train and evaluate BoVW model')
    parser.add_argument('--show', action='store_true', help='Show accuracy in the log file')
    parser.add_argument('--top-n', type=int, default=10, help='Show top N accuracy')
    args = parser.parse_args()

    return args


@utils.timer
def get_data(ratio, seed):
    data_path = os.path.join(REAL_PATH, 'data')
    train_path = os.path.join(data_path, 'traindata')
    test_path = os.path.join(data_path, 'testdata')
    label_path = os.path.join(data_path, 'Label2Names.csv')
    
    data = {}

    data['train'] = []
    data['val'] = []

    _label_path = sorted(glob.glob(os.path.join(train_path, '*')))
    
    for y in _label_path:
        _label_files_path = sorted(glob.glob(os.path.join(y, '*')))

        _train_img_path, _val_img_path = train_test_split(
            _label_files_path,
            test_size=ratio,
            random_state=seed,
            shuffle=True,
        )

        data['train'] += _train_img_path
        data['val'] += _val_img_path

    # data['train'] = sorted(glob.glob(os.path.join(train_path, '*', '*')))
    data['test'] = sorted(glob.glob(os.path.join(test_path, '*')))

    # data['train_label'] = sorted(os.listdir(train_path))
    data['train_str_label'], _label = np.unique(sorted(os.listdir(train_path)), return_inverse=True)
    data['train_cate_label'] = to_categorical(_label, len(_label))
    data['test_label'] = pd.read_csv(label_path, header=None, names=['Category', 'Names'])

    # # Add omitted labels
    omitted_label = [k for k in data['train_str_label'] if k not in list(data['test_label']['Names'])]
    for i, omitted in enumerate(omitted_label):
        data['test_label'].loc[data['test_label'].index.max()+i+1] = \
            [data['test_label']['Category'].max()+i+1, omitted]

    return data


def random_flip(image):
    hflip = np.random.random() > 0.5
    vflip = np.random.random() > 0.5
    
    if hflip and vflip:
        image = cv2.flip(image, -1)
    elif hflip:
        image = cv2.flip(image, -1)
    elif vflip:
        image = cv2.flip(image, 1)
    return image


def normalize_and_add_noise(image):
    image = image.astype(np.float32) / 255. - 0.5
    image += np.random.normal(loc=0, scale=0.1, size=image.shape)
    return image


def make_batch(features, labels):
    x = tf.convert_to_tensor(np.stack(features, axis=0))
    y = tf.convert_to_tensor(np.array(labels, dtype=np.float32)[:, np.newaxis])
    features.clear()
    labels.clear()
    return x, y


# @tf.function
# def read_data(path, label):
#     image = cv2.imread(path)
#     image = cv2.resize(image, (224, 224))
#     image = random_flip(image)
#     image = normalize_and_add_noise(image)

#     return image, label


def data_generator(batch_size, label_path, label_list, cate, img_size, **kwargs):
    epoch_order = np.random.permutation(label_path)
    features, labels = [], []
    for image_path in epoch_order:
        image = cv2.imread(image_path)
        idx = np.where(label_list == image_path.split(os.path.sep)[-2])
        label = cate[idx]

        # Resize to training resolution
        image = cv2.resize(image, (img_size[0], img_size[1]))

        # Randomly horizontal and vertical flip
        image = random_flip(image)

        # Normalize, center, and add Gaussian noise
        image = normalize_and_add_noise(image)
        
        features.append(image)
        labels.append(label)
        if len(features) == batch_size:
            yield make_batch(features, labels)


def train(args):
    data = get_data(args.ratio, args.seed)

    train_gen = data_generator(args.batch, data['train'], data['train_str_label'], data['train_cate_label'], args.image_size)

    n_classes=len(data['train_str_label'])

    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(3)

    base_model = tf.keras.applications.VGG19(
        input_shape=(args.image_size[0], args.image_size[0], 3), 
        include_top=False, 
        weights='imagenet',
    )
    # base_model.summary()

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    # print(model.summary())
    # print(train_gen)
    # print(type(train_gen))

    # img, label = next(iter(train_gen))
    # for i in range(5):
    #     print(img.numpy()[i].shape)
    #     print(label.numpy())


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_gen,
        # batch_size=args.batch,
        epochs=args.epochs,
    )
    acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # val_loss=history.history['val_loss']


    print('\nacc: {}'.format(acc))
    # print('\nval_acc: {}'.format(val_acc))
    print('\nloss: {}'.format(loss))
    # print('\nval_loss: {}'.format(val_loss))


    result_path = os.path.join(REAL_PATH, 'result')
    utils.check_path(result_path)

    tmp_path = os.path.join(result_path, 'tmp')
    utils.check_path(tmp_path)

    # pred = model.predict(test_ds)
    # pred = pred.argmax(axis=-1)
    # print(len(data['train_str_label']))
    # # print(data['test'])
    # # print(len(data['test']))
    # print(pred.shape)
    # # print(pred[0])
    # print(pred)

    # Logging
    # KST = datetime.timezone(datetime.timedelta(hours=9))
    # now = datetime.datetime.now(tz=KST).strftime('%Y%m%d-%H%M%S')

    # log_path = os.path.join(result_path, args.result_log)
    # with open(log_path, 'a') as f:
    #     f.write('{}\t{:.5f}\t{}\n'.format(now, val_acc[-1], args))

    # # Make a csv
    # submit_path = os.path.join(result_path, 'submit')
    # utils.check_path(submit_path)

    # submit_filename = 'submit_{:.5f}_{}.csv'.format(val_acc[-1], now)
    # submit_file_path = os.path.join(submit_path, submit_filename)
    # make_submission_csv(pred, data['train_str_label'], data['test_label'], submit_file_path)


# @utils.timer
# @tf.fuction
# def predict(hist_train, y_train, hist_val, hist_test, seed, C):
#     predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
#     workers=-1, use_multiprocessing=False
# )


def make_submission_csv(pred, submit_list, label, csv_path):
    # print(pred)
    # print(len(pred))
    # print(type(pred))
    result = []
    for i, p in enumerate(list(pred)):
        res = label[label['Names'].isin([p])]['Category']
        result.append(res.iloc[0])
    df = pd.DataFrame(list(zip(submit_list, result)), columns=['Id', 'Category'])
    df.sort_values(by=['Id'])
    df.to_csv(csv_path, index=False)


def main():
    args = get_args()
    # data = get_data(args.ratio, args.seed)
    
    # dd = data['train_label']
    # print(dd)
    # ss, ii = np.unique(dd, return_inverse=True)
    # print(ss)
    # print(ii)
    
    # print()
    # print(data['test_label'])

    # if args.train:
    #     train_grid(args)
    # if args.show:
    #     bovw.find_top_n(args)

    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("Num GPUs:", len(physical_devices))
    
    train(args)
    # _b = 'vgg'
    # b = backbone(_b)
    # model = b.construct(n_classes=10, group=19)
    # assert _b == b.backbone
    # print(model.summary())
    # print(b.backbone)



if __name__ == '__main__':
    main()
