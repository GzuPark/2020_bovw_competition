import argparse
import datetime
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import utils

from models import backbone
from utils import logger
from utils import REAL_PATH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed number')
    parser.add_argument('--ratio', type=float, default=0.25, help='Split ratio of train/validation')
    parser.add_argument('--image-size', type=int, nargs='+', default=[0, 0], help='Resize image size | (0,0) means that do not resize')
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
    train_label_path = glob.glob(os.path.join(train_path, '*'))
    
    for y in train_label_path:
        _label_files_path = glob.glob(os.path.join(y, '*'))

        _train_img_path, _val_img_path = train_test_split(
            _label_files_path,
            test_size=ratio,
            random_state=seed,
            shuffle=True,
        )

        data['train'] += _train_img_path
        data['val'] += _val_img_path

    # data['train'] = glob.glob(os.path.join(train_path, '*', '*'))
    data['test'] = glob.glob(os.path.join(test_path, '*'))

    data['train_label'] = os.listdir(train_path)
    data['test_label'] = pd.read_csv(label_path, header=None, names=['Category', 'Names'])

    # # Add omitted labels
    omitted_label = [k for k in data['train_label'] if k not in list(data['test_label']['Names'])]
    for i, omitted in enumerate(omitted_label):
        data['test_label'].loc[data['test_label'].index.max()+i+1] = \
            [data['test_label']['Category'].max()+i+1, omitted]

    return data


# @utils.timer
def load_data(data):
    _train_ds = tf.data.Dataset.list_files(data)
    labeled_train_ds = _train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    return labeled_train_ds


def load_test_data(data):
    _test_ds = tf.data.Dataset.list_files(data)
    labeled_test_ds = _test_ds.map(process_test_path)
    return labeled_test_ds


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


@tf.function
def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, IMG_SIZE)


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def process_test_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def prepare_for_training(ds, batch_size=32, suffle_buffer_size=1000):
    ds = ds.shuffle(buffer_size=suffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def show_batch(image_batch, label_batch):
    for n in range(5):
        print(image_batch[n].shape)
        print(CLASS_NAMES[label_batch[n]==1][0].title())


@utils.timer
def train(args):
    data = get_data(args.ratio, args.seed)

    global AUTOTUNE
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    global CLASS_NAMES
    CLASS_NAMES = np.array(data['train_label'])

    global IMG_SIZE
    IMG_SIZE = args.image_size

    labeled_train_ds = load_data(data['train'])
    labeled_val_ds = load_data(data['val'])

    train_ds = prepare_for_training(labeled_train_ds)
    val_ds = prepare_for_training(labeled_val_ds)

    labeled_test_ds = load_test_data(data['test'])
    test_ds = labeled_test_ds.batch(32)

    # image_batch, label_batch = next(iter(train_ds))
    # show_batch(image_batch.numpy(), label_batch.numpy())

    
    # _b = 'vgg'
    _b = 'sppnet'
    b = backbone(_b)
    # model = b.construct(n_classes=len(data['train_label']), group=16)
    model = b.construct(n_classes=len(data['train_label']))
    # model = b.pretrained()
    assert _b == b.backbone
    print(model.summary())
    # print(b.backbone)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=32,
        validation_steps=32,
        epochs=10,
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss=history.history['val_loss']


    print('\nacc: {}'.format(acc))
    print('\nval_acc: {}'.format(val_acc))
    print('\nloss: {}'.format(loss))
    print('\nval_loss: {}'.format(val_loss))


    result_path = os.path.join(REAL_PATH, 'result')
    utils.check_path(result_path)

    tmp_path = os.path.join(result_path, 'tmp')
    utils.check_path(tmp_path)

    pred = model.predict(test_ds)
    pred = pred.argmax(axis=-1)
    print(len(data['train_label']))
    # print(data['test'])
    # print(len(data['test']))
    print(pred.shape)
    # print(pred[0])
    print(pred)

    # Logging
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST).strftime('%Y%m%d-%H%M%S')

    log_path = os.path.join(result_path, args.result_log)
    with open(log_path, 'a') as f:
        f.write('{}\t{:.5f}\t{}\n'.format(now, val_acc[-1], args))

    # Make a csv
    submit_path = os.path.join(result_path, 'submit')
    utils.check_path(submit_path)

    submit_filename = 'submit_{:.5f}_{}.csv'.format(val_acc[-1], now)
    submit_file_path = os.path.join(submit_path, submit_filename)
    make_submission_csv(pred, data['train_label'], data['test_label'], submit_file_path)


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
