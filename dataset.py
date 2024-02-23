from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import tensorflow as tf
import random
import datetime
import math
import cv2
import pandas as pd
import glob

from keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def get_train_imgs():
    # prepare data from dataset markup
    train_markup = 'train_ship_segmentations_v2.csv'
    train_imgs = glob.glob('.\\train_v2\\*.jpg')
    train_annots = glob.glob('.\\annotations\\*.jpg')
    traing_imgs = np.sort(train_imgs)
    train_annots = np.sort(train_annots)
    markup = pd.read_csv(train_markup)

    # find subset of positives (images with ships) and negatives(without ships)

    na_filt = markup.EncodedPixels.isna()
    negative_samples_ind = markup[na_filt].ImageId.index
    positive_samples_ind = np.asarray(list(set(markup.ImageId.index).difference(set(markup[na_filt].ImageId.index))))

    df2 = pd.DataFrame({'path': traing_imgs, 'ind': np.arange(0, traing_imgs.shape[0], 1).astype(int)})

    df3 = pd.DataFrame({'path': traing_imgs, 'ind': np.arange(0, traing_imgs.shape[0], 1).astype(int)},
                       index=df2.path.apply(lambda s: s.split('\\')[-1]).values)

    unique_positive = df3.loc[np.unique(markup.iloc[positive_samples_ind].ImageId.values)]

    unique_negative = df3.loc[np.unique(markup.iloc[negative_samples_ind].ImageId.values)]

    # create lists of files with ships

    traing_imgs_pos = traing_imgs[unique_positive.ind.values]

    train_annots_pos = train_annots[unique_positive.ind.values]

    traing_imgs_neg = traing_imgs[unique_negative.ind.values]

    train_annots_neg = train_annots[unique_negative.ind.values]

    return traing_imgs_pos, train_annots_pos, traing_imgs_neg, train_annots_neg

def create_xy(traing_imgs_pos,train_annots_pos, train_inds, train=True, shape=(768, 768)):

    x_train = np.zeros((train_inds.shape[0], *shape, 3), dtype=np.float16)
    y_train = np.zeros((train_inds.shape[0], *shape), dtype=np.int8)
    for i, ind in enumerate(train_inds):
        #read images
        x_test = cv2.imread(traing_imgs_pos[ind])
        y_test = cv2.imread(train_annots_pos[ind]).mean(axis=-1)
        if train:
            # find x,y bounds of mask
            nonzero_0 = np.arange(y_test.shape[0])[y_test.sum(axis=-1) != 0]
            nonzero_1 = np.arange(y_test.shape[1])[y_test.sum(axis=-2) != 0]

            if nonzero_0.shape[0] > 0 and nonzero_1.shape[0] > 0 and np.random.randint(0, high=50) <= 30:
                # create random stretch of mask
                high_offset, low_offset = np.random.randint(0, high=y_test.shape[0], size=2)
                nonzero_1_offset = np.clip([nonzero_1[[0, -1]] + [-low_offset, high_offset]], 0, y_test.shape[0] - 1)[0]
                nonzero_0_offset = np.clip([nonzero_0[[0, -1]] + [-low_offset, high_offset]], 0, y_test.shape[0] - 1)[0]
                # resize cropped mask at original size
                x_train[i] = cv2.resize(
                    x_test[nonzero_0_offset[0]:nonzero_0_offset[1]][:, nonzero_1_offset[0]:nonzero_1_offset[1]],
                    y_test.shape) / 255
                y_train[i] = cv2.resize(
                    y_test[nonzero_0_offset[0]:nonzero_0_offset[1]][:, nonzero_1_offset[0]:nonzero_1_offset[1]].astype(
                        'float'), y_test.shape).astype('int8')
                y_train[i][y_train[i] <= 0.5] = 0
                y_train[i][y_train[i] > 0.5] = 1
            continue

        y_test[y_test <= 0.5] = 0
        y_test[y_test > 0.5] = 1

        x_train[i] = x_test / 255
        y_train[i] = y_test

    return x_train, y_train

def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):
    """Returns a TF Dataset."""

    def load_img_masks(input_img_path, input_target_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_jpeg(input_img, channels=3)
        #input_img = tf_image.resize(input_img, img_size) # all images are of the same size
        input_img = tf_image.convert_image_dtype(input_img, "float")

        target_img = tf_io.read_file(input_target_path)
        target_img = tf_io.decode_jpeg(target_img, channels=1)
        #target_img = tf_image.resize(target_img, img_size) # all images are of the same size
        target_img = tf_image.convert_image_dtype(target_img, "float")

        return input_img, target_img

    # For faster debugging limit the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)