from dataset import get_train_imgs, get_dataset, create_xy
from dataset import DataGenerator
import tensorflow
import math
import numpy as np
import datetime
from keras.optimizers import Adam

def prepare_dataset(len_dataset=1000, batch_size = 4,img_size=(768,768)):
    traing_imgs_pos, train_annots_pos, traing_imgs_neg, train_annots_neg = get_train_imgs()
    train_inds = np.random.choice(range(len(traing_imgs_pos)), len_dataset)

    # create random validation set
    val_inds = np.random.choice(list(set(range(len(traing_imgs_pos))).difference(set(train_inds))), 1000)
    val_set_x = traing_imgs_pos[val_inds]
    val_set_y = traing_imgs_pos[val_inds]

    # create train set with crop/resize preprocessing
    x_train, y_train = create_xy(traing_imgs_pos, train_annots_pos, train_inds)

    train_dataset = DataGenerator(x_train, y_train.astype(np.float16)[:, :, :], batch_size)
    valid_dataset = get_dataset(
        batch_size=batch_size,
        img_size=img_size,
        input_img_paths=val_set_x,
        target_img_paths=val_set_y,
        max_dataset_len=1000
    )
    return train_dataset, valid_dataset

def train_model(model, name, train_dataset, valid_dataset, use_pretrained, checkpoint_dir, batch_size, n_epochs,
                img_size = (768, 768), num_classes = 2):
    n_batches = len(train_dataset) / batch_size
    n_batches = math.ceil(n_batches)
    gpus = tensorflow.config.list_logical_devices('GPU')
    percentage = len(train_dataset)//250000 * 100

    if use_pretrained:
        # Load the previously saved weights
        latest = tensorflow.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)

    log_dir_tb = f"tb_logs/data_{name}_{percentage}%_{n_epochs}epochs/fit/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    model_dir = f"models/data_{name}_{percentage}%_{n_epochs}epochs/fit/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "cp-{epoch:04d}.ckpt"

    # callback for checkpoints

    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        verbose=1,
        save_weights_only=True,
        save_freq=n_batches)

    # callback for tensorboard
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir_tb, histogram_freq=1)

    model.compile(
        optimizer=Adam(1e-2, decay=1e-4), loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['binary_accuracy', 'AUC'], )

    with tensorflow.device(gpus[0].name):
        model.fit(train_dataset,
                  validation_data=valid_dataset,
                  verbose=1,
                  callbacks=[tensorboard_callback, cp_callback],
                  epochs=n_epochs)
    # save weights
    model.save_weights(model_dir.format(epoch=n_epochs))
    return model_dir.format(epoch=n_epochs)