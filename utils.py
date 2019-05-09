import numpy as np
import pandas as pd
import nibabel as nib
import os
import matplotlib.pyplot as plt
import keras
from keras import backend as k

# loss and metrics, handle missing labels, mask for empty value
MASK_VALUE = -1

# TODO: metric mean is calculated incorrectly do to some values being missing
#       i.e. all metrics are scaled by 1/ number_of_samples/number_of_labels

# Fix that does not work when a batch includes only missing labels:
# def masked_mae(y_true, y_pred):
#     mask = k.cast(k.not_equal(y_true, -1), k.floatx())
#     return k.sum(k.abs(y_true * mask - y_pred * mask)) / k.sum(mask)

# def masked_mse(y_true, y_pred):
#     mask = k.cast(k.not_equal(y_true, -1), k.floatx())
#     return k.sum(k.square(y_true * mask - y_pred * mask)) / k.sum(mask)

# def masked_binary_crossentropy(y_true, y_pred):
#     mask = k.cast(k.not_equal(y_true, MASK_VALUE), k.floatx())
#     return k.sum(k.binary_crossentropy(y_true * mask, y_pred * mask)) / k.sum(mask)

# def masked_accuracy(y_true, y_pred, threshold=0.5):
#     threshold = k.cast(threshold, y_pred.dtype)
#     mask = k.cast(k.not_equal(y_true, MASK_VALUE), k.floatx())
#     y_pred = k.cast(y_pred > threshold, y_pred.dtype)
#     return k.sum(k.cast(k.equal(y_true, y_pred), k.floatx())*mask) / k.sum(mask)

def masked_mae(y_true, y_pred):
    mask = k.cast(k.not_equal(y_true, MASK_VALUE), k.floatx())
    return keras.losses.mae(y_true * mask, y_pred * mask)

def masked_mse(y_true, y_pred):
    mask = k.cast(k.not_equal(y_true, MASK_VALUE), k.floatx())
    return keras.losses.mse(y_true * mask, y_pred * mask)

def masked_binary_crossentropy(y_true, y_pred):
    mask = k.cast(k.not_equal(y_true, MASK_VALUE), k.floatx())
    return keras.losses.binary_crossentropy(y_true * mask, y_pred * mask)

def masked_accuracy(y_true, y_pred):
    mask = k.cast(k.not_equal(y_true, MASK_VALUE), k.floatx())
    acc = k.cast(k.equal(y_true, k.round(y_pred)), k.floatx())
    return k.mean(acc*mask, axis=-1)




# Normalize each sample
def sampNorm(X):

    mean = np.mean(X)
    std = np.std(X)

    return (X - mean) / (std + np.finfo(float).eps)


class DataGen:

    def __init__(self, csv,  directory):
        df = pd.read_csv(os.path.join(directory, csv))
        self.dir = directory
        self.filename = np.array(df['file'])

        # missing values should be -1
        self.age = np.array(df['age'])

        self.sex = -np.ones(len(df))
        self.sex[df['sex'] == 'm'] = 0
        self.sex[df['sex'] == 'f'] = 1

        file_path = os.path.join(self.dir, self.filename[0])
        one_brain = np.array(nib.load(file_path).get_data())
        self.shape = one_brain.shape
        self.dtype = one_brain.dtype
        self.total_samples = len(self.filename)
        self.next_index = 0

        self.init()
        
    def __len__(self):
        return self.total_samples

    def init(self):
        rand_indicies = np.random.permutation(np.arange(len(self.filename)))
        self.filename = self.filename[rand_indicies]
        self.age = self.age[rand_indicies]
        self.sex = self.sex[rand_indicies]
        self.next_index = 0

    def samples_left(self):
        return self.total_samples - self.next_index

    def getBatch(self, n_samples):
        n_samples_left = self.samples_left()
        assert n_samples_left > 0, "No samples left, call dataGen.init()"

        n_samples = n_samples if n_samples_left > n_samples else n_samples_left
        data = np.empty((n_samples,)+self.shape+(1,), self.dtype)
        age_label = np.empty(n_samples, self.dtype)
        sex_label = np.empty(n_samples, self.dtype)
        
        for i in range(n_samples):
            file_path = os.path.join(self.dir, self.filename[self.next_index])
            data[i, :, :, :, 0] = sampNorm(nib.load(file_path).get_data())
            age_label[i] = self.age[self.next_index]
            sex_label[i] = self.sex[self.next_index]
            self.next_index += 1

        return data, [age_label, sex_label]

    # generator for keras fit_generator function
    def generator(self, batch_size):
        while True:
            if self.samples_left() == 0:
                self.init()
            if self.samples_left() < batch_size:
                # return rest of dataset
                yield self.getBatch(self.samples_left())
            else:
                yield self.getBatch(batch_size)
                
    # generator for keras predict_generator function
    def predict_generator(self, batch_size=1):
        assert self.samples_left() > 0, "No samples left, call dataGen.init()"
        while self.samples_left() > 0:
            if self.samples_left() < batch_size:
                # return rest of dataset
                yield self.getBatch(self.samples_left())
            else:
                yield self.getBatch(batch_size)
                
    def __call__(self):
        return self.predict_generator()

def kerasPlot(history):
    plt.rcParams['figure.figsize'] = [18, 18]
    fig = plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(3, 2, 3)
    plt.title('Age')
    plt.plot(history.history['age_loss'])
    plt.plot(history.history['val_age_loss'])
    plt.legend(['Training', 'Validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(3, 2, 4)
    plt.title('Sex')
    plt.plot(history.history['sex_loss'])
    plt.plot(history.history['val_sex_loss'])
    plt.legend(['Training', 'Validation'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(3, 2, 5)
    plt.plot(history.history['age_masked_mae'])
    plt.plot(history.history['val_age_masked_mae'])
    plt.legend(['Training', 'Validation'])
    plt.ylabel('MAE')
    plt.xlabel('Epoch')

    plt.subplot(3, 2, 6)
    plt.plot(history.history['sex_masked_accuracy'])
    plt.plot(history.history['val_sex_masked_accuracy'])
    plt.legend(['Training', 'Validation'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    return fig
