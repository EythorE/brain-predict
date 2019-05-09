#!/usr/bin/python3
# coding: utf-8

import argparse
description = ''' Parameter search using hyperopt's (http://hyperopt.github.io/hyperopt/)
                  implementation of Tree of Parzen Estimators (TPE)'''

parser = argparse.ArgumentParser(description=description)
parser.add_argument('MAX_EVALS', type=int, help='Max evaluations to perform')
parser.add_argument('EPOCHS', type=int, help='Number of epochs per evaluation')
parser.add_argument('OUT_FILE', type=str, help='output file \"test.csv\"')
args = parser.parse_args()

import os
import csv
import numpy as np
from utils import masked_mae, masked_mse, masked_binary_crossentropy, masked_accuracy
from utils import DataGen, kerasPlot # Utilities from utils.py
import keras
import tensorflow as tf
from keras import backend as k
from hyperopt import fmin, tpe, hp

## Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))
np.random.seed(42)
tf.set_random_seed(42)
##

FOLDER = './data'
DATASET = 'train.csv'
VALIDATION = 'validation.csv'
MAX_EVALS = args.MAX_EVALS
EPOCHS = args.EPOCHS
OUT_FILE = args.OUT_FILE

save_plots = EPOCHS>1
if save_plots:
    from utils import kerasPlot
    figure_dir = os.path.splitext(OUT_FILE)[0]+"_plots"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

dg = DataGen(DATASET, FOLDER)
dg_val = DataGen(VALIDATION, FOLDER)
print("Training on:", DATASET, "Validating on:", VALIDATION)
print("Training Samples:", dg.total_samples, ",  Validation Samples:", dg_val.total_samples,)

# Define the search space
space = {
    'optimizer': {'optimizer': 'SGD',
                  'learning_rate': 0.0003,
                  'momentum': 0.3},
    'age_loss': 'masked_mae',
    'sex_loss_weight': 0.7,
    'dropout_rate1': 0.0,
    'dropout_rate2': 0.0,
    'dropout_rate3': 0.0,
    'dropout_rate4': hp.uniform('dropout_rate4', 0.0, 1.0),
    'dropout_rate5': hp.uniform('dropout_rate5', 0.0, 1.0),
    'batch_size': 5
}
metrics={'age':masked_mae, 'sex':masked_accuracy}

from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Activation
from keras.layers import Dropout, Input, BatchNormalization, SpatialDropout3D
from keras.optimizers import Adam, SGD
from keras.models import Model
def Conv3D_model(input_shape, dropout_rate):
    input_layer = Input(input_shape)
    prev_layer = input_layer
    for i in range(4):
        layer = Conv3D(filters=(i+1)*8, kernel_size=(3, 3, 3))(prev_layer)
        layer = SpatialDropout3D(dropout_rate[i])(layer)
        layer = Activation('relu')(layer)
        layer = MaxPool3D(pool_size=(2, 2, 2))(layer)
        layer = BatchNormalization()(layer)
        prev_layer = layer

    flatten_layer = Flatten()(prev_layer)
    flatten_layer = Dropout(dropout_rate[4])(flatten_layer)
    age_layer = Dense(units=1, activation='linear' , name='age')(flatten_layer)
    sex_layer = Dense(units=1, activation='sigmoid', name='sex')(flatten_layer)

    return Model(inputs=input_layer, outputs=[age_layer, sex_layer])

VAL_BATCH_SIZE = 7
iteration = 0
def train(params):
    global iteration
    iteration += 1
    k.clear_session()
    tf.reset_default_graph()
    dg.init()
    dg_val.init()

    ## Extract parameters
    batch_size = int(params['batch_size'])

    dropout_rate=[params['dropout_rate1'],
                  params['dropout_rate2'],
                  params['dropout_rate3'],
                  params['dropout_rate4'],
                  params['dropout_rate5']]

    learning_rate = params['optimizer']['learning_rate']
    if params['optimizer']['optimizer'] == 'Adam':
        optimizer = Adam(lr = learning_rate)
    elif params['optimizer']['optimizer'] == 'SGD':
        optimizer = SGD(lr = learning_rate, momentum=params['optimizer']['momentum'])

    if params['age_loss'] == 'masked_mae':
         loss={'age':masked_mae, 'sex':masked_binary_crossentropy}
    elif params['age_loss'] == 'masked_mse':
         loss={'age':masked_mse, 'sex':masked_binary_crossentropy}

    sex_loss_weight = params['sex_loss_weight']
    loss_weights={'age': 1-sex_loss_weight, 'sex': sex_loss_weight}
    ##

    model = Conv3D_model(dg.shape+(1,), dropout_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    history = model.fit_generator(dg.generator(batch_size),
                                  steps_per_epoch = np.ceil(dg.total_samples/batch_size),
                                  epochs = EPOCHS,
                                  max_queue_size = 10,
                                  validation_data = dg_val.generator(VAL_BATCH_SIZE),
                                  validation_steps = np.ceil(dg_val.total_samples/VAL_BATCH_SIZE),
                                  callbacks = [keras.callbacks.TerminateOnNaN()],
                                  verbose=0)

    if save_plots:
        fig = kerasPlot(history)
        fig.savefig(os.path.join(figure_dir, "plot"+str(iteration)+".png"))

    sex_validation_loss = history.history['val_sex_loss'][-1]

    # Append result history to the csv file
    with open(OUT_FILE, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([sex_validation_loss, params, history.history, iteration])


    return sex_validation_loss

# Save search results to csv
with open(OUT_FILE, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['sex_acc', 'params', 'history', 'iteration'])

# Start hyper-parameter search
best_params = fmin(train, space=space, algo=tpe.suggest, max_evals=MAX_EVALS)
print("Best Parameters found")
print(best_params)
