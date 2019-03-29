#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 3/28/19

"""
ANN.py
 * Create y_input to pass FN to loss function
 * Define custom loss function for cost-sensitive learning
 * Define ANN
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.backend as K
import pandas as pd

def create_y_input(y_train, c_FN):
    y_str = pd.Series(y_train).reset_index(drop=True).apply(lambda x: str(int(x)))
    c_FN_str = pd.Series(c_FN).reset_index(drop=True).apply(lambda x: '0' *
                        (5-len(str(int(x)))) + str(int(x)))
    return y_str + '.' + c_FN_str

def custom_loss(c_FP, c_TP, c_TN):
    def loss_function(y_input, y_pred):
        y_true = K.round(y_input)
        c_FN = (y_input - y_true) * 1e5
        eps = 0.0001
        y_pred = K.minimum(1.0 - eps, K.maximum(0.0 + eps, y_pred))
        cost = y_true * (K.log(y_pred) * c_FN + K.log(1 - y_pred) * c_TP)
        cost += (1 - y_true) * (K.log(1 - y_pred) * c_FP + K.log(y_pred) * c_TN)
        return - K.mean(cost, axis=-1)
    return loss_function

def clf(indput_dim, dropout=0.2):
    model = Sequential([
    Dense(units=50, kernel_initializer='uniform', input_dim=indput_dim, activation='relu'),
    Dropout(dropout),
    Dense(units=25, kernel_initializer='uniform', activation='relu'),
    Dropout(dropout),
    Dense(15, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
    ])
    return model