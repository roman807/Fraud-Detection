#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 07:10:40 2019

@author: roman
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
os.chdir('/home/roman/Documents/Projects/Cost_Sensitive')
import ANN
import eval_results

# ---------- Prepare data ---------- #
data = pd.read_csv('data/creditcard.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
amount = data['Amount']
cost_FP = 3
cost_FN = amount
cost_TP = 3
cost_TN = 0
cost_mat = np.array([cost_FP * np.ones(data.shape[0]), cost_FN, 
                     cost_TP * np.ones(data.shape[0]), 
                     cost_TN * np.ones(data.shape[0])]).T
sets = train_test_split(X, y, cost_mat, test_size=0.33, random_state=123)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------- Random Model ---------- #
y_pos_train = y_train.sum() / y_train.shape[0]
y_pred_train_rand = np.random.binomial(1, y_pos_train, y_train.shape[0])
y_pred_test_rand = np.random.binomial(1, y_pos_train, y_test.shape[0])

# ---------- Logistic Regression ---------- #
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_train_lr = lr.predict(X_train)
y_pred_test_lr = lr.predict(X_test)

# ---------- ANN ---------- #
clf = ANN.clf(indput_dim=X_train.shape[1], dropout=0.2)
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
clf.fit(X_train, y_train, batch_size=50, epochs=3, verbose=1)
y_pred_train_ann_proba = clf.predict(X_train)
y_pred_test_ann_proba = clf.predict(X_test)
y_pred_train_ann = (y_pred_train_ann_proba > 0.5).astype(int).reshape(-1)
y_pred_test_ann = (y_pred_test_ann_proba > 0.5).astype(int).reshape(-1)

# ---------- ANN Cost Sensitive---------- #
cost_FN_train = cost_mat_train[:, 1]
y_input = ANN.create_y_input(y_train, cost_FN_train).apply(float)
clf = ANN.clf(indput_dim=X_train.shape[1], dropout=0.2)
clf.compile(optimizer='adam', loss=ANN.custom_loss(cost_FP, cost_TP, cost_TN),
            metrics=['accuracy'])
clf.fit(X_train, y_input, batch_size=50, epochs=3, verbose=1)
y_pred_train_ann_cs_proba = clf.predict(X_train)
y_pred_test_ann_cs_proba = clf.predict(X_test)
y_pred_train_ann_cs = (y_pred_train_ann_cs_proba > 0.5).astype(int).reshape(-1)
y_pred_test_ann_cs = (y_pred_test_ann_cs_proba > 0.5).astype(int).reshape(-1)

# ---------- Evaluate results ---------- #
eval_results.evaluate('Random', y_train, y_test, y_pred_train_rand, y_pred_test_rand,
                      cost_mat_train, cost_mat_test)
eval_results.evaluate('Logistic Regression', y_train, y_test, y_pred_train_lr, 
                      y_pred_test_lr, cost_mat_train, cost_mat_test)
eval_results.evaluate('ANN', y_train, y_test, y_pred_train_ann, y_pred_test_ann,
                      cost_mat_train, cost_mat_test)
eval_results.evaluate('ANN Cost Sensitive', y_train, y_test, y_pred_train_ann_cs, 
                      y_pred_test_ann_cs, cost_mat_train, cost_mat_test)

