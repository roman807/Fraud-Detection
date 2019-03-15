#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 07:10:40 2019

@author: roman
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
os.chdir('/home/roman/Documents/Projects/Cost_Sensitive')
import ANN
import eval_results
from sklearn.model_selection import KFold

def main():
    # ---------- Prepare data ---------- #
    data = pd.read_csv('data/creditcard.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    amount = data['Amount']
    cost_FP = 3
    cost_FN = amount
    cost_TP = 3
    cost_TN = 0
    cost_mat = np.array([cost_FP * np.ones(data.shape[0]), cost_FN, 
                         cost_TP * np.ones(data.shape[0]), 
                         cost_TN * np.ones(data.shape[0])]).T
    
    # Prepare 5 train / test splits for 5-fold CV:
    n_splits = 5
    kf = KFold(n_splits=n_splits, random_state=123, shuffle=True)
    kf.get_n_splits(X)
    X_train_l, X_test_l = [], []
    y_train_l, y_test_l = [], []
    cost_mat_train_l, cost_mat_test_l = [], []
    for train_index, test_index in kf.split(X):
        X_train_l.append(X[train_index, :])
        X_test_l.append(X[test_index, :])
        y_train_l.append(y.iloc[train_index])
        y_test_l.append(y.iloc[test_index])
        cost_mat_train_l.append(cost_mat[train_index, :])
        cost_mat_test_l.append(cost_mat[test_index, :])
    
    # ---------- Random Model ---------- #
    y_pred_train_rand, y_pred_test_rand = [], []
    print('Random Model ...')
    for y_train, y_test in zip(y_train_l, y_test_l):
        y_pos_train = y_train.sum() / y_train.shape[0]
        y_pred_train_rand.append(np.random.binomial(1, y_pos_train, y_train.shape[0]))
        y_pred_test_rand.append(np.random.binomial(1, y_pos_train, y_test.shape[0]))
        
    # ---------- Logistic Regression ---------- #
    y_pred_train_lr_probas, y_pred_test_lr_probas = [], []
    y_pred_train_lr, y_pred_test_lr = [], []
    for i, (X_train, X_test, y_train) in enumerate(zip(X_train_l, X_test_l, y_train_l)):
        print('Logistic regression ' + str(i + 1) + '/' + str(n_splits) + ' ...')
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred_train_lr_probas.append(np.round(lr.predict_proba(X_train)[:, 1], 3))
        y_pred_test_lr_probas.append(np.round(lr.predict_proba(X_test)[:, 1], 3))
        y_pred_train_lr.append(lr.predict(X_train))
        y_pred_test_lr.append(lr.predict(X_test))

    # ---------- ANN ---------- #
    y_pred_train_ann_probas, y_pred_test_ann_probas = [], []
    y_pred_train_ann, y_pred_test_ann = [], []
    for i, (X_train, X_test, y_train) in enumerate(zip(X_train_l, X_test_l, y_train_l)):
        print('ANN ' + str(i + 1) + '/' + str(n_splits) + ' ...')
        clf = ANN.clf(indput_dim=X_train.shape[1], dropout=0.2)
        clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        clf.fit(X_train, y_train, batch_size=50, epochs=2, verbose=1)
        y_pred_train_ann_proba = np.round(clf.predict(X_train, verbose=1), 3).reshape(-1)
        y_pred_test_ann_proba = np.round(clf.predict(X_test, verbose=1), 3).reshape(-1)
        y_pred_train_ann_probas.append(y_pred_train_ann_proba)
        y_pred_test_ann_probas.append(y_pred_test_ann_proba)
        y_pred_train_ann.append((y_pred_train_ann_proba > 0.5).astype(int).reshape(-1))
        y_pred_test_ann.append((y_pred_test_ann_proba > 0.5).astype(int).reshape(-1))
    
    # ---------- ANN Cost Sensitive---------- #
    y_pred_train_ann_cs_probas, y_pred_test_ann_cs_probas = [], []
    y_pred_train_ann_cs, y_pred_test_ann_cs = [], []
    for i, (X_train, X_test, y_train, cost_mat_train) in enumerate(zip(X_train_l, 
                                       X_test_l, y_train_l, cost_mat_train_l)): 
        print('ANN Cost Sensitive ' + str(i + 1) + '/' + str(n_splits) + ' ...')
        cost_FN_train = cost_mat_train[:, 1]
        y_input = ANN.create_y_input(y_train, cost_FN_train).apply(float)
        clf = ANN.clf(indput_dim=X_train.shape[1], dropout=0.2)
        clf.compile(optimizer='adam', loss=ANN.custom_loss(cost_FP, cost_TP, cost_TN),
                    metrics=['accuracy'])
        clf.fit(X_train, y_input, batch_size=50, epochs=2, verbose=1)
        y_pred_train_ann_cs_proba = clf.predict(X_train, verbose=1)
        y_pred_test_ann_cs_proba = clf.predict(X_test, verbose=1)
        y_pred_train_ann_cs_probas.append(y_pred_train_ann_cs_proba)
        y_pred_test_ann_cs_probas.append(y_pred_test_ann_cs_proba)
        y_pred_train_ann_cs.append((y_pred_train_ann_cs_proba > 0.5).\
                                   astype(int).reshape(-1))
        y_pred_test_ann_cs.append((y_pred_test_ann_cs_proba > 0.5).\
                                  astype(int).reshape(-1))
    
    # Logistic Regression classify according to expected minimum costs (mc)
    y_pred_train_lr_mc, y_pred_test_lr_mc = [], []
    for y_train_proba, y_test_proba, cm_train, cm_test in zip(y_pred_train_lr_probas,\
                                y_pred_test_lr_probas, cost_mat_train_l, cost_mat_test_l):
        cost_0 = (1 - y_train_proba) * cm_train[:, 3] + y_train_proba * cm_train[:, 1]
        cost_1 = (1 - y_train_proba) * cm_train[:, 0] + y_train_proba * cm_train[:, 2]
        y_pred_train_lr_mc.append((cost_1 < cost_0).astype(int))        
        cost_0 = (1 - y_test_proba) * cm_test[:, 3] + y_test_proba * cm_test[:, 1]
        cost_1 = (1 - y_test_proba) * cm_test[:, 0] + y_test_proba * cm_test[:, 2]
        y_pred_test_lr_mc.append((cost_1 < cost_0).astype(int))

    # ANN classify according to expected minimum costs (mc)
    y_pred_train_ann_mc, y_pred_test_ann_mc = [], []
    for y_train_proba, y_test_proba, cm_train, cm_test in zip(y_pred_train_ann_probas,\
                                y_pred_test_ann_probas, cost_mat_train_l, cost_mat_test_l):
        cost_0 = (1 - y_train_proba) * cm_train[:, 3] + y_train_proba * cm_train[:, 1]
        cost_1 = (1 - y_train_proba) * cm_train[:, 0] + y_train_proba * cm_train[:, 2]
        y_pred_train_ann_mc.append((cost_1 < cost_0).astype(int))  
        cost_0 = (1 - y_test_proba) * cm_test[:, 3] + y_test_proba * cm_test[:, 1]
        cost_1 = (1 - y_test_proba) * cm_test[:, 0] + y_test_proba * cm_test[:, 2]
        y_pred_test_ann_mc.append((cost_1 < cost_0).astype(int))
    
    # ---------- Save results ---------- #
    np.save('results/y_pred_train_lr.npy', y_pred_train_lr)
    np.save('results/y_pred_test_lr.npy', y_pred_test_lr)
    np.save('results/y_pred_train_lr_probas.npy', y_pred_train_lr_probas)
    np.save('results/y_pred_test_lr_probas.npy', y_pred_test_lr_probas)
    np.save('results/y_pred_train_ann.npy', y_pred_train_ann)
    np.save('results/y_pred_test_ann.npy', y_pred_test_ann)
    np.save('results/y_pred_train_ann_probas.npy', y_pred_train_ann_probas)
    np.save('results/y_pred_test_ann_probas.npy', y_pred_test_ann_probas)
    np.save('results/y_pred_train_ann_cs.npy', y_pred_train_ann_cs)
    np.save('results/y_pred_test_ann_cs.npy', y_pred_test_ann_cs)
    np.save('results/y_pred_train_ann_cs_probas.npy', y_pred_train_ann_cs_probas)
    np.save('results/y_pred_test_ann_cs_probas.npy', y_pred_test_ann_cs_probas)
    
    # ---------- Evaluate results ---------- #
    eval_results.evaluate('Random', y_train_l, y_test_l, y_pred_train_rand, y_pred_test_rand,
                          cost_mat_train_l, cost_mat_test_l)
    eval_results.evaluate('Logistic Regression', y_train_l, y_test_l, y_pred_train_lr, 
                          y_pred_test_lr, cost_mat_train_l, cost_mat_test_l)
    eval_results.evaluate('ANN', y_train_l, y_test_l, y_pred_train_ann, y_pred_test_ann,
                          cost_mat_train_l, cost_mat_test_l)
    eval_results.evaluate('ANN Cost Sensitive', y_train_l, y_test_l, y_pred_train_ann_cs, 
                          y_pred_test_ann_cs, cost_mat_train_l, cost_mat_test_l)
    eval_results.evaluate('Logistic Regression (min costs)', y_train_l, y_test_l, 
                          y_pred_train_lr_mc, y_pred_test_lr_mc, cost_mat_train_l, 
                          cost_mat_test_l)
    eval_results.evaluate('ANN (min costs)', y_train_l, y_test_l, y_pred_train_ann_mc,
                          y_pred_test_ann_mc, cost_mat_train_l, cost_mat_test_l)

if __name__ == '__main__':
    main()
    
    
    
    
    
