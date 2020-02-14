#!/usr/bin/env python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error as MAE

import argparse
import ctypes
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import N_FOLD, SEED
from kaggler.data_io import load_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC


def train_predict(train_feature_file, test_feature_file, 
                  predict_valid_file, predict_test_file,
                  C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', 
                  retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_feature_file)
    X_tst, _ = load_data(test_feature_file)

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED).split(X, y)

    p_val = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0]) 
    for i, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}'.format(i))

        logging.info('Training Logistic Regression')
        clf = LogisticRegression(C=C, 
                                 class_weight=class_weight,
                                 max_iter=max_iter, 
                                 solver=solver, 
                                 random_state=SEED)

        clf = clf.fit(X[i_trn], y[i_trn])
        p_val[i_val] = clf.predict_proba(X[i_val])[:, 1]
        logging.info('CV #{}: {:.4f}'.format(i, AUC(y[i_val], p_val[i_val])))

        if not retrain:
            p_tst += clf.predict_proba(X_tst)[:, 1] / N_FOLD

    logging.info('CV: {:.4f}'.format(AUC(y, p_val)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        clf = LogisticRegression(C=C, 
                                 class_weight=class_weight,
                                 max_iter=max_iter, 
                                 solver=solver, 
                                 random_state=SEED)

        clf = clf.fit(X, y)
        p_tst = clf.predict_proba(X_tst)[:, 1]

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--C', type=float, default=1, dest='C')
    parser.add_argument('--regularizer', default='l2', dest='regularizer')
    parser.add_argument('--class_weight', default='balanced', dest='class_weight')
    parser.add_argument('--solver', default='lfbgs', dest='solver')
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_feature_file=args.train_feature_file,
                  test_feature_file=args.test_feature_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  C=args.C,
                  class_weight=args.class_weight, 
                  solver=args.solver, 
                  retrain=args.retrain)
                  
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
