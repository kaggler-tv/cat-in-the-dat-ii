#!/usr/bin/env python

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC

import argparse
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import N_FOLD, SEED
from kaggler.data_io import load_data
from kaggler.model import AutoLGB

import lightgbm as lgb


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_stop=100, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    model = AutoLGB(objective='binary', metric='auc', n_random_col=0)
    model.tune(pd.DataFrame(X), pd.Series(y))

    params = model.params
    n_est = model.n_best

    logging.info(f'params: {params}')
    logging.info(f'n_best: {n_est}')

    p = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    n_bests = []
    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info('Training model #{}'.format(i))
        trn_lgb = lgb.Dataset(X[i_trn], label=y[i_trn])
        val_lgb = lgb.Dataset(X[i_val], label=y[i_val])

        logging.info('Training with early stopping')
        clf = lgb.train(params, trn_lgb, n_est, val_lgb, early_stopping_rounds=n_stop, verbose_eval=100)
        n_best = clf.best_iteration
        n_bests.append(n_best)
        logging.info('best iteration={}'.format(n_best))

        p[i_val] = clf.predict(X[i_val])
        logging.info('CV #{}: {:.4f}'.format(i, AUC(y[i_val], p[i_val])))

        p_tst += clf.predict(X_tst) / N_FOLD

    logging.info('CV: {:.4f}'.format(AUC(y, p)))
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--early-stop', type=int, dest='n_stop')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_stop=args.n_stop)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
