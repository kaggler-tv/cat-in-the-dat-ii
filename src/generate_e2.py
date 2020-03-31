#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data, save_data
from sklearn import metrics, preprocessing


# ref: # https://www.kaggle.com/abhishek/same-old-entity-embeddings
def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    train = pd.read_csv(train_file, index_col='id')
    test = pd.read_csv(test_file, index_col='id')

    y_trn = train['target']
    del train['target']    
    N_train = len(train)
    N_test = len(test)
    train_test = train.append(test)
    logging.info('trn_shape:{}, tst_shape: {}, all shape: {}'.format(train.shape, test.shape, train_test.shape))

    logging.info('Create features for entity embedding: fill in missing with mode')

    features = [x for x in train.columns if x not in ["id"]]
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        fillin_val = train_test[feat].mode()
        train_test[feat] = lbl_enc.fit_transform(train_test[feat].fillna(fillin_val).astype(str).values)

    train = train_test[:N_train].reset_index(drop=True)
    test = train_test[N_train:].reset_index(drop=True)

    assert((test.loc[:, features].values.shape[1]) == 23)

    test_data = [test.loc[:, features].values[:, k] for k in range(test.loc[:, features].values.shape[1])]

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(train.columns):
            if col != 'id':
                f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    save_data(train, y_trn, train_feature_file)
    save_data(test, None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    
    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_map_file)

    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

