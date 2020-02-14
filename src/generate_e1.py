#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data, save_data
from kaggler.preprocessing import OneHotEncoder

# ref: https://www.kaggle.com/cuijamm/simple-onehot-logisticregression-score-0-80801
def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file, index_col='id')
    tst = pd.read_csv(test_file, index_col='id')

    y_trn = trn['target']
    del trn['target']
    trn_tst = trn.append(tst)
    n_trn = len(trn)
    logging.info('trn_shape:{}, tst_shape: {}, all shape: {}'.format(trn.shape, tst.shape, trn_tst.shape))

    logging.info('One Hot Encoding categorical variables')
    ohe = trn_tst_ohe = OneHotEncoder(min_obs=1)
    trn_tst_ohe = ohe.fit_transform(trn_tst)
    trn_tst_ohe = trn_tst_ohe.tocsr()
    logging.info(f'shape of all data after OHE: {trn_tst_ohe.shape}')

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(trn.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    save_data(trn_tst_ohe[:n_trn], y_trn, train_feature_file)
    save_data(trn_tst_ohe[n_trn:], None, test_feature_file)


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

