#!/usr/bin/env python
from __future__ import division
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from kaggler.data_io import load_data, save_data
from kaggler.preprocessing import LabelEncoder, TargetEncoder

from const import ID_COL, TARGET_COL, SEED, N_FOLD


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file, index_col=ID_COL)
    tst = pd.read_csv(test_file, index_col=ID_COL)

    y = trn[TARGET_COL]
    n_trn = trn.shape[0]

    trn.drop(TARGET_COL, axis=1, inplace=True)
    logging.info(f'categorical: {trn.shape[1]})')

    df = pd.concat([trn, tst], axis=0)

    logging.info('label encoding categorical variables')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    te = TargetEncoder(cv=cv)
    te.fit(trn, y)
    df = te.transform(df)

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(df.columns):
            f.write('{}\t{}\tq\n'.format(i, col))

    logging.info('saving features')
    save_data(df.values[:n_trn,], y.values, train_feature_file)
    save_data(df.values[n_trn:,], None, test_feature_file)


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

