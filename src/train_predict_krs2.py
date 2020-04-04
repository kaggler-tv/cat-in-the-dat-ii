from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC

import argparse
import logging
import keras.backend as K
import numpy as np
import os
import pandas as pd
import time


from kaggler.data_io import load_data
from const import N_FOLD, SEED
from fallback_auc import auc

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils

np.random.seed(SEED) # for reproducibility

# ref: # https://www.kaggle.com/abhishek/same-old-entity-embeddings
def create_keras_embedding_model(data, cat_cols, N_uniq=50, N_dense=300, drop_rate=0.3):
    inputs = []
    outputs = []
    for col in cat_cols:
        n_unique = int(data[col].nunique())
        embed_dim = int(min(np.ceil(n_unique/2), N_uniq))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(n_unique + 1, embed_dim, name=col)(inp)
        out = layers.SpatialDropout1D(drop_rate)(out)
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        inputs.append(inp)
        outputs.append(out)
    
    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(N_dense, activation='relu')(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(N_dense, activation='relu')(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.BatchNormalization()(x)
    
    y = layers.Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=y)
    
    return model


def train_predict(train_file, test_file, predict_valid_file, predict_test_file, feature_map, retrain=True):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X_trn, y_trn = load_data(train_file)
    X_tst, _ = load_data(test_file)
    
    feature_map = pd.read_table(feature_map, index_col=0, header=None, names=['feature_names', 'feature_type'])
    features = feature_map['feature_names'].values
    train_df = pd.DataFrame(data=X_trn.toarray(), columns=feature_map['feature_names']) 
    test_df = pd.DataFrame(data=X_tst.toarray(), columns=feature_map['feature_names'])
    train_test = train_df.append(test_df)
    
    test_data = [test_df.loc[:, features].values[:, k] for k in range(test_df.loc[:, features].values.shape[1])]

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=SEED)

    vld_preds = np.zeros_like(y_trn)
    tst_preds = np.zeros((X_tst.shape[0],))
    for cv_idx, (i_trn, i_vld) in enumerate(cv.split(X_trn, y_trn), 1):

        X_trn_cv = train_df.iloc[i_trn, :].reset_index(drop=True)
        X_vld_cv = train_df.iloc[i_vld, :].reset_index(drop=True)
        y_trn_cv = y_trn[i_trn]
        y_vld_cv = y_trn[i_vld]

        logging.info('Training model #{}'.format(cv_idx))

        clf = create_keras_embedding_model(train_test, features)
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

        X_trn_cv = [X_trn_cv.loc[:, features].values[:, k] for k in range(X_trn_cv.loc[:, features].values.shape[1])]
        X_vld_cv = [X_vld_cv.loc[:, features].values[:, k] for k in range(X_vld_cv.loc[:, features].values.shape[1])]
        
        es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,
                                     verbose=1, mode='max', baseline=None, restore_best_weights=True)

        rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                          patience=3, min_lr=1e-6, mode='max', verbose=1)
        
        clf.fit(X_trn_cv,
                utils.to_categorical(y_trn_cv),
                validation_data=(X_vld_cv, utils.to_categorical(y_vld_cv)),
                verbose=0,
                batch_size=1024,
                callbacks=[es, rlr],
                epochs=50)

        vld_preds[i_vld] = clf.predict(X_vld_cv)[:, 1]
        
        logging.info('CV #{}: {:.4f}'.format(cv_idx, auc(y_trn[i_vld], vld_preds[i_vld])))

        if not retrain:
            tst_preds += (model.predict(test_data)[:, 1] / N_FOLDS).ravel()

    logging.info('Saving validation predictions...')
    logging.info('CV: {:.4f}'.format(AUC(y_trn, vld_preds)))
    np.savetxt(predict_valid_file, vld_preds, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')

        clf = create_keras_embedding_model(train_test, features)
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

        X_trn_all = [train_df.loc[:, features].values[:, k] for k in range(train_df.loc[:, features].values.shape[1])]

        clf.fit(X_trn_all,
                utils.to_categorical(y_trn),
                validation_data=(X_trn_all, utils.to_categorical(y_trn)),
                verbose=0,
                batch_size=1024,
                callbacks=[es, rlr],
                epochs=50)

        tst_preds = (clf.predict(test_data)[:, 1]).ravel()


    logging.info('Saving normalized test predictions...')
    np.savetxt(predict_test_file, tst_preds, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--feature-map', required=True, dest='feature_map')
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_feature_file,
                  test_file=args.test_feature_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_map=args.feature_map,
                  retrain=args.retrain)

    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                 60.))
