from sklearn.metrics import roc_auc_score as AUC
import tensorflow as tf

# ref: https://www.kaggle.com/abhishek/same-old-entity-embeddings
def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return AUC(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)