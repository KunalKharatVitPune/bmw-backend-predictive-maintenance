import tensorflow as tf
import numpy as np

def cost_sensitive_loss(weight=6.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean((1 + weight * y_true) * bce)
    return loss

def create_sequences(X, y1, y2, time_steps=50):
    Xs, y_fail, y_rul = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        y_fail.append(y1.iloc[i+time_steps])
        y_rul.append(y2.iloc[i+time_steps])
    return np.array(Xs), np.array(y_fail), np.array(y_rul)
