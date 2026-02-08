from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def build_autoencoder(dim):
    inp = Input(shape=(dim,))
    enc = Dense(32, activation='relu')(inp)
    dec = Dense(dim, activation='linear')(enc)
    return Model(inp, dec)
