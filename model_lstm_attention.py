from tensorflow.keras.layers import Input, LSTM, Dense, Attention, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(64, return_sequences=True)(x)

    attn = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn)

    x = Dense(64, activation='relu')(x)

    failure = Dense(1, activation='sigmoid', name='failure')(x)
    rul = Dense(1, activation='linear', name='rul')(x)

    return Model(inputs, [failure, rul])
