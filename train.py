import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import engineer_features
from utils import create_sequences, cost_sensitive_loss
from model_lstm_attention import build_model
from model_autoencoder import build_autoencoder

df = pd.read_csv("data/ev_dataset.csv")

X, y_fail, y_rul = engineer_features(df)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_seq, yF, yR = create_sequences(X_scaled, y_fail, y_rul)

model = build_model(X_seq.shape[1:])
model.compile(
    optimizer='adam',
    loss={
        'failure': cost_sensitive_loss(),
        'rul': 'mse'
    }
)
model.fit(X_seq, {'failure': yF, 'rul': yR}, epochs=15, batch_size=64)
model.save("models/lstm_attention.h5")

ae = build_autoencoder(X_scaled.shape[1])
ae.compile(optimizer='adam', loss='mse')
ae.fit(X_scaled, X_scaled, epochs=10, batch_size=64)
ae.save("models/autoencoder.h5")
