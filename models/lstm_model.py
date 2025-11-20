import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def directional_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(true_dir == pred_dir) * 100

def theils_u(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num = np.sqrt(np.mean((y_pred - y_true)**2))
    den = np.sqrt(np.mean(y_true[:-1]**2) + np.mean(y_pred[:-1]**2))
    return num / den

def correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def create_supervised_dataset(series, lookback):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y

def create_lstm_model(config):
    model = Sequential()
    model.add(LSTM(config["units"], input_shape=(config["lookback"], 1)))
    model.add(Dropout(config["dropout"]))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    return model

def train_and_predict_lstm(series, config, test_size=30):
    # Convertir a array
    values = series.values.reshape(-1, 1)

    # SPLIT
    train_values = values[:-test_size]
    test_values  = values[-test_size:]

    # SCALER
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    full_scaled = scaler.transform(values)

    # DATASET SUPERVISADO DEL TRAIN
    X_train, y_train = create_supervised_dataset(train_scaled.flatten(), config["lookback"])

    # MODELO
    model = create_lstm_model(config)

    # CALL BACK early stopping
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    # ENTRENAR
    history = model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=0,
        callbacks=[es]
    )

    # PREDICCIÓN ONE-STEP DE LOS ÚLTIMOS 30 DÍAS
    preds_scaled = []
    for i in range(len(values) - test_size, len(values)):
        start_idx = i - config["lookback"]
        window = full_scaled[start_idx:i].reshape(1, config["lookback"], 1)
        pred = model.predict(window, verbose=0)[0][0]
        preds_scaled.append(pred)

    # Desescalar predicciones
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()

    # Valores reales
    y_true = test_values.flatten()

    # MÉTRICAS
    results = {
        "MAE": mean_absolute_error(y_true, preds),
        "MSE": mean_squared_error(y_true, preds),
        "RMSE": rmse(y_true, preds),
        "MAPE": mape(y_true, preds),
        "DA": directional_accuracy(y_true, preds),
        "Theils_U": theils_u(y_true, preds),
        "Correlation": correlation(y_true, preds),
    }

    return y_true, preds, results, history
