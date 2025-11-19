# EduFinance Simulator - Streamlit Dashboard (versión completa)
# Archivo: app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date

# -------------------------------
# CONFIGURACIÓN INICIAL
# -------------------------------
st.set_page_config(
    page_title="EduFinance Simulator",
    layout="wide",
)

st.title("📊 EduFinance Simulator")
st.write("Dashboard Interactivo de Modelado Financiero (ARIMA, GARCH, LSTM)")

# -------------------------------
# CARGA DE DATOS
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_prices.csv", index_col=0, parse_dates=True)
    log_diff = pd.read_csv("data_returns.csv", index_col=0, parse_dates=True)
    arima_pred = pd.read_csv("pred_arima.csv", index_col=0, parse_dates=True)
    garch_vol = pd.read_csv("garch_vol.csv", index_col=0, parse_dates=True)
    lstm_price = pd.read_csv("lstm_price.csv", index_col=0, parse_dates=True)
    lstm_returns = pd.read_csv("lstm_returns.csv", index_col=0, parse_dates=True)
    metrics = pd.read_csv("model_metrics.csv")
    return df, log_diff, arima_pred, garch_vol, lstm_price, lstm_returns, metrics

# NOTA: El usuario debe colocar estos CSV generados por sus modelos.

# Simulación si no existen
try:
    df, log_diff, arima_pred, garch_vol, lstm_price, lstm_returns, metrics = load_data()
except:
    st.warning("⚠ No se encontraron archivos de datos. Coloca los CSV de tus modelos.")
    st.stop()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙ Configuración")
asset = st.sidebar.selectbox("Selecciona un activo", df.columns)

horizon = st.sidebar.slider("Horizonte de predicción (días)", 7, 60, 30)

show_info = st.sidebar.checkbox("Mostrar información básica", True)

# -------------------------------
# PANEL DE INFORMACIÓN BÁSICA
# -------------------------------
if show_info:
    st.sidebar.subheader("ℹ Información del activo")
    st.sidebar.write(f"**Activo:** {asset}")
    st.sidebar.write("**Conceptos clave:**")
    st.sidebar.markdown("- *Volatilidad:* mide cuánto varía el precio")
    st.sidebar.markdown("- *ARIMA:* modelo lineal para predicción de series")
    st.sidebar.markdown("- *GARCH:* modelo para estimar volatilidad")
    st.sidebar.markdown("- *LSTM:* red neuronal para dependencias no lineales")

# -------------------------------
# MÉTRICAS POR MODELO
# -------------------------------
st.header(f"📌 Resumen del activo: **{asset}**")

asset_metrics = metrics[metrics["asset"] == asset]

col1, col2, col3 = st.columns(3)

# Manejo seguro si no hay métricas para el activo
if not asset_metrics.empty:
    col1.metric("RMSE ARIMA", f"{asset_metrics['rmse_arima'].values[0]:.3f}")
    col2.metric("RMSE LSTM", f"{asset_metrics['rmse_lstm'].values[0]:.3f}")
else:
    col1.metric("RMSE ARIMA", "N/A")
    col2.metric("RMSE LSTM", "N/A")

# Manejo seguro si no existe la serie de volatilidad para el activo
try:
    vol_val = f"{garch_vol[asset].iloc[-1]:.4f}"
except Exception:
    vol_val = "N/A"

col3.metric("Volatilidad GARCH", vol_val)

# -------------------------------
# GRÁFICA: HISTÓRICO + ARIMA + LSTM
# -------------------------------
st.subheader("📈 Predicción de Precios (Histórico + ARIMA + LSTM)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[asset], mode="lines", name="Histórico"))

# Añadir trazas solo si existen las columnas
if asset in arima_pred.columns:
    fig.add_trace(go.Scatter(x=arima_pred.index, y=arima_pred[asset], mode="lines", name="ARIMA"))
if asset in lstm_price.columns:
    fig.add_trace(go.Scatter(x=lstm_price.index, y=lstm_price[asset], mode="lines", name="LSTM"))

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# GRÁFICA: RETORNOS + VOLATILIDAD GARCH
# -------------------------------
st.subheader("📉 Retornos + Volatilidad (GARCH)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=log_diff.index, y=log_diff[asset], mode="lines", name="Retornos"))
if asset in garch_vol.columns:
    fig2.add_trace(go.Scatter(x=garch_vol.index, y=garch_vol[asset], mode="lines", name="Volatilidad GARCH"))

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# TABLA DE VALIDACIÓN
# -------------------------------
st.subheader("📑 Tabla de Validación de Predicciones")

validation_df = pd.DataFrame({
    "Actual": df[asset].iloc[-horizon:],
    "ARIMA": arima_pred[asset].iloc[-horizon:] if asset in arima_pred.columns else np.nan,
    "LSTM": lstm_price[asset].iloc[-horizon:] if asset in lstm_price.columns else np.nan,
})

validation_df["AbsError_ARIMA"] = abs(validation_df["Actual"] - validation_df["ARIMA"])
validation_df["AbsError_LSTM"] = abs(validation_df["Actual"] - validation_df["LSTM"])

st.dataframe(validation_df)

# -------------------------------
# DESCARGA DE RESULTADOS
# -------------------------------
def convert_df(df):
    return df.to_csv().encode("utf-8")

st.download_button(
    "⬇ Descargar tabla en CSV",
    convert_df(validation_df),
    file_name=f"validacion_{asset}.csv",
    mime="text/csv",
)
