# EduFinance Simulator - Streamlit Dashboard
# Proyecto: Análisis, Predicción y Simulación Financiera con Series de Tiempo
# Autores: Didier J. Agamez, María V. Serna, Luis M. Díaz
# Universidad Tecnológica de Bolívar

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pathlib import Path
import os

# -------------------------------
# CONFIGURACIÓN INICIAL
# -------------------------------
st.set_page_config(
    page_title="EduFinance Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 EduFinance Simulator")
st.write("Dashboard Interactivo de Modelado Financiero (ARIMA & GARCH)")
st.caption("Análisis de ETFs, Acciones y Criptomonedas")

# -------------------------------
# CARGA DE DATOS
# -------------------------------
@st.cache_data
def load_data():
    """Carga todos los datos del proyecto desde las rutas correctas"""
    base_path = Path(__file__).parent
    
    # Datos de precios y retornos
    prices = pd.read_csv(base_path / "data/time_series/prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(base_path / "data/time_series/returns.csv", index_col=0, parse_dates=True)
    log_prices = pd.read_csv(base_path / "data/time_series/log_prices.csv", index_col=0, parse_dates=True)
    
    # Métricas de modelos
    arima_metrics = pd.read_csv(base_path / "models_results/Arima_results/ARIMA_all_metrics.csv")
    garch_metrics = pd.read_csv(base_path / "models_results/Garch_results/GARCH_all_metrics.csv")
    
    # Cargar comparaciones de ARIMA y GARCH para todos los activos
    arima_results = {}
    garch_results = {}
    
    tickers = prices.columns.tolist()
    
    for ticker in tickers:
        # ARIMA
        arima_path = base_path / f"models_results/Arima_results/{ticker}_comparison.csv"
        if arima_path.exists():
            arima_results[ticker] = pd.read_csv(arima_path, index_col=0)
        
        # GARCH
        garch_path = base_path / f"models_results/Garch_results/{ticker}_garch_comparison.csv"
        if garch_path.exists():
            garch_results[ticker] = pd.read_csv(garch_path, index_col=0)
    
    return prices, returns, log_prices, arima_metrics, garch_metrics, arima_results, garch_results, tickers

try:
    prices, returns, log_prices, arima_metrics, garch_metrics, arima_results, garch_results, tickers = load_data()
except Exception as e:
    st.error(f"⚠ Error cargando datos: {e}")
    st.info("Verifica que existan los archivos en data/time_series/ y models_results/")
    st.stop()

# -------------------------------
# SIDEBAR - CONFIGURACIÓN
# -------------------------------
st.sidebar.header("⚙ Configuración")

# Información sobre los activos
asset_info = {
    "BTC-USD": "Bitcoin - Criptomoneda descentralizada",
    "XRP-USD": "XRP - Cripto enfocada en pagos rápidos",
    "TSLA": "Tesla Inc. - Sector tecnológico y automotriz",
    "V": "Visa Inc. - Servicios financieros globales",
    "VOO": "Vanguard S&P 500 ETF",
    "QQQ": "Invesco QQQ - NASDAQ-100",
    "EUNL.DE": "iShares Core MSCI World",
    "XAR": "SPDR Aerospace & Defense ETF"
}

asset = st.sidebar.selectbox(
    "Selecciona un activo", 
    tickers,
    format_func=lambda x: f"{x} - {asset_info.get(x, 'Activo financiero')}"
)

# Tabs para organizar la información
tab_config, tab_info = st.sidebar.tabs(["⚙ Configuración", "ℹ Info del Proyecto"])

with tab_config:
    show_info = st.checkbox("Mostrar información básica", True)

with tab_info:
    st.markdown("### 📚 Sobre el Proyecto")
    st.markdown("""
    **EduFinance Simulator** es una herramienta interactiva para analizar, modelar y predecir 
    el comportamiento de activos financieros mediante modelos estadísticos y de deep learning.
    """)
    
    st.markdown("---")
    st.markdown("### 🎓 Conceptos Clave")
    
    with st.expander("📊 ¿Qué es ARIMA?"):
        st.markdown("""
        **ARIMA** (AutoRegressive Integrated Moving Average) es un modelo estadístico clásico 
        para predicción de series temporales que combina:
        - **AR**: Autoregresión (valores pasados)
        - **I**: Integración (diferenciación para estacionariedad)
        - **MA**: Media móvil (errores pasados)
        
        Se usa para predecir precios futuros basándose en patrones históricos.
        """)
    
    with st.expander("📈 ¿Qué es GARCH?"):
        st.markdown("""
        **GARCH** (Generalized AutoRegressive Conditional Heteroskedasticity) es un modelo 
        que estima la **volatilidad condicional** de un activo financiero.
        
        - Captura la variabilidad del riesgo a lo largo del tiempo
        - Modela cómo la volatilidad cambia en períodos de alta/baja incertidumbre
        - Útil para gestión de riesgos y pricing de opciones
        """)
    
    with st.expander("🧠 ¿Qué es LSTM?"):
        st.markdown("""
        **LSTM** (Long Short-Term Memory) es un tipo de red neuronal recurrente diseñada 
        para aprender dependencias a largo plazo en secuencias de datos.
        
        - Ideal para series temporales complejas
        - Captura patrones no lineales que ARIMA no puede modelar
        - Se entrena con grandes volúmenes de datos históricos
        """)
    
    with st.expander("📉 ¿Qué es Volatilidad?"):
        st.markdown("""
        La **volatilidad** mide cuánto varía el precio de un activo en un período de tiempo.
        
        - **Alta volatilidad**: Mayor riesgo y potencial de ganancia/pérdida
        - **Baja volatilidad**: Movimientos de precio más estables
        - Se calcula como la desviación estándar de los retornos
        """)
    
    with st.expander("🔢 ¿Qué es Retorno Logarítmico?"):
        st.markdown("""
        El **retorno logarítmico** es una medida de cambio porcentual entre dos períodos:
        
        ```
        r(t) = ln(P(t) / P(t-1))
        ```
        
        **Ventajas:**
        - Aditivo en el tiempo
        - Simétrico (pérdidas y ganancias)
        - Asume distribución más cercana a la normal
        """)
    
    with st.expander("📏 ¿Qué son RMSE, MAE y MAPE?"):
        st.markdown("""
        Son métricas para evaluar la precisión de las predicciones:
        
        **RMSE** (Root Mean Squared Error):
        - Raíz del error cuadrático medio
        - Penaliza más los errores grandes
        - En las mismas unidades que la variable predicha
        - Valores más bajos indican mejor precisión
        
        **MAE** (Mean Absolute Error):
        - Error absoluto promedio
        - Más robusto ante valores atípicos que RMSE
        - Fácil de interpretar: promedio de cuánto se equivoca el modelo
        
        **MAPE** (Mean Absolute Percentage Error):
        - Error promedio en porcentaje
        - Fácil de interpretar (ej: 5% de error)
        - Útil para comparar modelos en diferentes escalas
        """)
    
    with st.expander("📐 ¿Qué son AIC y BIC?"):
        st.markdown("""
        **AIC** (Akaike Information Criterion) y **BIC** (Bayesian Information Criterion) 
        son criterios para seleccionar el mejor modelo:
        
        - Evalúan el balance entre **bondad de ajuste** y **complejidad del modelo**
        - **Valores más bajos** indican mejores modelos
        - AIC penaliza menos la complejidad que BIC
        - BIC favorece modelos más simples (parsimoniosos)
        - Se usan para comparar diferentes órdenes ARIMA o GARCH
        
        **Fórmula general:**
        - AIC = -2·log(L) + 2·k
        - BIC = -2·log(L) + k·log(n)
        
        Donde L es la verosimilitud, k es el número de parámetros, y n el tamaño de muestra.
        """)
    
    st.markdown("---")
    st.markdown("### 💼 Activos Analizados")
    
    activos_detalle = {
        "VOO": {
            "nombre": "Vanguard S&P 500 ETF",
            "tipo": "ETF",
            "sector": "Diversificado (500 empresas de EE.UU.)",
            "descripcion": "Replica el índice S&P 500, representa las 500 empresas más grandes de EE.UU."
        },
        "QQQ": {
            "nombre": "Invesco QQQ Trust",
            "tipo": "ETF",
            "sector": "Tecnología (NASDAQ-100)",
            "descripcion": "Sigue las 100 empresas tecnológicas más grandes del NASDAQ (Apple, Microsoft, Amazon, etc.)"
        },
        "EUNL.DE": {
            "nombre": "iShares Core MSCI World",
            "tipo": "ETF",
            "sector": "Global - Mercados Desarrollados",
            "descripcion": "Cobertura global con exposición a mercados desarrollados de todo el mundo."
        },
        "XAR": {
            "nombre": "SPDR Aerospace & Defense",
            "tipo": "ETF",
            "sector": "Defensa y Aeroespacial",
            "descripcion": "Empresas del sector defensa, aeronáutica y tecnología espacial."
        },
        "TSLA": {
            "nombre": "Tesla Inc.",
            "tipo": "Acción",
            "sector": "Automotriz / Tecnología",
            "descripcion": "Fabricante de vehículos eléctricos y soluciones de energía sostenible."
        },
        "V": {
            "nombre": "Visa Inc.",
            "tipo": "Acción",
            "sector": "Servicios Financieros",
            "descripcion": "Líder global en procesamiento de pagos digitales y tarjetas de crédito."
        },
        "BTC-USD": {
            "nombre": "Bitcoin",
            "tipo": "Criptomoneda",
            "sector": "Activo Digital Descentralizado",
            "descripcion": "Primera y más grande criptomoneda, conocida por su alta volatilidad y uso como reserva de valor digital."
        },
        "XRP-USD": {
            "nombre": "XRP (Ripple)",
            "tipo": "Criptomoneda",
            "sector": "Pagos y Transferencias",
            "descripcion": "Criptomoneda enfocada en pagos transfronterizos rápidos y de bajo costo."
        }
    }
    
    for ticker, info in activos_detalle.items():
        with st.expander(f"**{ticker}** - {info['nombre']}"):
            st.markdown(f"""
            - **Tipo:** {info['tipo']}
            - **Sector:** {info['sector']}
            - **Descripción:** {info['descripcion']}
            """)
    
    st.markdown("---")
    st.markdown("### 👥 Equipo")
    st.markdown("""
    - **Didier Jesús Agamez Escobar**
    - **María Valentina Serna González**
    - **Luis Mario Díaz Martínez**
    
    *Universidad Tecnológica de Bolívar*
    """)

show_info = tab_config.checkbox("Mostrar información básica", True) if 'show_info' not in locals() else show_info

# -------------------------------
# PANEL DE INFORMACIÓN BÁSICA
# -------------------------------
if show_info:
    st.sidebar.subheader("ℹ Información del activo")
    st.sidebar.write(f"**{asset}**")
    st.sidebar.write(asset_info.get(asset, "Activo financiero"))
    st.sidebar.divider()
    st.sidebar.write("**Conceptos clave:**")
    st.sidebar.markdown("- **Volatilidad:** mide la variabilidad del precio")
    st.sidebar.markdown("- **ARIMA:** modelo para predicción de series temporales")
    st.sidebar.markdown("- **GARCH:** modelo para estimar volatilidad condicional")

# -------------------------------
# MÉTRICAS POR MODELO
# -------------------------------
st.header(f"📌 Resumen del activo: **{asset}**")

# Obtener métricas del activo
arima_metric = arima_metrics[arima_metrics["ticker"] == asset]
garch_metric = garch_metrics[garch_metrics["ticker"] == asset]

col1, col2, col3, col4 = st.columns(4)

# Métricas ARIMA
if not arima_metric.empty:
    col1.metric("RMSE ARIMA", f"{arima_metric['rmse'].values[0]:.3f}")
    col2.metric("MAPE ARIMA", f"{arima_metric['mape'].values[0]:.2f}%")
else:
    col1.metric("RMSE ARIMA", "N/A")
    col2.metric("MAPE ARIMA", "N/A")

# Métricas GARCH
if not garch_metric.empty:
    col3.metric("RMSE GARCH (Vol)", f"{garch_metric['rmse_vol'].values[0]:.4f}")
    col4.metric("Persistencia", f"{garch_metric['persistence'].values[0]:.3f}")
else:
    col3.metric("RMSE GARCH", "N/A")
    col4.metric("Persistencia", "N/A")

# -------------------------------
# GRÁFICA: HISTÓRICO DE PRECIOS + PREDICCIONES ARIMA
# -------------------------------
st.subheader("📈 Histórico de Precios y Predicciones ARIMA")

fig = go.Figure()

# Precios históricos
fig.add_trace(go.Scatter(
    x=prices.index, 
    y=prices[asset], 
    mode="lines", 
    name="Histórico",
    line=dict(color='blue', width=2)
))

# Predicciones ARIMA si existen
if asset in arima_results:
    arima_df = arima_results[asset]
    fig.add_trace(go.Scatter(
        x=arima_df.index, 
        y=arima_df['predicted'], 
        mode="lines", 
        name="Predicción ARIMA",
        line=dict(color='red', dash='dash', width=2)
    ))

fig.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Precio",
    hovermode='x unified',
    template='plotly_white',
    xaxis=dict(
        range=['2017-11-01', '2025-09-30']
    )
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# GRÁFICA: RETORNOS + VOLATILIDAD GARCH
# -------------------------------
st.subheader("📉 Retornos y Volatilidad GARCH")

fig2 = go.Figure()

# Retornos
fig2.add_trace(go.Scatter(
    x=returns.index, 
    y=returns[asset], 
    mode="lines", 
    name="Retornos",
    line=dict(color='green', width=1)
))

# Volatilidad GARCH si existe
if asset in garch_results:
    garch_df = garch_results[asset]
    if 'volatility' in garch_df.columns:
        fig2.add_trace(go.Scatter(
            x=garch_df.index, 
            y=garch_df['volatility'], 
            mode="lines", 
            name="Volatilidad GARCH",
            line=dict(color='orange', width=2)
        ))

fig2.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Valor",
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# TABLA DE VALIDACIÓN ARIMA
# -------------------------------
st.subheader("📑 Tabla de Validación de Predicciones ARIMA")

if asset in arima_results:
    validation_df = arima_results[asset].copy()
    
    # Renombrar columnas para mejor visualización
    if 'actual' in validation_df.columns and 'predicted' in validation_df.columns:
        validation_df = validation_df.rename(columns={
            'actual': 'Precio Real',
            'predicted': 'Predicción ARIMA',
            'abs_error': 'Error Absoluto',
            'mape (%)': 'MAPE (%)'
        })
        
        # Mostrar últimas 30 predicciones
        st.dataframe(validation_df.tail(30), use_container_width=True)
        
        # Estadísticas de error
        col1, col2, col3 = st.columns(3)
        if 'Error Absoluto' in validation_df.columns:
            col1.metric("Error Medio Absoluto", f"{validation_df['Error Absoluto'].mean():.2f}")
            col2.metric("Error Máximo", f"{validation_df['Error Absoluto'].max():.2f}")
        if 'MAPE (%)' in validation_df.columns:
            col3.metric("MAPE Promedio", f"{validation_df['MAPE (%)'].mean():.2f}%")
    else:
        st.dataframe(validation_df.tail(30), use_container_width=True)
else:
    st.warning(f"No hay resultados ARIMA disponibles para {asset}")

# -------------------------------
# TABLA DE VALIDACIÓN GARCH
# -------------------------------
st.subheader("📊 Resultados GARCH - Volatilidad")

if asset in garch_results:
    garch_df = garch_results[asset].copy()
    st.dataframe(garch_df.tail(30), use_container_width=True)
    
    # Estadísticas de volatilidad
    if 'volatility' in garch_df.columns:
        col1, col2, col3 = st.columns(3)
        col1.metric("Volatilidad Promedio", f"{garch_df['volatility'].mean():.4f}")
        col2.metric("Volatilidad Máxima", f"{garch_df['volatility'].max():.4f}")
        col3.metric("Volatilidad Mínima", f"{garch_df['volatility'].min():.4f}")
else:
    st.warning(f"No hay resultados GARCH disponibles para {asset}")

# -------------------------------
# COMPARACIÓN DE MODELOS
# -------------------------------
st.subheader("🔍 Comparación de Modelos")

col1, col2 = st.columns(2)

with col1:
    st.write("**Métricas ARIMA**")
    if not arima_metric.empty:
        metrics_arima = pd.DataFrame({
            'Métrica': ['RMSE', 'MAE', 'MAPE (%)', 'AIC', 'BIC'],
            'Valor': [
                f"{arima_metric['rmse'].values[0]:.4f}",
                f"{arima_metric['mae'].values[0]:.4f}",
                f"{arima_metric['mape'].values[0]:.2f}",
                f"{arima_metric['aic'].values[0]:.2f}",
                f"{arima_metric['bic'].values[0]:.2f}"
            ]
        })
        st.dataframe(metrics_arima, hide_index=True, use_container_width=True)
    else:
        st.write("No disponible")

with col2:
    st.write("**Métricas GARCH**")
    if not garch_metric.empty:
        metrics_garch = pd.DataFrame({
            'Métrica': ['RMSE Vol', 'RMSE Ret', 'MAPE Ret (%)', 'Persistencia', 'AIC'],
            'Valor': [
                f"{garch_metric['rmse_vol'].values[0]:.4f}",
                f"{garch_metric['rmse_ret'].values[0]:.4f}",
                f"{garch_metric['mape_ret'].values[0]:.2f}",
                f"{garch_metric['persistence'].values[0]:.4f}",
                f"{garch_metric['aic'].values[0]:.2f}"
            ]
        })
        st.dataframe(metrics_garch, hide_index=True, use_container_width=True)
    else:
        st.write("No disponible")

# -------------------------------
# DESCARGA DE RESULTADOS
# -------------------------------
st.subheader("⬇ Descarga de Datos")

col1, col2, col3 = st.columns(3)

def convert_df(df):
    return df.to_csv().encode("utf-8")

# Descarga de datos históricos
with col1:
    csv_prices = convert_df(prices[[asset]])
    st.download_button(
        "📊 Descargar Precios",
        csv_prices,
        file_name=f"{asset}_precios.csv",
        mime="text/csv",
    )

with col2:
    if asset in arima_results:
        csv_arima = convert_df(arima_results[asset])
        st.download_button(
            "📈 Descargar ARIMA",
            csv_arima,
            file_name=f"{asset}_arima.csv",
            mime="text/csv",
        )

with col3:
    if asset in garch_results:
        csv_garch = convert_df(garch_results[asset])
        st.download_button(
            "📉 Descargar GARCH",
            csv_garch,
            file_name=f"{asset}_garch.csv",
            mime="text/csv",
        )

# -------------------------------
# PIE DE PÁGINA
# -------------------------------
st.divider()
st.caption("**EduFinance Simulator** - Proyecto académico de análisis financiero")
st.caption("Autores: Didier J. Agamez, María V. Serna, Luis M. Díaz | Universidad Tecnológica de Bolívar")
st.caption("Modelos: ARIMA (predicción de precios) y GARCH (estimación de volatilidad)")
