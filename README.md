<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

</head>

<body>

<h1>EduFinance – Simulator</h1>
<h2>Modelo Académico de Análisis, Predicción y Simulación Financiera con Series de Tiempo</h2>

<h3>Descripción general</h3>
<p>
EduFinance – Simulator es un proyecto académico y práctico orientado al análisis, predicción y simulación de activos financieros mediante modelos estadísticos y de aprendizaje profundo.
</p>
<p>
El objetivo principal es desarrollar una herramienta pedagógica que permita analizar, modelar y simular escenarios de inversión, combinando métodos clásicos como <strong>ARIMA/GARCH</strong> con enfoques modernos como <strong>LSTM</strong>, usando datos históricos reales de acciones, ETFs y criptomonedas.
</p>

<h3>Objetivos del proyecto</h3>
<ul>
    <li>Analizar el comportamiento histórico de distintos activos financieros (ETFs, acciones, criptomonedas).</li>
    <li>Detectar tendencias, volatilidad y estacionariedad mediante herramientas estadísticas.</li>
    <li>Aplicar modelos de predicción clásicos (Regresión lineal, ARIMA, GARCH) y modernos (LSTM).</li>
    <li>Validar y comparar el desempeño de modelos predictivos.</li>
    <li>Simular escenarios de inversión y riesgo con base en las predicciones obtenidas.</li>
    <li>Desarrollar un notebook interactivo y documentado como herramienta de aprendizaje y simulación.</li>
</ul>

<h3>Estructura del notebook</h3>
<table>
    <tr><th>Fase</th><th>Descripción</th><th>Resultados clave</th></tr>
    <tr><td>0. Portada y contexto</td><td>Presenta la motivación y alcance del simulador.</td><td>Marco conceptual del proyecto.</td></tr>
    <tr><td>1. Preparación del entorno</td><td>Instalación de dependencias y configuración del entorno.</td><td>Entorno reproducible (Colab/VSCode).</td></tr>
    <tr><td>2. Obtención de datos</td><td>Descarga automatizada desde Yahoo Finance.</td><td>Dataset con precios ajustados diarios.</td></tr>
    <tr><td>3. Limpieza y normalización</td><td>Sincronización temporal y tratamiento de valores nulos.</td><td>Series limpias y homogéneas.</td></tr>
    <tr><td>4. Exploración estadística (EDA)</td><td>Análisis descriptivo, ADF, ACF y PACF.</td><td>Identificación de estacionariedad y volatilidad.</td></tr>
    <tr><td>5. Transformaciones y resumen</td><td>Evaluación de transformaciones (log, diff, retornos).</td><td>Series transformadas listas para modelado.</td></tr>
    <tr><td>6. Modelos ARIMA</td><td>Predicción clásica de precios/log-precios y validación (RMSE, MAPE).</td><td>Resultados por activo, interpretación y visualización.</td></tr>
    <tr><td>7. Modelos GARCH (opcional)</td><td>Estimación de volatilidad condicional.</td><td>Medición de riesgo y varianza esperada.</td></tr>
    <tr><td>8. Modelo LSTM (fase actual)</td><td>Implementación de red neuronal recurrente para predicción multiactivo.</td><td>Predicciones dinámicas y simulaciones de escenarios futuros.</td></tr>
    <tr><td>9. Simulación de inversión</td><td>Uso de pronósticos (ARIMA/LSTM) para evaluar escenarios.</td><td>Simulador interactivo educativo.</td></tr>
</table>

<h3>Conjunto piloto de activos</h3>
<table>
    <tr><th>Tipo</th><th>Símbolo</th><th>Descripción</th></tr>
    <tr><td>ETF</td><td>VOO</td><td>Vanguard S&P 500 ETF – replica el índice S&P 500.</td></tr>
    <tr><td></td><td>QQQ</td><td>Invesco QQQ – sigue el NASDAQ-100.</td></tr>
    <tr><td></td><td>EUNL.DE</td><td>iShares Core MSCI World – cobertura global de mercados desarrollados.</td></tr>
    <tr><td></td><td>XAR</td><td>SPDR Aerospace & Defense ETF – sector defensa.</td></tr>
    <tr><td>Acciones</td><td>TSLA</td><td>Tesla Inc. – sector tecnológico y automotriz.</td></tr>
    <tr><td></td><td>V</td><td>Visa Inc. – servicios financieros globales.</td></tr>
    <tr><td>Criptomonedas</td><td>BTC-USD</td><td>Bitcoin – activo descentralizado, alta volatilidad.</td></tr>
    <tr><td></td><td>XRP-USD</td><td>XRP – cripto enfocada en pagos rápidos.</td></tr>
</table>

<h3>Modelos incluidos</h3>

<h4>Modelos estadísticos clásicos</h4>
<ul>
    <li><strong>Regresión lineal simple:</strong> detección de tendencia en precios o log-precios.</li>
    <li><strong>ARIMA/SARIMA:</strong> modelado de la media y predicción temporal.</li>
    <li><strong>GARCH/EGARCH:</strong> modelado de la varianza y estimación de riesgo.</li>
</ul>

<h4>Modelos de aprendizaje profundo</h4>
<ul>
    <li><strong>LSTM (Long Short-Term Memory):</strong></li>
    <ul>
        <li>Predicción secuencial multiactivo.</li>
        <li>Entrenamiento con división temporal (train/test).</li>
        <li>Evaluación mediante RMSE, MAPE y comparación con ARIMA.</li>
        <li>Capacidad de extenderse a simulaciones de escenarios de inversión.</li>
    </ul>
</ul>

<h3>Requisitos y dependencias</h3>
<p>El notebook es completamente reproducible en <strong>Google Colab</strong> o <strong>Visual Studio Code</strong>.</p>

<pre><code>numpy==1.26.4
scipy==1.12.0
pandas==2.1.4
statsmodels==0.14.2
matplotlib==3.9.2
seaborn==0.13.2
yfinance==0.2.44
plotly==5.24.1
pyyaml==6.0.2
tensorflow&gt;=2.16.0  # para el modelo LSTM
</code></pre>

<h3>Metodología general</h3>
<ul>
    <li>Normalización temporal de las series a días hábiles.</li>
    <li>Transformación logarítmica y/o diferenciación para estabilizar varianzas.</li>
    <li>Evaluación de estacionariedad (ADF) y autocorrelaciones (ACF/PACF).</li>
    <li>Entrenamiento de modelos ARIMA con validación temporal (30 días).</li>
    <li>Entrenamiento del modelo LSTM con ventanas deslizantes y optimización por lotes.</li>
    <li>Comparación de resultados entre modelos clásicos y neuronales.</li>
    <li>Simulación de escenarios futuros basados en predicciones de LSTM.</li>
</ul>

<h3>Resultados esperados</h3>
<ul>
    <li>Series limpias, estacionarias y listas para predicción.</li>
    <li>Métricas de rendimiento (AIC, BIC, RMSE, MAPE) por activo y modelo.</li>
    <li>Predicciones de corto y mediano plazo mediante ARIMA y LSTM.</li>
    <li>Visualizaciones comparativas de desempeño predictivo.</li>
    <li>Simulador de escenarios de inversión con datos reales y proyecciones simuladas.</li>
</ul>

<h3>Autores</h3>
<ul>
    <li><strong>Didier Jesús Agamez Escobar</strong></li>
    <li><strong>María Valentina Serna González</strong></li>
    <li><strong>Luis Mario Díaz Martínez</strong></li>
</ul>

<h3>Licencia</h3>
<p>
Distribuido bajo licencia <strong>MIT</strong>.  
Puedes reutilizar, modificar y compartir el código citando el repositorio original.
</p>

<footer>
Proyecto desarrollado para el <strong>Programa de Ciencia de Datos e Ingeniería de Sistemas – Universidad Tecnológica de Bolívar</strong>.
</footer>

</body>
</html>
