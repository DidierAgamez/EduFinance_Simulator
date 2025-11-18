import pandas as pd
from typing import Tuple, Dict
import numpy as np
import importlib
from functools import lru_cache

def first_valid_dates_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 1 - validamoslas fechas de cada ticker

    Objetivo:
    - Encontrar para cada ticker en el DataFrame:
      • Fecha mínima disponible (first_date).
      • Fecha máxima disponible (last_date).
      • Número total de registros (n_total).
      • Número de valores no nulos en 'close' (n_nonnull).
      • Primera fecha válida (no nula) en 'close' (first_valid_date).
      • Proporción de cobertura de datos válidos (coverage_ratio).

    Parámetros:
    ----------
    df : pd.DataFrame
        Dataset en formato tidy con al menos las columnas:
        ['date', 'ticker', 'close'].
        - 'date' puede estar en datetime64 o string; se normaliza a date.
        - 'ticker' identifica el activo.
        - 'close' es el precio de cierre (puede contener NaN).

    Retorna:
    --------
    pd.DataFrame
        Tabla resumen con columnas:
        ['ticker', 'first_valid_date', 'first_date', 'last_date',
         'n_total', 'n_nonnull', 'coverage_ratio']
        donde:
        - ticker: símbolo del activo.
        - first_valid_date: primera fecha con dato válido en 'close'.
        - first_date: fecha mínima en el dataset.
        - last_date: fecha máxima en el dataset.
        - n_total: número total de filas para el ticker.
        - n_nonnull: número de filas con 'close' no nulo.
        - coverage_ratio: n_nonnull / n_total (proporción de cobertura).

    Notas:
    ------
    • Esta función es útil en el pipeline de normalización de series,
      ya que permite identificar el activo con el inicio más reciente.
    • Ese inicio común servirá para alinear todas las series al mismo rango.
    """

    # Normalizamos el tipo de dato de la columna 'date' a objeto date
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.date

    # Agrupamos por ticker y calculamos métricas básicas
    g = tmp.sort_values(["ticker", "date"]).groupby("ticker", as_index=False)
    agg = g.agg(
        first_date=("date", "min"),     # fecha mínima en el dataset
        last_date=("date", "max"),      # fecha máxima en el dataset
        n_total=("close", "size"),      # número total de registros
        n_nonnull=("close", lambda s: s.notna().sum())  # registros no nulos
    )

    # Obtenemos la primera fecha con dato no nulo de 'close' por ticker
    first_valid = (
        tmp[tmp["close"].notna()]
        .sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .agg(first_valid_date=("date", "min"))
    )

    # Unimos ambos resultados
    out = agg.merge(first_valid, on="ticker", how="left")

    # Calculamos proporción de cobertura
    out["coverage_ratio"] = out["n_nonnull"] / out["n_total"]

    return out[[
        "ticker", "first_valid_date", "first_date", "last_date",
        "n_total", "n_nonnull", "coverage_ratio"
    ]]

def common_start_date(df: pd.DataFrame, strict: bool = True) -> Tuple[pd.Timestamp, pd.DataFrame]:
    """
    Paso 2 - Determinamos una fecha de inicio comun entre los diferentes tickers
    cogiendo como fecha de incio común la primera fecha con dato válido en 'close'.

    Objetivo:
    ----------
    - A partir de un DataFrame de series financieras, identificar:
      • La primera fecha con dato válido (no nulo) de cada ticker.
      • La fecha de inicio común más reciente (máximo de esas fechas).
    - Esta fecha común se usará como punto de arranque del dataset
      normalizado para garantizar que todos los tickers tengan datos
      desde ese día en adelante.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame en formato tidy con al menos:
        ['date', 'ticker', 'close'].
        - 'date' puede estar en datetime64 o string.
        - 'ticker' identifica el activo.
        - 'close' es el precio de cierre (puede contener NaN).

    strict : bool, opcional (default=True)
        • True  → si algún ticker no tiene datos no nulos en 'close',
                   lanza un ValueError.
        • False → ignora esos tickers (se eliminan de la tabla).

    Retorna:
    --------
    common_start : datetime.date
        Fecha más reciente entre las primeras fechas válidas de todos los tickers.
        Marca el inicio común para normalizar el dataset.

    table : pd.DataFrame
        Tabla resumen generada por `first_valid_dates_by_ticker(df)`,
        con las columnas:
        ['ticker', 'first_valid_date', 'first_date', 'last_date',
         'n_total', 'n_nonnull', 'coverage_ratio'].

    Excepciones:
    ------------
    ValueError
        Si strict=True y existen tickers sin ningún dato no nulo.

    Notas:
    ------
    • Si un ticker no tiene registros válidos, puede excluirse con strict=False.
    • Este paso es crítico para sincronizar series con distintos inicios,
      como ocurre entre acciones/ETFs y criptomonedas.
    """
    table = first_valid_dates_by_ticker(df)

    # Si un ticker no tiene 'first_valid_date' (todo NaN), manejarlo
    if table["first_valid_date"].isna().any():
        missing = table[table["first_valid_date"].isna()]["ticker"].tolist()
        msg = f"Tickers sin datos no nulos: {missing}"
        if strict:
            raise ValueError(msg)
        # En modo no estricto, los excluimos
        table = table.dropna(subset=["first_valid_date"])

    # Fecha de inicio común (la más tardía entre todos los tickers)
    common_start = pd.to_datetime(table["first_valid_date"]).max().date()
    return common_start, table

def make_business_index(common_start, end_date):
    """
    Paso 3 - Generamos un rango de fechas validas (lunes-viernes)

    Objetivo
    --------
    - Construir un rango de fechas con frecuencia de días hábiles
      (business days) entre una fecha de inicio común y una fecha de corte.
    - Este índice servirá para reindexar todas las series financieras y
      normalizarlas a un mismo calendario de negociación.

    Parámetros
    ----------
    common_start : str o datetime.date
        Fecha de inicio común, normalmente determinada por el ticker con
        el inicio de datos más reciente. Puede ser string en formato
        'YYYY-MM-DD' o un objeto datetime.date.

    end_date : str o datetime.date
        Fecha de corte fija que marca el final del análisis. Puede ser string
        en formato 'YYYY-MM-DD' o un objeto datetime.date.

    Retorna
    -------
    business_index : numpy.ndarray de tipo date
        Lista de fechas (naive, sin zona horaria) correspondientes a todos los
        días hábiles (lunes–viernes) entre common_start y end_date.

    Notas
    -----
    • Usa la frecuencia 'B' de pandas, que excluye fines de semana automáticamente.
    • No excluye feriados específicos de cada mercado; si se desea ese nivel de
      precisión, se debería usar `CustomBusinessDay` con un calendario bursátil.
    • Este paso permite alinear ETFs/acciones (que no cotizan en fines de semana)
      con criptomonedas, generando una base común de fechas hábiles.
    """
    idx = pd.date_range(start=common_start, end=end_date, freq="B")
    return idx.date

def reindex_all_to_business(df: pd.DataFrame, business_index) -> pd.DataFrame:
    """
    Paso 4 - Vamos a reindexar todas las series al calendario hábil
    común (lunes–viernes)

    Objetivo
    --------
    - Alinear todos los tickers a un mismo calendario de días hábiles
    (business days), usando el índice común previamente construido.
    - Mantener el dataset en formato "tidy" y reinyectar los metadatos
    por ticker (asset_class y currency) tras el reindexado.

    Entradas
    --------
    df : pd.DataFrame
        DataFrame en formato tidy con columnas mínimas:
        ['date', 'ticker', 'asset_class', 'close', 'currency'].
        • 'date' puede ser datetime64 o date (se normaliza internamente).
        • 'ticker' identifica el activo.
        • 'close' puede contener NaN (especialmente en fines de semana/feriados).

    business_index : iterable de fechas (p. ej., numpy.ndarray / list[date])
        Índice de fechas hábiles (lunes–viernes) generado por `make_business_index`.
        Este calendario será el que usemos para reindexar todas las series.

    Salida
    ------
    pd.DataFrame
        DataFrame tidy reindexado al calendario común, con columnas:
        ['date', 'ticker', 'asset_class', 'close', 'currency'].
        • Las fechas fuera del calendario se eliminan.
        • Las fechas agregadas por el reindexado quedan con NaN en aquellos
          tickers que no operan en esos días (antes de la limpieza final).

    Detalles de implementación
    --------------------------
    1) Normaliza 'date' a objeto date (naive).
    2) Extrae y conserva metadatos únicos por ticker: asset_class, currency.
    3) Convierte el dataset a formato ancho (pivot: filas=fecha, columnas=ticker).
    4) Reindexa el ancho al business_index (frecuencia hábil).
    5) Vuelve a formato tidy (melt) y ordena por ['ticker', 'date'].
    6) Reinyecta los metadatos por ticker (asset_class, currency).

    Notas
    -----
    • Este paso NO elimina NaN: solo armoniza calendarios. La eliminación de
      días con faltantes se hace en el paso posterior (`drop_days_with_any_nan`).
    • Si se requieren feriados específicos por mercado, el business_index debe
      construirse con un calendario bursátil custom (p. ej., `CustomBusinessDay`).
    • Mantener tidy facilita comparaciones y posteriores operaciones de groupby.

    Ejemplo
    -------
    >>> bidx = make_business_index(common_start, "2025-09-08")
    >>> df_b = reindex_all_to_business(market_df, bidx)
    >>> df_b.head()
    """
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.date

    # Pivot a formato ancho
    wide = tmp.pivot(index="date", columns="ticker", values="close")

    # Asegurar que el índice tenga nombre para que reset_index cree 'date'
    wide.index.name = "date"

    # Eliminar días con algún NaN
    wide_clean = wide.dropna()

    # Volver a tidy (manejo robusto del nombre de la columna fecha)
    df_reset = wide_clean.reset_index()
    if "index" in df_reset.columns and "date" not in df_reset.columns:
        df_reset = df_reset.rename(columns={"index": "date"})

    tidy = (
        df_reset
        .melt(id_vars="date", var_name="ticker", value_name="close")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # Reinyectar metadatos desde el original
    meta = (
        df.dropna(subset=["ticker"])
          .groupby("ticker", as_index=False)
          .agg(asset_class=("asset_class", "first"),
               currency=("currency", "first"))
          .set_index("ticker")
    )
    tidy["asset_class"] = tidy["ticker"].map(meta["asset_class"])
    tidy["currency"]    = tidy["ticker"].map(meta["currency"])

    return tidy[["date", "ticker", "asset_class", "close", "currency"]]

def drop_days_with_any_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 5 - Eliminar filas con NaN en cualquier ticker

    Elimina del dataset todos los días en los que algún ticker tenga valores nulos
    en el precio de cierre, devolviendo un DataFrame limpio y balanceado.

    Propósito
    ---------
    - Normalizar la longitud de las series temporales de todos los tickers.
    - Asegurar que cada fila del dataset final represente un día en el cual
      todos los activos tienen precio válido (sin NaN).
    - Preparar un conjunto de datos homogéneo para análisis comparativos
      y modelado de series de tiempo.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame en formato tidy con columnas mínimas:
        - 'date'       : fecha de negociación (datetime.date o string ISO)
        - 'ticker'     : símbolo del activo
        - 'asset_class': tipo de activo (ETF, Stock, Crypto, ...)
        - 'close'      : precio de cierre ajustado
        - 'currency'   : moneda principal de cotización

    Retorna
    -------
    pd.DataFrame
        Nuevo DataFrame tidy con:
        - Fechas limitadas únicamente a los días con datos completos.
        - Columnas: ['date', 'ticker', 'asset_class', 'close', 'currency'].
        - Sin valores nulos en 'close'.

    Notas
    -----
    - La función pivota temporalmente los datos a formato ancho
      (tickers en columnas), elimina las filas con cualquier NaN y vuelve a
      deshacer el pivote a formato tidy.
    - Se reinyectan metadatos (`asset_class` y `currency`) desde el dataset
      original para preservar información.
    - Esta limpieza garantiza series perfectamente alineadas entre activos,
      aunque implica pérdida de información (elimina fines de semana de criptos).

    Ejemplos
    --------
    >>> df_clean = drop_days_with_any_nan(market_df)
    >>> df_clean.head()
           date ticker asset_class   close currency
    0  2015-01-02   VOO        ETF  193.98      USD
    1  2015-01-02   QQQ        ETF  102.73      USD
    2  2015-01-02  TSLA      Stock   42.33      USD
    3  2015-01-02     V      Stock   65.50      USD
    4  2015-01-02 BTC-USD    Crypto 315.23      USD
    """
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.date

    # Pivotar a formato ancho (tickers en columnas, fechas en índice)
    wide = tmp.pivot(index="date", columns="ticker", values="close")

    # Eliminar días con al menos 1 valor nulo
    wide_clean = wide.dropna()

    # Volver a formato tidy
    tidy = (
        wide_clean.reset_index(names="date")
                  .melt(id_vars="date", var_name="ticker", value_name="close")
                  .sort_values(["ticker", "date"])
                  .reset_index(drop=True)
    )

    # Reinyectar metadatos desde el DataFrame original
    meta = (
        df.dropna(subset=["ticker"])
          .groupby("ticker", as_index=False)
          .agg(asset_class=("asset_class", "first"),
               currency=("currency", "first"))
          .set_index("ticker")
    )
    tidy["asset_class"] = tidy["ticker"].map(meta["asset_class"])
    tidy["currency"] = tidy["ticker"].map(meta["currency"])

    return tidy[["date", "ticker", "asset_class", "close", "currency"]]

def normalize_market_timeseries(
    df: pd.DataFrame,
    end_date,
    strict: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Paso 6 - Normalizar un dataset de series de tiempo de activos

    Normaliza un dataset de series de tiempo financieras, alineando todos
    los tickers a un calendario común de días hábiles y eliminando fechas
    inconsistentes.

    Propósito
    ---------
    - Construir un dataset homogéneo en el que todos los activos tengan la misma
      longitud temporal y estén alineados día a día.
    - Establecer una fecha de inicio común para todos los tickers, evitando
      sesgos
      por activos con históricos más cortos.
    - Reindexar los datos a un calendario de días hábiles (lunes–viernes).
    - Eliminar días incompletos (con algún valor nulo) para garantizar
      consistencia.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame en formato tidy con las columnas mínimas:
        - 'date'       : fecha
        - 'ticker'     : símbolo del activo
        - 'asset_class': tipo de activo (ETF, Stock, Crypto, …)
        - 'close'      : precio de cierre ajustado
        - 'currency'   : moneda de cotización
    end_date : str | datetime.date | datetime.datetime
        Fecha de corte (último día a considerar en el calendario común).
        Ej.: '2025-09-05'.
    strict : bool, opcional (default=True)
        - True  → si un ticker no tiene datos válidos, lanza un error.
        - False → excluye los tickers sin datos válidos.

    Retorna
    -------
    Tuple[pd.DataFrame, dict]
        - df_clean : DataFrame tidy normalizado, con todos los tickers alineados
          al mismo calendario y sin días con NaN.
        - meta : diccionario con información de auditoría:
            * 'common_start'    : fecha inicial común detectada.
            * 'end_date'        : fecha de corte usada.
            * 'first_valid_table': tabla con primeras fechas válidas por ticker.
            * 'coverage_table'   : tabla comparativa
              (filas antes y después de limpiar).

    Notas
    -----
    - Este proceso elimina fines de semana y festivos de criptomonedas,
      para alinear con el calendario de acciones/ETFs (lunes–viernes).
    - La proporción de datos retenidos por ticker se guarda en `coverage_table`,
      lo que permite evaluar la pérdida de información.
    - Si se requiere trabajar con criptos 24/7, este método puede no ser adecuado.

    Ejemplos
    --------
    >>> df_norm, meta = normalize_market_timeseries(market_df, "2025-09-05")
    >>> df_norm.head()
           date   ticker asset_class   close currency
    0  2015-01-02     VOO         ETF  193.98      USD
    1  2015-01-02     QQQ         ETF  102.73      USD
    2  2015-01-02  TSLA      Stock   42.33      USD
    3  2015-01-02     V      Stock   65.50      USD
    4  2015-01-02 BTC-USD    Crypto 315.23      USD

    >>> meta["coverage_table"]
      ticker  n_rows_before  n_rows_after  retained_ratio
    0   BTC-USD          3921          2710           0.69
    1     QQQ           2710          2710           1.00
    2    TSLA           2710          2710           1.00
    """
    # 1) Calcular fecha de inicio común en base a los datos válidos
    common_start, table = common_start_date(df, strict=strict)

    # 2) Generar calendario de días hábiles desde common_start hasta end_date
    bindex = make_business_index(common_start, end_date)

    # 3) Reindexar todas las series al calendario común
    df_b = reindex_all_to_business(df, bindex)

    # 4) Eliminar días con NaN en cualquier ticker
    df_clean = drop_days_with_any_nan(df_b)

    # --- Métricas de auditoría ---
    # Conteos antes de la limpieza
    before_counts = (
        df.groupby("ticker")["date"].count().reset_index(name="n_rows_before")
    )
    # Conteos después de la limpieza
    after_counts = (
        df_clean.groupby("ticker")["date"].count().reset_index(name="n_rows_after")
    )
    # Tabla comparativa
    coverage_tbl = before_counts.merge(after_counts, on="ticker", how="left")
    coverage_tbl["retained_ratio"] = (
        coverage_tbl["n_rows_after"] / coverage_tbl["n_rows_before"]
    )

    meta = {
        "common_start": common_start,
        "end_date": pd.to_datetime(end_date).date(),
        "first_valid_table": table.sort_values("first_valid_date"),
        "coverage_table": coverage_tbl.sort_values("ticker")
    }
    return df_clean, meta
