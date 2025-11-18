import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display_html
import warnings
warnings.filterwarnings("ignore")
from typing import Tuple, Dict


def compare_summaries(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """
    Construye tablas comparativas entre dataset crudo (antes) y normalizado (después).

    Propósito
    ---------
    - Ver rápidamente cuánto se pierde/retiene por ticker.
    - Ver rangos temporales antes y después.
    - Ver las medias, desviaciones estándar y varianzas de los precios.
    - Ver los valores nulos (NaN) por ticker.
    - Dejar todo en una estructura dict para mostrar por partes.

    Retorna
    -------
    dict con:
      - range_raw      : rango (first/last) y n_rows del crudo
      - range_clean    : rango (first/last) y n_rows del limpio
    """
    range_raw = (
        df_raw.groupby("ticker")
        .agg(first_date=("date", "min"),
             last_date=("date", "max"),
             n_rows=("date", "size"),
             n_missing=("close", lambda x: x.isna().sum()),
             mean = ("close", "mean"),
             var =  ("close", "var"),
             std  = ("close", "std"),
            asset_class = ("asset_class", "first"),
            currency = ("currency", "first"))
        .reset_index()
        .sort_values("ticker")
    )

    range_clean = (
        df_clean.groupby("ticker")
        .agg(first_date=("date", "min"),
             last_date=("date", "max"),
             n_rows=("date", "size"),
             n_missing=("close", lambda x: x.isna().sum()),
             mean = ("close", "mean"),
             var =  ("close", "var"),
             std  = ("close", "std"),
             asset_class = ("asset_class", "first"),
            currency = ("currency", "first"))
        .reset_index()
        .sort_values("ticker")
    )

    return {
        "range_raw": range_raw,
        "range_clean": range_clean
    }

def plot_before_after_timeseries(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """
    Dibuja una grilla con 2 columnas por cada ticker:
      - Columna izquierda: serie cruda (antes de limpiar)
      - Columna derecha : serie normalizada (después de limpiar)

    Notas
    -----
    - Mantiene el mismo límite de Y por ticker (min/max de ambas series)
      para una comparación justa.
    - Comparte eje X por fila (cada ticker).
    """
    tickers = sorted(df_raw["ticker"].unique())
    n = len(tickers)
    ncols = 2
    nrows = n

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(16, 2.8 * nrows),
                             sharex=False, sharey=False)

    if nrows == 1:
        axes = np.array([axes])  # homogeneizar forma cuando hay un solo ticker

    for r, tk in enumerate(tickers):
        a_raw = axes[r, 0]
        a_cln = axes[r, 1]

        s_raw = df_raw[df_raw["ticker"] == tk][["date", "close"]].sort_values("date")
        s_cln = df_clean[df_clean["ticker"] == tk][["date", "close"]].sort_values("date")

        # Límites comunes por ticker
        ymin = min(s_raw["close"].min(), s_cln["close"].min())
        ymax = max(s_raw["close"].max(), s_cln["close"].max())

        # Izquierda: antes
        a_raw.plot(s_raw["date"], s_raw["close"], lw=1.3)
        a_raw.set_title(f"{tk} — Antes", fontsize=11)
        a_raw.set_ylim(ymin, ymax)
        a_raw.grid(True, alpha=0.3)

        # Derecha: después
        a_cln.plot(s_cln["date"], s_cln["close"], lw=1.3)
        a_cln.set_title(f"{tk} — Después", fontsize=11)
        a_cln.set_ylim(ymin, ymax)
        a_cln.grid(True, alpha=0.3)

        # Etiqueta Y solo en la izquierda
        a_raw.set_ylabel("Precio cierre")

    # Títulos de columnas
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    plt.suptitle("Comparativa antes vs. después de la limpieza por ticker", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_before_after_corr(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """
    Muestra dos heatmaps lado a lado:
      - Izquierda: correlación de precios (crudo)
      - Derecha  : correlación de precios (normalizado)
    """
    def to_wide(df):
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        wide = tmp.pivot(index="date", columns="ticker", values="close")
        return wide

    wide_raw = to_wide(df_raw)
    wide_cln = to_wide(df_clean)

    corr_raw = wide_raw.corr(min_periods=1)
    corr_cln = wide_cln.corr(min_periods=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(corr_raw, ax=axes[0], annot=True, fmt=".2f",
                cmap="coolwarm", center=0, cbar_kws={"shrink": .8})
    axes[0].set_title("Correlación — Antes (crudo)", fontsize=12)

    sns.heatmap(corr_cln, ax=axes[1], annot=True, fmt=".2f",
                cmap="coolwarm", center=0, cbar_kws={"shrink": .8})
    axes[1].set_title("Correlación — Después (normalizado)", fontsize=12)

    plt.suptitle("Heatmaps de correlación — antes vs. después", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_side_by_side(df_before: pd.DataFrame, df_after: pd.DataFrame,
                      title_before="Antes de limpieza", title_after="Después de limpieza"):
    """
    Muestra dos DataFrames en paralelo (lado a lado) con títulos personalizados.
    Útil para comparar rangos, coberturas o métricas antes y después del procesamiento.
    """
    html_before = f"<h3 style='text-align:center'>{title_before}</h3>" + df_before.to_html(index=False)
    html_after  = f"<h3 style='text-align:center'>{title_after}</h3>" + df_after.to_html(index=False)

    display_html(
        f"<div style='display:flex; justify-content:space-around; gap:40px;'>"
        f"<div>{html_before}</div>"
        f"<div>{html_after}</div>"
        f"</div>", raw=True
    )

def show_heads_by_ticker(df_before, df_after, n=5, tickers=None):
    """
    Usa la función show_side_by_side para mostrar comparaciones por ticker.

    Parámetros
    ----------
    df_before, df_after : DataFrames tidy con columnas ['date','ticker','close',...]
    n : int
        Número de filas a mostrar (head).
    tickers : list | None
        Lista de tickers a mostrar. Si es None, usa los comunes en ambos datasets.
    """
    # Detectamos tickers comunes
    common = sorted(set(df_before["ticker"]) & set(df_after["ticker"]))
    if tickers:
        common = [t for t in tickers if t in common]

    for tk in common:
        before_tk = (df_before[df_before["ticker"] == tk]
                     .sort_values("date")
                     .head(n)
                     .reset_index(drop=True))
        after_tk = (df_after[df_after["ticker"] == tk]
                    .sort_values("date")
                    .head(n)
                    .reset_index(drop=True))

        # Reusamos tu función existente para mostrar lado a lado
        show_side_by_side(before_tk, after_tk,
                          title_before=f"{tk} — Antes",
                          title_after=f"{tk} — Después")