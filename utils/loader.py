import pandas as pd
from pathlib import Path
from .paths import DATA_DIR
import yaml
from typing import Any, Dict

from .paths import BASE_DIR, DATA_DIR

def load_csv(name: str) -> pd.DataFrame:
    """
    Carga un archivo CSV desde la carpeta DATA_DIR.
    """
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe: {path}")
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, name: str):
    """
    Guarda un DataFrame como CSV dentro de DATA_DIR.
    """
    path = DATA_DIR / name
    df.to_csv(path, index=False)
    print(f"Archivo guardado en: {path}")

def file_path(name: str) -> Path:
    """
    Devuelve la ruta absoluta de un archivo en DATA_DIR.
    """
    return DATA_DIR / name

def load_yaml(relative_path: str) -> Dict[str, Any]:
    """
    Carga un archivo YAML desde cualquier ubicación relativa al proyecto.
    Ejemplo: load_yaml("config/settings.yaml")
    """
    path = BASE_DIR / relative_path

    if not path.exists():
        raise FileNotFoundError(f"YAML no encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], relative_path: str):
    """
    Guarda un diccionario como YAML usando ruta relativa al proyecto.
    Ejemplo: save_yaml(config_dict, "config/settings.yaml")
    """
    path = BASE_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    print(f"YAML guardado en: {path}")