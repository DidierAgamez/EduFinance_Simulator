from pathlib import Path

def get_project_root():
    """
    Devuelve la ruta raíz del proyecto, independientemente
    de desde dónde se ejecute el script o el notebook.
    """
    # Path(__file__) → ruta del archivo paths.py
    # .resolve().parent.parent → subimos 2 niveles para llegar a la raíz del proyecto
    return Path(__file__).resolve().parent.parent

# Raíz del proyecto
BASE_DIR = get_project_root()

# Carpetas del proyecto
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"


# Crear carpetas si no existen
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
