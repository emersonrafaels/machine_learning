import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

# Resolve o diretório base do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Inclui a raiz no sys.path para permitir imports do src/
sys.path.append(str(BASE_DIR))

from src.config.config_logger import logger
from src.config.config_dynaconf import get_settings
from src.pipeline import run_pipeline

# Inicializa configurações
settings = get_settings()


if __name__ == "__main__":
    run_pipeline()
