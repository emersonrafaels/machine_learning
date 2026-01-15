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

def main_model():
    
    """
    
    Função principal para executar a pipeline de segmentação de clientes.
    
    """
    
    # Configurações possiveis para encontrar o número ideal de clusters
    find_optimal_clusters_options = ["silhouette", "elbow"]
    find_optimal_clusters_method = find_optimal_clusters_options[1]
    
    # Lista de colunas a serem descartadas
    list_columns_to_drop = ["Channel", "Region"]
    
    # Número de clusters (defina como None para encontrar automaticamente)
    k_clusters = 3
    
    # Executa a pipeline completa de ciência de dados para segmentação de clientes.
    run_pipeline(list_columns_to_drop=list_columns_to_drop, 
                 k_clusters=k_clusters,
                 find_optimal_clusters_method=find_optimal_clusters_method)

if __name__ == "__main__":
    main_model()
