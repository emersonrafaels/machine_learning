"""
Este módulo busca o conjunto de dados Wholesale Customers do UCI Machine Learning Repository.
Ele fornece os recursos (features) e os alvos (targets) do conjunto de dados como DataFrames do pandas.
"""

import sys
from pathlib import Path

from ucimlrepo import fetch_ucirepo

base_dir = Path(__file__).resolve().parents[3]

sys.path.append(str(base_dir))

from src.utils.data.data_functions import export_data
from src.config.config_logger import logger
from src.config.config_dynaconf import get_settings

# Inicializando as configurações
settings = get_settings()


def buscar_dados_wholesale_customers():
    """
    Busca o conjunto de dados Wholesale Customers do UCI Machine Learning Repository.

    Retorna:
        tuple: Uma tupla contendo os recursos (X) e os alvos (y) como DataFrames do pandas.
    """

    logger.info(
        "Buscando o conjunto de dados Wholesale Customers do UCI ML Repository..."
    )

    # Esse é um dataset não supervisionado, então retornamos apenas X
    y = None

    # Busca o conjunto de dados
    wholesale_customers = fetch_ucirepo(id=292)

    # Extrai os dados
    X = wholesale_customers.data.features.copy()
    X["Region"] = wholesale_customers.data.targets

    logger.info("Conjunto de dados obtido com sucesso!")

    return X, y


def imprimir_metadados(df, n=5):
    """
    Imprime os metadados e as informações das variáveis do conjunto de dados Wholesale Customers.
    """

    # Imprime os metadados
    print(df.head(n))


def main_get_data():
    """
    Função principal para buscar e exportar os dados do Wholesale Customers.
    """

    # Buscando os dados
    df = buscar_dados_wholesale_customers()

    # Visualizando os dados
    imprimir_metadados(df, n=3)

    # Exporta os dados para o diretório data
    export_data(df, Path(base_dir, "data/wholesale_datasets.csv"))

    return df


if __name__ == "__main__":

    # Pipeline de obtenção dos dados
    _ = main_get_data()
