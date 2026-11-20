from pathlib import Path
import pandas as pd


def load_raw_sales(raw_path: Path) -> pd.DataFrame:
    """Carrega o arquivo bruto sales.csv."""
    raise NotImplementedError


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Converte datas, padroniza status, remove duplicados e cria a coluna de receita."""
    raise NotImplementedError


def save_clean_sales(df: pd.DataFrame, output_path: Path) -> Path:
    """Salva o dataset limpo com separador ';' e decimal ','."""
    raise NotImplementedError


def load_clean_sales(clean_path: Path) -> pd.DataFrame:
    """Lê o dataset limpo usando separador ';' e decimal ','."""
    raise NotImplementedError


def prepare_sales_pipeline(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """Executa o fluxo completo de preparação e retorna o DataFrame limpo."""
    raise NotImplementedError
