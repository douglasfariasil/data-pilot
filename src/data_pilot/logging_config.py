"""Configuração central de logging da aplicação."""

import logging
from contextvars import ContextVar

request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Retorna o identificador da consulta atual."""
    return request_id_context.get()


def configure_logging() -> None:
    """Configura o logging sem adicionar handlers duplicados."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Respeita handlers já configurados pelo Streamlit ou pelo ambiente.
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
