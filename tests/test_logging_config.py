import logging

from data_pilot.logging_config import configure_logging


def test_configure_logging_adds_only_one_handler(monkeypatch):
    root_logger = logging.getLogger()

    monkeypatch.setattr(root_logger, "handlers", [])
    monkeypatch.setattr(root_logger, "level", root_logger.level)

    configure_logging()
    configure_logging()

    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.StreamHandler)


def test_configure_logging_respects_existing_handlers(monkeypatch):
    root_logger = logging.getLogger()
    existing_handler = logging.StreamHandler()

    monkeypatch.setattr(root_logger, "handlers", [existing_handler])
    monkeypatch.setattr(root_logger, "level", root_logger.level)

    configure_logging()

    assert len(root_logger.handlers) == 1
    assert root_logger.handlers[0] is existing_handler
