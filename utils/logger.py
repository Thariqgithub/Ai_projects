import logging
import sys
from logging.handlers import RotatingFileHandler
from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — force UTF-8 so Unicode chars work on Windows cp1252 terminals
    console_handler = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
        if sys.platform == "win32" else sys.stdout
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation — always UTF-8
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
