"""日志配置"""
from __future__ import annotations

import logging


def setup_logging(title: str = "app", log_level: str = "INFO") -> logging.Logger:
    numeric = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(title)
    logger.setLevel(numeric)

    if numeric > logging.DEBUG:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)

    return logger
