"""日志配置"""
from __future__ import annotations

import logging


def setup_logging(title: str = "app", log_level: str = "INFO") -> None:
    numeric = getattr(logging, log_level.upper(), logging.INFO)

    class AppTitleFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.app_title = title
            return True

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(app_title)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.addFilter(AppTitleFilter())

    root = logging.getLogger()
    root.setLevel(numeric)
    root.addHandler(handler)

    if numeric > logging.DEBUG:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
