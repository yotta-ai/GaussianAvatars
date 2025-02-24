import logging
import colorlog

def setup_logger(name: str) -> logging.Logger:
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(filename)s - %(levelname)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger