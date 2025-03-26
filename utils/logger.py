import os
import logging
from datetime import datetime


def init_logger(log_path, log_file=None, log_level=logging.NOTSET):
    """
    Initialize the logger
    Args:
        log_path: str, the path to the directory where the logs will be saved
        log_file: str, the name of the log file
        log_level: int, the level of logging to use
    Returns:
        logger: logging.Logger, the logger object
    """
    if log_file is None:
        log_file = datetime.now().strftime("%d%b%Y_%H-%M-%S.log")

    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger
