import logging
from logging import shutdown

from logging.handlers import MemoryHandler

def create_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create formatter
    # formatter = logging.Formatter('%(asctime)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    # add memoryhandler to logger
    fh = logging.FileHandler(filename=filename, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # mh = MemoryHandler(capacity=100, flushLevel=logging.ERROR, target=fh)
    # logger.addHandler(mh)
    logger.addHandler(fh)

    return logger


def killLoggers(logger):
    for h in logger.handlers:
        logger.removeHandler(h)
    shutdown()
    return