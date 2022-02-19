import logging


def initialize_logger(format='%(asctime)s %(processName)-10s %(name)s %(levelname)-1s %(message)s'):
    root = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(format)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    root.setLevel(logging.INFO)
