import logging


def get_console_logger(level='info'):
    # Prepare a logger
    log = logging.getLogger()
    if level == 'debug':
        log.setLevel(logging.DEBUG)
    elif level == 'info':
        log.setLevel(logging.INFO)
    elif level == 'warning':
        log.setLevel(logging.WARNING)
    elif level == 'error':
        log.setLevel(logging.ERROR)
    elif level == 'critical':
        log.setLevel(logging.CRITICAL)
    else:
        log.setLevel(logging.NOTSET)

    if not log.handlers:
        # Just log to the console
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        # Put a timestamp on everything
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)

        log.addHandler(sh)

    return log
