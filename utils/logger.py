import io
import logging

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


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
        sh.setFormatter(log_formatter)

        log.addHandler(sh)

    return log


def add_file_handler(logger, log_path):
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(log_formatter)
    logger.addHandler(fileHandler)


class PBToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(PBToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


log = get_console_logger()

pb_log = PBToLogger(log)
