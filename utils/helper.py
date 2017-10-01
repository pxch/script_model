import os
from bz2 import BZ2File
from gzip import GzipFile

import consts


def supress_fd(fd_number):
    assert fd_number in [1, 2]
    # open a null file descriptor
    null_fd = os.open(os.devnull, os.O_RDWR)
    # save the current stdout(1) / stderr(2) file descriptor
    save_fd = os.dup(fd_number)
    # put /dev/null fd on stdout(1) / stderr(2)
    os.dup2(null_fd, fd_number)

    return null_fd, save_fd


def restore_fd(fd_number, null_fd, save_fd):
    # restore stdout(1) / stderr(2) to the saved file descriptor
    os.dup2(save_fd, fd_number)
    # close the null file descriptor
    os.close(null_fd)


def get_class_name(class_type):
    return '{}.{}'.format(class_type.__module__, class_type.__name__)


def check_type(variable, class_type):
    assert isinstance(variable, class_type), \
        'expecting an instance of {}, {} found'.format(
            get_class_name(class_type), type(variable))


def convert_ontonotes_ner_tag(tag, to_corenlp=False):
    if to_corenlp:
        return consts.ontonotes_to_corenlp_mapping.get(tag, '')
    else:
        return consts.ontonotes_to_valid_mapping.get(tag, '')


def convert_corenlp_ner_tag(tag):
    return consts.corenlp_to_valid_mapping.get(tag, '')


def smart_file_handler(filename):
    if filename.endswith('bz2'):
        f = BZ2File(filename, 'r')
    elif filename.endswith('gz'):
        f = GzipFile(filename, 'r')
    else:
        f = open(filename, 'r')
    return f
