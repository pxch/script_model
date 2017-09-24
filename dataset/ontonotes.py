import timeit

import on
from on.common.util import FancyConfigParser

from config import cfg
from utils import consts, supress_fd, restore_fd


def get_default_ontonotes_config():
    on_cfg = FancyConfigParser()
    on_cfg.add_section('corpus')
    on_cfg.set('corpus', '__name__', 'corpus')
    on_cfg.set('corpus', 'granularity', 'source')
    on_cfg.set('corpus', 'banks', 'parse coref name')
    on_cfg.set('corpus', 'wsd-indexing', 'word')
    on_cfg.set('corpus', 'name-indexing', 'word')
    return on_cfg


def load_ontonotes(corpus):
    assert corpus in consts.valid_ontonotes_corpus, \
        'ontonotes corpora can only be one of {}'.format(
            consts.valid_ontonotes_corpus)

    print '\nReading Ontonotes corpus {} from {}'.format(
        corpus, cfg.ontonotes_root)

    on_cfg = get_default_ontonotes_config()
    on_cfg.set('corpus', 'data_in', cfg.ontonotes_root)
    on_cfg.set('corpus', 'load', corpus)

    start_time = timeit.default_timer()

    # suppress stderr, as the following commands print too much useless info
    null_fd, save_fd = supress_fd(2)

    a_ontonotes = on.ontonotes(on_cfg)

    assert len(a_ontonotes) == 1
    subcorpus = a_ontonotes[0]

    # restore stderr
    restore_fd(2, null_fd, save_fd)

    elapsed = timeit.default_timer() - start_time
    print '\tDone in {:.3f} seconds'.format(elapsed)

    print '\tFound {} files with extensions {}'.format(
        len(subcorpus['document']), on_cfg.get('corpus', 'banks').split())

    return subcorpus
