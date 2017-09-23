import platform
import socket


class DefaultConfig(object):
    # root directory for all corpora
    corpus_root = '/Users/pengxiang/corpora/'


class CondorConfig(DefaultConfig):
    corpus_root = '/scratch/cluster/pxcheng/corpora/'


class MaverickConfig(DefaultConfig):
    corpus_root = '/work/03155/pxcheng/maverick/corpora/'


def get_config():
    system_name = platform.system()

    # local MacBook
    if system_name == 'Darwin':
        return DefaultConfig()

    if system_name == 'Linux':
        host_name = socket.getfqdn()

        # UTCS Condor cluster
        if 'cs.utexas.edu' in host_name:
            return CondorConfig()

        # TACC Maverick cluster
        if 'maverick.tacc.utexas.edu' in host_name:
            return MaverickConfig()

    raise RuntimeError('Unrecognized platform')
