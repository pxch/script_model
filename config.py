import platform
import socket
from os.path import join


class DefaultConfig(object):
    # root directory for all corpora
    corpus_root = '/Users/pengxiang/corpora/'

    # path to Penn Treebank WSJ corpus (relative to corpus_root)
    wsj_path = 'penn-treebank-rel3/parsed/mrg/wsj'
    # file pattern to read PTB data from WSJ corpus
    wsj_file_pattern = '\d\d/wsj_.*\.mrg'

    @property
    def wsj_root(self):
        return join(self.corpus_root, self.wsj_path)

    # path to Propbank corpus (relative to corpus_root)
    propbank_path = 'propbank-LDC2004T14/data'
    # file name of propositions in Propbank corpus
    propbank_file = 'prop.txt'
    # file name of verb list in Propbank corpus
    propbank_verbs_file = 'verbs.txt'

    @property
    def propbank_root(self):
        return join(self.corpus_root, self.propbank_path)

    # path to Nombank corpus (relative to corpus_root)
    nombank_path = 'nombank.1.0'
    # file name of propositions in Nombank corpus
    nombank_file = 'nombank.1.0_sorted'
    # file name of noun list in Nombank corpus
    nombank_nouns_file = 'nombank.1.0.words'

    @property
    def nombank_root(self):
        return join(self.corpus_root, self.nombank_path)

    # file pattern to read frame data from Propbank/Nombank corpus
    frame_file_pattern = 'frames/.*\.xml'


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
