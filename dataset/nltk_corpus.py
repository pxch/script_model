import timeit
from collections import defaultdict

from nltk.corpus import BracketParseCorpusReader
from nltk.corpus import NombankCorpusReader
from nltk.corpus import PropbankCorpusReader
from nltk.data import FileSystemPathPointer

from config import cfg

wsj_treebank = BracketParseCorpusReader(
    root=cfg.wsj_root,
    fileids=cfg.wsj_file_pattern,
    tagset='wsj',
    encoding='ascii'
)


def fileid_xform_function(fileid):
    # result = re.sub(r'^wsj/', '', fileid)
    # return result
    return fileid[4:]


propbank = PropbankCorpusReader(
    root=FileSystemPathPointer(cfg.propbank_root),
    propfile=cfg.propbank_file,
    framefiles=cfg.frame_file_pattern,
    verbsfile=cfg.propbank_verbs_file,
    parse_fileid_xform=fileid_xform_function,
    parse_corpus=wsj_treebank
)

nombank = NombankCorpusReader(
    root=FileSystemPathPointer(cfg.nombank_root),
    nomfile=cfg.nombank_file,
    framefiles=cfg.frame_file_pattern,
    nounsfile=cfg.nombank_nouns_file,
    parse_fileid_xform=fileid_xform_function,
    parse_corpus=wsj_treebank
)


class PTBReader(object):
    def __init__(self):
        print '\nBuilding PTBReader from {}'.format(cfg.wsj_root)
        self.treebank = wsj_treebank
        print '\tFound {} files'.format(len(self.treebank.fileids()))

        self.all_sents = []
        self.all_tagged_sents = []
        self.all_parsed_sents = []
        self.treebank_fileid = ''

    def read_file(self, treebank_fileid):
        if treebank_fileid != self.treebank_fileid:
            self.all_sents = self.treebank.sents(fileids=treebank_fileid)
            self.all_tagged_sents = \
                self.treebank.tagged_sents(fileids=treebank_fileid)
            self.all_parsed_sents = \
                self.treebank.parsed_sents(fileids=treebank_fileid)
            self.treebank_fileid = treebank_fileid


class SemanticCorpusReader(object):
    def __init__(self, instances, indexing=False):
        self.instances = instances
        self.num_instances = len(self.instances)
        print '\tFound {} instances'.format(self.num_instances)

        self.instances_by_fileid = defaultdict(list)
        if indexing:
            self.build_index()

    def build_index(self):
        print '\tBuilding index by fileid'
        start_time = timeit.default_timer()
        for instance in self.instances:
            fileid = self.convert_fileid(instance.fileid)
            self.instances_by_fileid[fileid].append(instance)
        elapsed = timeit.default_timer() - start_time
        print '\tDone in {:.3f} seconds'.format(elapsed)

    @staticmethod
    def convert_fileid(fileid):
        # result = re.sub(r'^\d\d/', '', fileid)
        # result = re.sub(r'\.mrg$', '', result)
        # return result
        return fileid[3:11]

    def search_by_fileid(self, fileid):
        return self.instances_by_fileid.get(fileid, [])

    def search_by_pointer(self, pointer):
        for instance in self.search_by_fileid(pointer.fileid):
            if instance.sentnum == pointer.sentnum \
                    and instance.wordnum == pointer.tree_pointer.wordnum:
                return instance
        return None


class PropbankReader(SemanticCorpusReader):
    def __init__(self, indexing=False):
        print '\nBuilding PropbankReader from {}/{}'.format(
            cfg.propbank_root, cfg.propbank_file)
        super(PropbankReader, self).__init__(
            propbank.instances(), indexing=indexing)


class NombankReader(SemanticCorpusReader):
    def __init__(self, indexing=False):
        print '\nBuilding NombankReader from {}/{}'.format(
            cfg.nombank_root, cfg.nombank_file)
        super(NombankReader, self).__init__(
            nombank.instances(), indexing=indexing)
