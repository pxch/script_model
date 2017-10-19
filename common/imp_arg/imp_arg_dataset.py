import pickle as pkl
import timeit
from os.path import exists, join

import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold
from tqdm import tqdm

from common.imp_arg import helper
from common.imp_arg.candidate_dict import CandidateDict
from common.imp_arg.proposition import Proposition
from config import cfg
from dataset.corenlp import read_corenlp_doc
from dataset.imp_arg import imp_arg_instances
from dataset.nltk import PTBReader, NombankReader, PropbankReader
from utils import PBToLogger, get_console_logger

log = get_console_logger()


class ImplicitArgumentDataset(object):
    def __init__(self, max_dist=2):
        # maximum distance for candidates
        self._max_dist = max_dist

        # list of all implicit argument instances
        self._instances = imp_arg_instances
        # number of all implicit argument instances
        self._num_instances = len(imp_arg_instances)

        # list of all implicit argument instances, sorted by predicate node
        self._instances_sorted = []
        # order of original instances in sorted list
        self._instance_order_list = []

        self.sort_instances()

        # number of splits in cross validation
        self._n_splits = 0
        # list of cross validation train/test instance indices in sorted list
        self._train_test_folds = []

        # list of all implicit argument propositions (processed instance)
        self._propositions = []

        # PennTreebank reader
        self._ptb_reader = None
        # Propbank Reader
        self._propbank_reader = None
        # Nombank Reader
        self._nombank_reader = None

        # dictionary of all CoreNLP documents used in the dataset
        self._corenlp_mapping = None

        # mapping between predicate nodes and nominal predicates
        self._predicate_mapping = None

        # dictionary of all candidates indexed by fileid:sentnum
        self._candidate_dict = None

    def sort_instances(self):
        log.info('Sorting all instances by their predicate node')
        self._instances_sorted = sorted(
            self._instances,
            key=lambda ins: (ins.pred_node.fileid, ins.pred_node.sentnum,
                             ins.pred_node.wordnum))

        self._instance_order_list = \
            [self._instances_sorted.index(instance)
             for instance in self._instances]

    def all_instances(self, sort=True):
        if sort:
            return self._instances_sorted
        else:
            return self._instances

    @property
    def num_instances(self):
        return self._num_instances

    @property
    def n_splits(self):
        return self._n_splits

    @n_splits.setter
    def n_splits(self, n_splits):
        assert type(n_splits) == int and n_splits > 0, \
            'number of splits must be a positive integer'
        self._n_splits = n_splits

    def create_train_test_folds(self, n_splits=10):
        log.info('Creating {} fold cross validation split'.format(n_splits))
        self.n_splits = n_splits

        self._train_test_folds = []
        instance_order_list = np.asarray(self._instance_order_list)

        kf = KFold(n_splits=self.n_splits, shuffle=False)
        for train, test in kf.split(self._instance_order_list):
            train_indices = instance_order_list[train]
            test_indices = instance_order_list[test]
            self._train_test_folds.append((train_indices, test_indices))

    def get_train_fold(self, fold_idx):
        assert 0 <= fold_idx < self.n_splits, \
            'fold index {} out of range'.format(fold_idx)
        return self._train_test_folds[fold_idx][0]

    def get_test_fold(self, fold_idx):
        assert 0 <= fold_idx < self.n_splits, \
            'fold index {} out of range'.format(fold_idx)
        return self._train_test_folds[fold_idx][1]

    @property
    def ptb_reader(self):
        if self._ptb_reader is None:
            self._ptb_reader = PTBReader()
        return self._ptb_reader

    @property
    def propbank_reader(self):
        if self._propbank_reader is None:
            self._propbank_reader = PropbankReader(indexing=True)
        return self._propbank_reader

    @property
    def nombank_reader(self):
        if self._nombank_reader is None:
            self._nombank_reader = NombankReader(indexing=True)
        return self._nombank_reader

    @property
    def corenlp_mapping(self):
        if self._corenlp_mapping is None:
            if not exists(helper.corenlp_mapping_path):
                log.info('Building CoreNLP mapping from {}'.format(
                    cfg.wsj_corenlp_root))

                self._corenlp_mapping = {}

                for instance in self.all_instances(sort=True):
                    fileid = instance.pred_node.fileid
                    if fileid not in self._corenlp_mapping:

                        path = join(cfg.wsj_corenlp_root, 'idx',
                                    helper.expand_wsj_fileid(fileid))
                        idx_mapping = []
                        with open(path, 'r') as fin:
                            for line in fin:
                                idx_mapping.append(
                                    [int(i) for i in line.split()])

                        path = join(
                            cfg.wsj_corenlp_root, 'parsed',
                            helper.expand_wsj_fileid(fileid, '.xml.bz2'))
                        doc = read_corenlp_doc(path)

                        self._corenlp_mapping[fileid] = (idx_mapping, doc)

                log.info('Saving CoreNLP mapping to {}'.format(
                    helper.corenlp_mapping_path))
                pkl.dump(self._corenlp_mapping,
                         open(helper.corenlp_mapping_path, 'w'))
            else:
                start_time = timeit.default_timer()
                log.info('Loading CoreNLP mapping from {}'.format(
                    helper.corenlp_mapping_path))
                self._corenlp_mapping = \
                    pkl.load(open(helper.corenlp_mapping_path, 'r'))
                elapsed = timeit.default_timer() - start_time
                log.info('Done in {:.3f} seconds'.format(elapsed))

        return self._corenlp_mapping

    @property
    def predicate_mapping(self):
        if self._predicate_mapping is None:
            if not exists(helper.predicate_mapping_path):
                log.info('Building predicate mapping')
                self._predicate_mapping = {}

                lemmatizer = WordNetLemmatizer()

                for instance in self.all_instances(sort=True):
                    pred_node = instance.pred_node

                    self.ptb_reader.read_file(
                        helper.expand_wsj_fileid(pred_node.fileid, '.mrg'))

                    word = self.ptb_reader.all_sents[
                        pred_node.sentnum][pred_node.wordnum]

                    n_pred = lemmatizer.lemmatize(word.lower(), pos='n')

                    if n_pred not in helper.nominal_predicate_mapping:
                        for subword in n_pred.split('-'):
                            if subword in helper.nominal_predicate_mapping:
                                n_pred = subword
                                break

                    assert n_pred in helper.nominal_predicate_mapping, \
                        'unexpected nominal predicate: {}'.format(n_pred)
                    assert str(pred_node) not in self._predicate_mapping, \
                        'pred_node {} already found'.format(pred_node)
                    self._predicate_mapping[str(pred_node)] = n_pred

                log.info('Saving predicate mapping to {}'.format(
                    helper.predicate_mapping_path))
                pkl.dump(self._predicate_mapping,
                         open(helper.predicate_mapping_path, 'w'))
            else:
                log.info('Loading predicate mapping from {}'.format(
                    helper.predicate_mapping_path))
                self._predicate_mapping = \
                    pkl.load(open(helper.predicate_mapping_path, 'r'))
        return self._predicate_mapping

    @property
    def candidate_dict(self):
        return self._candidate_dict

    def build_candidate_dict(self, save_path=None):
        if self._candidate_dict:
            log.warning('Found existing candidate dict')
            return

        log.info('Building candidate dict from Propbank and Nombank')

        self._candidate_dict = CandidateDict(
            propbank_reader=self.propbank_reader,
            nombank_reader=self.nombank_reader,
            max_dist=self._max_dist)

        for instance in tqdm(self._instances_sorted, file=PBToLogger(log),
                             desc='Processed', ncols=100, mininterval=5):
            pred_node = instance.pred_node
            self._candidate_dict.add_candidates(
                fileid=pred_node.fileid, sentnum=pred_node.sentnum)

        if save_path is not None:
            self._candidate_dict.save(save_path)

    def load_candidate_dict(self, path=helper.candidate_dict_path):
        self._candidate_dict = CandidateDict.load(
            path, max_dist=self._max_dist)

    @property
    def propositions(self):
        return self._propositions

    def build_propositions(self, save_path=None, filter_conflict=False,
                           filter_incorporated=False, verbose=False):
        if len(self._propositions) > 0:
            log.warning('Found {} existing propositions'.format(
                len(self._propositions)))
            return

        log.info('Building propositions from instances')
        for instance in self.all_instances(sort=True):
            proposition = Proposition.build(instance)
            proposition.set_pred(self.predicate_mapping)

            self._propositions.append(proposition)

        nombank_reader = self.nombank_reader
        log.info('Check explicit arguments with Nombank instances')
        for proposition in self._propositions:
            nombank_instance = nombank_reader.search_by_pointer(
                proposition.pred_pointer)
            proposition.check_exp_args(
                nombank_instance,
                filter_conflict=filter_conflict,
                verbose=verbose)

        ptb_reader = self.ptb_reader
        log.info('Parse subtrees for all explicit / implicit arguments')
        for proposition in tqdm(self._propositions, file=PBToLogger(log),
                                desc='Processed', ncols=100, mininterval=5):
            proposition.parse_arg_subtrees(ptb_reader)

        if filter_incorporated:
            log.info('Filter incorporated arguments')
            for proposition in self._propositions:
                proposition.filter_incorporated_argument(verbose=verbose)

        if save_path is not None:
            log.info('Saving all propositions to {}'.format(
                helper.propositions_path))
            pkl.dump(self._propositions, open(helper.propositions_path, 'w'))

    def load_propositions(self, path=helper.propositions_path):
        log.info('Loading all propositions from {}'.format(path))
        start_time = timeit.default_timer()
        self._propositions = pkl.load(open(helper.propositions_path, 'r'))
        elapsed = timeit.default_timer() - start_time
        log.info('Done in {:.3f} seconds'.format(elapsed))

    def parse_corenlp(self, save=False):
        log.info('Parsing CoreNLP info for arguments of all propositions')
        for proposition in self._propositions:
            idx_mapping, doc = self.corenlp_mapping[
                proposition.pred_pointer.fileid]
            proposition.parse_arg_corenlp(doc, idx_mapping)

        self._candidate_dict.parse_corenlp(self.corenlp_mapping)

        if save:
            log.info('Saving all propositions with CoreNLP information '
                     'to {}'.format(helper.propositions_w_corenlp_path))
            pkl.dump(self._propositions,
                     open(helper.propositions_w_corenlp_path, 'w'))

            self._candidate_dict.save(helper.candidate_dict_w_corenlp_path)

    def add_candidates(self):
        log.info('Adding candidates to propositions')
        for proposition in self.propositions:
            for candidate in self.candidate_dict.get_candidates(
                    proposition.pred_pointer):
                proposition.add_candidate(candidate)
