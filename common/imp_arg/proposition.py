from collections import defaultdict
from copy import deepcopy

from common.corenlp import Document
from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.nombank import NombankSplitTreePointer

from common.imp_arg import helper
from common.imp_arg.candidate import Candidate
from common.imp_arg.tree_pointer import TreePointer
from dataset.imp_arg import ImplicitArgumentInstance
from dataset.nltk import PTBReader
from utils import check_type, get_console_logger

log = get_console_logger()


class Proposition(object):
    def __init__(self, pred_pointer, imp_args, exp_args):
        self._pred_pointer = pred_pointer
        self._n_pred = ''
        self._v_pred = ''

        for label, arg_pointers in imp_args.items():
            for arg_pointer in arg_pointers:
                check_type(arg_pointer, TreePointer)
        self._imp_args = imp_args

        for label, arg_pointers in exp_args.items():
            for arg_pointer in arg_pointers:
                check_type(arg_pointer, TreePointer)

        self._exp_args = exp_args
        self._candidates = []

    @property
    def pred_pointer(self):
        return self._pred_pointer

    @property
    def n_pred(self):
        return self._n_pred

    @property
    def v_pred(self):
        return self._v_pred

    def set_pred(self, predicate_mapping):
        self._n_pred = predicate_mapping[str(self._pred_pointer)]
        self._v_pred = helper.nominal_predicate_mapping[self._n_pred]

    @property
    def imp_args(self):
        return self._imp_args

    def has_imp_arg(self, label):
        return label in self.imp_args

    def num_imp_args(self):
        return len(self._imp_args)

    def has_imp_arg_in_range(self, label, max_dist=2):
        if label in self.imp_args:
            for filler in self.imp_args[label]:
                pred_sentnum = self.pred_pointer.sentnum
                if pred_sentnum - max_dist <= filler.sentnum <= pred_sentnum:
                    return True
        return False

    def num_imp_args_in_range(self):
        return sum([1 for label in self.imp_args
                    if self.has_imp_arg_in_range(label)])

    def has_oracle(self, label):
        for candidate in self._candidates:
            if candidate.is_oracle(self.imp_args[label]):
                return True
        return False

    def num_oracles(self):
        return sum([1 for label in self.imp_args if self.has_oracle(label)])

    @property
    def exp_args(self):
        return self._exp_args

    def check_exp_args(self, nombank_instance, filter_conflict=False,
                       verbose=False):
        if verbose:
            put_log = log.warning
        else:
            put_log = log.debug

        unmatched_labels = deepcopy(self.exp_args.keys())

        fileid = self.pred_pointer.fileid
        sentnum = self.pred_pointer.sentnum

        if nombank_instance is not None:
            nombank_arg_dict = defaultdict(list)
            for arg_pointer, label in nombank_instance.arguments:
                cvt_label = helper.convert_nombank_label(label)
                if cvt_label:
                    nombank_arg_dict[cvt_label].append(arg_pointer)

            for label in nombank_arg_dict:
                nombank_args = nombank_arg_dict[label]

                if label not in self.exp_args:
                    message = \
                        '{} has {} in Nombank but not found in ' \
                        'explicit arguments.'.format(self.pred_pointer, label)
                    if filter_conflict:
                        put_log(message)
                        put_log(
                            'Adding missing explicit {}: {}.'.format(
                                label, nombank_args))
                        self.exp_args[label] = \
                            [TreePointer(fileid, sentnum, arg)
                             for arg in nombank_args]
                        if label in self.imp_args:
                            put_log(
                                'Removing implicit {}.'.format(label))
                            self.imp_args.pop(label, None)
                    else:
                        put_log('Ignored... ' + message)

                    continue

                exp_args = [p.tree_pointer for p in self.exp_args[label]]
                unmatched_labels.remove(label)

                if exp_args != nombank_args:
                    message = '{} has mismatch in {}: {} --> {}'.format(
                        self.pred_pointer, label, exp_args, nombank_args)
                    if len(nombank_args) == 1:
                        nombank_arg = nombank_args[0]
                        if isinstance(nombank_arg, NombankSplitTreePointer):
                            if all(p in nombank_arg.pieces for p in exp_args):
                                self.exp_args[label] = \
                                    [TreePointer(fileid, sentnum, nombank_arg)]
                                put_log('Replaced... ' + message)
                                continue
                        if isinstance(nombank_arg, NombankChainTreePointer):
                            if all(p in nombank_arg.pieces for p in exp_args):
                                put_log('Ignored... ' + message)
                                continue

                    raise AssertionError(message)

        if unmatched_labels:
            message = '{} has {} in explicit arguments but not found in ' \
                      'Nombank.'.format(self.pred_pointer, unmatched_labels)
            raise AssertionError(message)

    def parse_arg_subtrees(self, ptb_reader):
        check_type(ptb_reader, PTBReader)

        fileid = self.pred_pointer.fileid
        ptb_reader.read_file(helper.expand_wsj_fileid(fileid, '.mrg'))

        for label, fillers in self.imp_args.items():
            for arg_pointer in fillers:
                arg_pointer.parse_subtree(
                    ptb_reader.all_parsed_sents[arg_pointer.sentnum])

        for label, fillers in self.exp_args.items():
            for arg_pointer in fillers:
                arg_pointer.parse_subtree(
                    ptb_reader.all_parsed_sents[arg_pointer.sentnum])

    def filter_incorporated_argument(self, verbose=False):
        if verbose:
            put_log = log.warning
        else:
            put_log = log.debug

        pred_sentnum = self.pred_pointer.sentnum
        pred_wordnum = self.pred_pointer.tree_pointer.wordnum

        label_to_remove = []

        for label, fillers in self._imp_args.items():
            filtered_filler_idx_list = []
            for idx, filler in enumerate(fillers):
                if pred_sentnum == filler.sentnum \
                        and pred_wordnum in filler.flat_idx_list():
                    filtered_filler_idx_list.append(idx)
                    put_log('Find incorporated {} {} of {}, {}'.format(
                        label, filler, self.pred_pointer, self.n_pred))
            if filtered_filler_idx_list:
                self._imp_args[label] = \
                    [fillers[idx] for idx in range(len(fillers))
                     if idx not in filtered_filler_idx_list]
            if len(self._imp_args[label]) == 0:
                label_to_remove.append(label)

        for label in label_to_remove:
            put_log('Remove incorporated label {} of {}, {}'.format(
                label, self.pred_pointer, self.n_pred))
            self._imp_args.pop(label, None)

    def add_candidate(self, candidate):
        check_type(candidate, Candidate)
        self._candidates.append(candidate)

    def parse_arg_corenlp(self, doc, idx_mapping):
        check_type(doc, Document)

        for label, fillers in self.imp_args.items():
            for arg_pointer in fillers:
                sentnum = arg_pointer.sentnum
                corenlp_sent = doc.get_sent(sentnum)
                sent_idx_mapping = idx_mapping[sentnum]
                arg_pointer.parse_corenlp(corenlp_sent, sent_idx_mapping)

        for label, fillers in self.exp_args.items():
            for arg_pointer in fillers:
                sentnum = arg_pointer.sentnum
                corenlp_sent = doc.get_sent(sentnum)
                sent_idx_mapping = idx_mapping[sentnum]
                arg_pointer.parse_corenlp(corenlp_sent, sent_idx_mapping)

    @classmethod
    def build(cls, instance):
        check_type(instance, ImplicitArgumentInstance)

        pred_node = instance.pred_node

        tmp_imp_args = defaultdict(list)
        exp_args = defaultdict(list)

        for argument in instance.arguments:

            label = argument[0].lower()
            arg_node = argument[1]
            attribute = argument[2]

            # remove arguments located in sentences following the predicate
            if arg_node.fileid != pred_node.fileid or \
                    arg_node.sentnum > pred_node.sentnum:
                continue

            # add explicit arguments to exp_args
            if attribute == 'Explicit':
                exp_args[label].append(TreePointer.from_node(arg_node))
                # remove the label from tmp_imp_args, as we do not process
                # an implicit argument if some explicit arguments with
                # the same label exist
                tmp_imp_args.pop(label, None)

            # add non-explicit arguments to tmp_imp_args
            else:
                # do not add the argument when some explicit arguments with
                # the same label exist
                if label not in exp_args:
                    tmp_imp_args[label].append((arg_node, attribute))

        # process implicit arguments
        imp_args = defaultdict(list)
        for label, fillers in tmp_imp_args.items():

            # remove incorporated arguments from tmp_imp_args
            # incorporated argument: argument with the same node as
            # the predicate itself
            if pred_node in [node for node, _ in fillers]:
                continue

            # add non-split arguments to imp_args
            split_nodes = []
            for node, attribute in fillers:
                if attribute == '':
                    imp_args[label].append(TreePointer.from_node(node))
                else:
                    split_nodes.append(node)

            sentnum_set = set([node.sentnum for node in split_nodes])

            # group split arguments by their sentnum,
            # and sort pieces by nombank_pointer.wordnum within each group
            grouped_split_nodes = []
            for sentnum in sentnum_set:
                grouped_split_nodes.append(sorted(
                    [node for node in split_nodes if node.sentnum == sentnum],
                    key=lambda n: n.wordnum))

            # add each split pointer to imp_args
            for node_group in grouped_split_nodes:
                imp_args[label].append(TreePointer.from_node(*node_group))

        pred_pointer = TreePointer.from_node(pred_node)

        return cls(pred_pointer, imp_args, exp_args)
