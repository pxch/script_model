from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.nombank import NombankInstance
from nltk.corpus.reader.nombank import NombankTreePointer
from nltk.corpus.reader.propbank import PropbankChainTreePointer
from nltk.corpus.reader.propbank import PropbankInstance
from nltk.corpus.reader.propbank import PropbankTreePointer

from common.imp_arg import helper
from common.imp_arg.tree_pointer import TreePointer


class Candidate(object):
    def __init__(self, fileid, sentnum, pred, arg, arg_label, tree, src):
        self._fileid = fileid
        self._sentnum = sentnum

        self._arg_pointer = TreePointer(fileid, sentnum, arg)
        self._arg_pointer.parse_subtree(tree)

        pred_pointer = TreePointer(fileid, sentnum, pred)
        assert not pred_pointer.is_split_pointer, \
            'pred_pointer cannot be a split pointer'
        pred_pointer.parse_subtree(tree)

        assert src in ['P', 'N'], \
            'source can only be P (Propbank) or N (Nombank)'
        self._pred_pointer_list = [(pred_pointer, arg_label, src)]

    @property
    def arg_pointer(self):
        return self._arg_pointer

    @property
    def pred_pointer_list(self):
        return self._pred_pointer_list

    def merge(self, candidate):
        assert isinstance(candidate, Candidate)
        assert self.arg_pointer == candidate.arg_pointer
        self._pred_pointer_list.extend(candidate.pred_pointer_list)

    @classmethod
    def from_instance(cls, instance):
        candidate_list = []

        fileid = helper.shorten_wsj_fileid(instance.fileid)
        sentnum = instance.sentnum

        if isinstance(instance, PropbankInstance):
            pred = PropbankTreePointer(instance.wordnum, 0)
            src = 'P'
        elif isinstance(instance, NombankInstance):
            pred = NombankTreePointer(instance.wordnum, 0)
            src = 'N'
        else:
            raise AssertionError(
                'unrecognized instance type: {}'.format(type(instance)))

        tree = instance.tree

        for arg, label in instance.arguments:
            cvt_label = helper.convert_nombank_label(label)
            if cvt_label in helper.core_arg_list:
                if isinstance(arg, NombankChainTreePointer) or \
                        isinstance(arg, PropbankChainTreePointer):
                    for p in arg.pieces:
                        candidate_list.append(cls(
                            fileid, sentnum, pred, p, cvt_label, tree, src))
                else:
                    candidate_list.append(cls(
                        fileid, sentnum, pred, arg, cvt_label, tree, src))

        return candidate_list

    def is_oracle(self, imp_args):
        for imp_arg in imp_args:
            if imp_arg.fileid == self.arg_pointer.fileid \
                    and imp_arg.sentnum == self.arg_pointer.sentnum:

                for arg_subtree in imp_arg.subtree_list:
                    for cand_subtree in self.arg_pointer.subtree_list:
                        if cand_subtree.equal(arg_subtree):
                            return True
                        if cand_subtree.equal_without_preceding_preposition(
                                arg_subtree):
                            return True

        return False

    def parse_corenlp(self, corenlp_sent, sent_idx_mapping):
        self._arg_pointer.parse_corenlp(corenlp_sent, sent_idx_mapping)
        for pred_pointer, _, _ in self._pred_pointer_list:
            pred_pointer.parse_corenlp(corenlp_sent, sent_idx_mapping)

    def dice_score(self, imp_args):
        dice_score = 0.0

        if len(imp_args) > 0:
            dice_score_list = []
            for arg in imp_args:
                dice_score_list.append(self.arg_pointer.dice_score(arg))

            dice_score = max(dice_score_list)

        return dice_score
