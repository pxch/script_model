from nltk.corpus.reader.nombank import NombankChainTreePointer
from nltk.corpus.reader.propbank import PropbankChainTreePointer

from common.imp_arg import helper
from common.imp_arg.tree_pointer import TreePointer
from utils import get_console_logger

log = get_console_logger()


class Candidate(object):
    def __init__(self, fileid, sentnum, pred, arg, arg_label, tree):
        self._fileid = fileid
        self._sentnum = sentnum

        self._arg_pointer = TreePointer(fileid, sentnum, arg)
        self._arg_pointer.parse_subtree(tree)

        pred_pointer = TreePointer(fileid, sentnum, pred)
        pred_pointer.parse_subtree(tree)
        self._pred_pointer_list = [(pred_pointer, arg_label)]

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
        pred = instance.predicate
        tree = instance.tree

        for arg_pointer, label in instance.arguments:
            cvt_label = helper.convert_nombank_label(label)
            if cvt_label in helper.core_arg_list:
                if isinstance(arg_pointer, NombankChainTreePointer) or \
                        isinstance(arg_pointer, PropbankChainTreePointer):
                    for p in arg_pointer.pieces:
                        candidate_list.append(cls(
                            fileid, sentnum, pred, p, cvt_label, tree))
                else:
                    candidate_list.append(cls(
                        fileid, sentnum, pred, arg_pointer, cvt_label, tree))

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
