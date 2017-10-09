from nltk.corpus.reader.nombank import NombankSplitTreePointer
from nltk.corpus.reader.nombank import NombankTreePointer
from nltk.corpus.reader.propbank import PropbankSplitTreePointer
from nltk.corpus.reader.propbank import PropbankTreePointer
from nltk.tree import Tree

from dataset.imp_arg import ImplicitArgumentNode
from utils import check_type, get_console_logger

log = get_console_logger()


class Subtree(object):
    def __init__(self, treepos, idx_list, word_list, pos_list):
        self._treepos = treepos
        self._idx_list = idx_list
        self._word_list = word_list
        self._pos_list = pos_list

        self._filtered_idx_list = \
            [idx for i, idx in enumerate(idx_list) if pos_list[i] != '-NONE-']
        self._filtered_word_list = \
            [w for i, w in enumerate(word_list) if pos_list[i] != '-NONE-']

    @property
    def treepos(self):
        return self._treepos

    def idx_list(self, no_trace=True):
        if filter:
            return self._filtered_idx_list
        else:
            return self._idx_list

    def word_list(self, no_trace=True):
        if filter:
            return self._filtered_word_list
        else:
            return self._word_list

    def surface(self, no_trace=True):
        return ' '.join(self.word_list(no_trace))

    def equal(self, other):
        if isinstance(other, Subtree):
            # match index list without tracing node
            if self._filtered_idx_list == other._filtered_idx_list:
                return True
        return False

    def equal_without_preceding_preposition(self, other):
        if isinstance(other, Subtree):
            # match by removing preceding preposition
            if self._idx_list[1:] == other._idx_list and \
                    self.treepos == other.treepos[:-1] and \
                    self._pos_list[0] in ['IN', 'TO']:
                return True
        return False

    @classmethod
    def parse(cls, tree_pointer, tree):
        assert isinstance(tree_pointer, NombankTreePointer) or \
               isinstance(tree_pointer, PropbankTreePointer)
        treepos = tree_pointer.treepos(tree)
        idx_list = []
        word_list = []
        pos_list = []
        for idx in range(len(tree.leaves())):
            if tree.leaf_treeposition(idx)[:len(treepos)] == treepos:
                idx_list.append(idx)
                word_list.append(tree.pos()[idx][0])
                pos_list.append(tree.pos()[idx][1])

        return cls(treepos, idx_list, word_list, pos_list)


class TreePointer(object):
    def __init__(self, fileid, sentnum, tree_pointer):
        # treebank file name, format: wsj_0000
        assert fileid[:4] == 'wsj_' and fileid[4:].isdigit(), \
            'fileid must be in wsj_0000 format'
        self._fileid = fileid

        # sentence number, starting from 0
        assert type(sentnum) == int and sentnum >= 0, \
            'sentnum must be a non-negative integer'
        self._sentnum = sentnum

        # is_split_pointer = False if Propbank/NombankTreePointer
        if isinstance(tree_pointer, NombankTreePointer) or \
                isinstance(tree_pointer, PropbankTreePointer):
            self._is_split_pointer = False
        # is_split_pointer = True if Propbank/NombankSplitTreePointer
        elif isinstance(tree_pointer, NombankSplitTreePointer) or \
                isinstance(tree_pointer, PropbankSplitTreePointer):
            self._is_split_pointer = True
        # raise AssertionError otherwise
        else:
            raise AssertionError(
                'Unrecognized tree_pointer type: {}'.format(type(tree_pointer)))
        # tree pointer
        self._tree_pointer = tree_pointer

        # list of subtrees parsed from self._tree_pointer and self._tree
        # length of the list should be 1 if tree_pointer is not a split pointer
        self._subtree_list = []

    @property
    def fileid(self):
        return self._fileid

    @property
    def sentnum(self):
        return self._sentnum

    @property
    def tree_pointer(self):
        return self._tree_pointer

    @property
    def is_split_pointer(self):
        return self._is_split_pointer

    def has_subtree(self):
        return len(self._subtree_list) > 0

    def parse_subtree(self, tree):
        check_type(tree, Tree)

        if self.has_subtree():
            log.warning('Overriding existing subtrees')
        self._subtree_list = []

        if self._is_split_pointer:
            for piece in self.tree_pointer.pieces:
                self._subtree_list.append(Subtree.parse(piece, tree))
        else:
            self._subtree_list.append(Subtree.parse(self.tree_pointer, tree))

    @property
    def subtree_list(self):
        return self._subtree_list

    def idx_list(self, no_trace=True):
        assert self.has_subtree()
        return [subtree.idx_list(no_trace) for subtree in self._subtree_list]

    def flat_idx_list(self, no_trace=True):
        return [idx for sub_list in self.idx_list(no_trace) for idx in sub_list]

    def word_list(self, no_trace=True):
        assert self.has_subtree()
        return [subtree.word_list(no_trace) for subtree in self._subtree_list]

    def flat_word_list(self, no_trace=True):
        return [word for sub_list in self.word_list(no_trace)
                for word in sub_list]

    def surface(self, no_trace=True):
        assert self.has_subtree()
        return ' '.join(
            [subtree.surface(no_trace) for subtree in self._subtree_list])

    def __str__(self):
        return '{}:{}:{}'.format(
            self.fileid, self.sentnum, self.tree_pointer)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return str(self) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def from_node(cls, *args):
        assert len(args) >= 1, 'must provide at least one node'
        nombank_pointer_list = []

        fileid = -1
        sentnum = -1
        for idx, arg in enumerate(args):
            check_type(arg, ImplicitArgumentNode)
            if idx == 0:
                fileid = arg.fileid
                sentnum = arg.sentnum
            else:
                assert fileid == arg.fileid, \
                    'inconsistent fileid: {}, {}'.format(fileid, arg.fileid)
                assert sentnum == arg.sentnum, \
                    'inconsistent sentnum: {}, {}'.format(sentnum, arg.sentnum)
            nombank_pointer_list.append(
                NombankTreePointer(arg.wordnum, arg.height))

        if len(nombank_pointer_list) == 1:
            tree_pointer = nombank_pointer_list[0]
        else:
            tree_pointer = NombankSplitTreePointer(nombank_pointer_list)
        return cls(fileid, sentnum, tree_pointer)
