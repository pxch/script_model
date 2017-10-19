from collections import Counter

from nltk.corpus.reader.nombank import NombankSplitTreePointer
from nltk.corpus.reader.nombank import NombankTreePointer
from nltk.corpus.reader.propbank import PropbankSplitTreePointer
from nltk.corpus.reader.propbank import PropbankTreePointer
from nltk.tree import Tree

from common.corenlp import Sentence
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
        if no_trace:
            return self._filtered_idx_list
        else:
            return self._idx_list

    def word_list(self, no_trace=True):
        if no_trace:
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


class CoreNLPInfo(object):
    def __init__(self, idx_list, head_idx, entity_idx, mention_idx):
        self.idx_list = idx_list
        self.head_idx = head_idx
        self.entity_idx = entity_idx
        self.mention_idx = mention_idx

    @classmethod
    def build(cls, idx_list, corenlp_sent, sent_idx_mapping, head_only=True,
              msg_prefix=''):
        assert all(idx in sent_idx_mapping for idx in idx_list)
        mapped_idx_list = [sent_idx_mapping.index(idx) for idx in idx_list]
        head_idx = -1
        if mapped_idx_list:
            head_idx = corenlp_sent.dep_graph.get_head_token_idx(
                mapped_idx_list[0], mapped_idx_list[-1] + 1, msg_prefix)

        entity_idx = -1
        mention_idx = -1

        if mapped_idx_list and head_idx != -1:
            head_token = corenlp_sent.get_token(head_idx)
            if head_token.coref:
                entity_idx = head_token.coref_idx()
                mention_idx = head_token.mention_idx()

            elif not head_only:
                token_list = \
                    [corenlp_sent.get_token(idx) for idx in mapped_idx_list]
                entity_idx_counter = Counter()
                for token in token_list:
                    if token.coref_idx() != -1:
                        entity_idx_counter[token.coref_idx()] += 1

                if entity_idx_counter:
                    entity_idx = entity_idx_counter.most_common(1)[0][0]

                    mention_idx_counter = Counter()
                    for token in token_list:
                        if token.coref_idx() == entity_idx:
                            mention_idx = token.mention_idx()
                            mention_idx_counter[mention_idx] += 1

                    mention_idx = \
                        mention_idx_counter.most_common(1)[0][0]

        return cls(mapped_idx_list, head_idx, entity_idx, mention_idx)


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

        # list of subtrees parsed from gold parsed tree
        # length of the list should be 1 if tree_pointer is not a split pointer
        self._subtree_list = []

        # list of CoreNLP chunks parsed from and CoreNLP Document
        self._corenlp_list = []

        # index of the head piece (0 if not a split pointer)
        self._head_piece = -1
        # index of the entity the pointer is linked to, -1 otherwise
        self._entity_idx = -1
        # index of the mention the pointer is linked to, -1 otherwise
        self._mention_idx = -1

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

    def has_corenlp(self):
        return len(self._corenlp_list) > 0

    def parse_corenlp(self, corenlp_sent, sent_idx_mapping):
        assert self.has_subtree(), 'must have subtree info to parse CoreNLP'

        if self.has_corenlp():
            log.warning('Overriding existing CoreNLP info')
        self._corenlp_list = []

        check_type(corenlp_sent, Sentence)

        for subtree in self.subtree_list:
            self._corenlp_list.append(CoreNLPInfo.build(
                idx_list=subtree.idx_list(no_trace=True),
                corenlp_sent=corenlp_sent,
                sent_idx_mapping=sent_idx_mapping,
                head_only=False,
                msg_prefix=self.fileid))

        self._head_piece = -1
        min_root_path_length = 999

        for piece_idx, corenlp_info in enumerate(self.corenlp_list):
            if corenlp_info.head_idx != -1:
                root_path = corenlp_sent.dep_graph.get_root_path(
                    corenlp_info.head_idx, msg_prefix=self.fileid)
                # with same root_path_length, take the latter token
                if len(root_path) <= min_root_path_length:
                    min_root_path_length = len(root_path)
                    self._head_piece = piece_idx

        self._entity_idx = -1
        self._mention_idx = -1

        head_corenlp_info = self.corenlp_list[self._head_piece]
        if head_corenlp_info.entity_idx != -1:
            self._entity_idx = head_corenlp_info.entity_idx
            self._mention_idx = head_corenlp_info.mention_idx
        else:
            entity_idx_counter = Counter()
            for corenlp_info in self.corenlp_list:
                if corenlp_info.entity_idx != -1:
                    entity_idx_counter[corenlp_info.entity_idx] += 1

            if entity_idx_counter:
                self._entity_idx = entity_idx_counter.most_common(1)[0][0]

                mention_idx_counter = Counter()
                for corenlp_info in self.corenlp_list:
                    if corenlp_info.entity_idx == self._entity_idx:
                        mention_idx_counter[corenlp_info.mention_idx] += 1

                self._mention_idx = mention_idx_counter.most_common(1)[0][0]

    @property
    def corenlp_list(self):
        return self._corenlp_list

    @property
    def head_piece(self):
        return self._head_piece

    @property
    def entity_idx(self):
        return self._entity_idx

    @property
    def mention_idx(self):
        return self._mention_idx

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
