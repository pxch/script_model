from copy import deepcopy

from nltk.tree import Tree

from common.corenlp.coreference import Coreference
from common.corenlp.dependency import Dependency
from common.corenlp.mention import Mention
from common.corenlp.sentence import Sentence
from common.corenlp.token import Token
from utils import consts, convert_corenlp_ner_tag


class CoreNLPTarget(object):
    def __init__(self):
        self.sents = []
        self.corefs = []
        self.sent = None
        self.coref = None
        self.tag = ''
        self.word = ''
        self.lemma = ''
        self.pos = ''
        self.ner = ''
        self.dep_label = ''
        self.gov_idx = -1
        self.dep_idx = -1
        self.extra = False
        # string representation of the constituency tree (nltk.tree.Tree)
        self.tree_str = ''
        self.sent_idx = -1
        self.start_token_idx = -1
        self.end_token_idx = -1
        self.head_token_idx = -1
        self.rep = False
        self.text = ''
        self.parse_sent = False
        self.parse_dep = False
        self.parse_coref = False
        self.copied_dep = False

    def start(self, tag, attrib):
        self.tag = tag
        if tag == 'sentences':
            self.parse_sent = True
        elif tag == 'sentence':
            if self.parse_sent:
                self.sent = Sentence(int(attrib['id']) - 1)
        elif tag == 'dependencies':
            if attrib['type'] == consts.corenlp_dependency_type \
                    and self.parse_sent:
                self.parse_dep = True
                self.copied_dep = False
        elif tag == 'dep':
            if self.parse_dep:
                self.dep_label = attrib['type']
                if 'extra' in attrib:
                    self.extra = True
        elif tag == 'governor':
            if self.parse_dep:
                self.gov_idx = int(attrib['idx']) - 1
                if 'copy' in attrib:
                    self.copied_dep = True
        elif tag == 'dependent':
            if self.parse_dep:
                self.dep_idx = int(attrib['idx']) - 1
                if 'copy' in attrib:
                    self.copied_dep = True
        elif tag == 'coreference':
            if not self.parse_coref:
                self.parse_coref = True
            self.coref = Coreference(len(self.corefs))
        elif tag == 'mention':
            if self.parse_coref:
                if 'representative' in attrib:
                    self.rep = True

    def data(self, data):
        data = data.strip()
        if data != '':
            if self.parse_sent:
                if self.tag == 'word':
                    self.word += data
                elif self.tag == 'lemma':
                    self.lemma += data
                elif self.tag == 'POS':
                    self.pos += data
                elif self.tag == 'NER':
                    self.ner += data
                elif self.tag == 'parse':
                    self.tree_str += data
            elif self.parse_coref:
                if self.tag == 'sentence':
                    self.sent_idx = int(data) - 1
                elif self.tag == 'start':
                    self.start_token_idx = int(data) - 1
                elif self.tag == 'end':
                    self.end_token_idx = int(data) - 1
                elif self.tag == 'head':
                    self.head_token_idx = int(data) - 1
                elif self.tag == 'text':
                    self.text += data

    def end(self, tag):
        self.tag = ''
        if tag == 'sentences':
            if self.parse_sent:
                self.parse_sent = False
        elif tag == 'sentence':
            if self.parse_sent:
                if self.sent is not None:
                    self.sents.append(deepcopy(self.sent))
                    self.sent = None
        elif tag == 'token':
            # map corenlp ner tags to coarse grained ner tags
            token = Token(self.word,
                          self.lemma,
                          self.pos,
                          ner=convert_corenlp_ner_tag(self.ner))
            self.sent.add_token(deepcopy(token))
            self.word = ''
            self.lemma = ''
            self.pos = ''
            self.ner = ''
        elif tag == 'parse':
            self.sent.tree = Tree.fromstring(self.tree_str)
            self.tree_str = ''
        elif tag == 'dependencies':
            if self.parse_dep:
                self.parse_dep = False
        elif tag == 'dep':
            if self.parse_dep:
                if not self.copied_dep:
                    if self.dep_label != 'root':
                        dep = Dependency(self.dep_label,
                                         self.gov_idx,
                                         self.dep_idx,
                                         self.extra)
                        self.sent.add_dep(deepcopy(dep))
                else:
                    self.copied_dep = False
                self.dep_label = ''
                self.gov_idx = -1
                self.dep_idx = -1
                self.extra = False
        elif tag == 'coreference':
            if self.parse_coref:
                if self.coref is not None:
                    self.corefs.append(deepcopy(self.coref))
                    self.coref = None
                else:
                    self.parse_coref = False
        elif tag == 'mention':
            mention = Mention(self.sent_idx,
                              self.start_token_idx,
                              self.end_token_idx,
                              head_token_idx=self.head_token_idx,
                              rep=self.rep,
                              text=self.text.encode('ascii', 'ignore'))
            self.coref.add_mention(deepcopy(mention))
            self.sent_idx = -1
            self.start_token_idx = -1
            self.end_token_idx = -1
            self.head_token_idx = -1
            self.rep = False
            self.text = ''

    def close(self):
        sents, self.sents = self.sents, []
        corefs, self.corefs = self.corefs, []
        return sents, corefs
