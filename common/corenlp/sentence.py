from dependency import Dependency, DependencyGraph
from token import Token
from utils import check_type


class Sentence(object):
    def __init__(self, idx):
        # index of the sentence in the document
        self._idx = idx
        # list of all tokens in the sentence
        self._tokens = []
        # list of all dependencies in the sentence
        # excluding the root dependency
        self._deps = []
        # dependency graph built upon all dependencies
        self._dep_graph = None

    @property
    def tokens(self):
        return self._tokens

    def add_token(self, token):
        check_type(token, Token)
        # set the sent_idx attrib of the token
        token.sent_idx = self._idx
        # set the token_idx attrib of the token
        token.token_idx = len(self._tokens)
        self._tokens.append(token)

    def add_dep(self, dep):
        check_type(dep, Dependency)
        self._deps.append(dep)

    def get_token(self, idx):
        assert 0 <= idx < len(self._tokens), \
            'Token idx {} out of range'.format(idx)
        result = self._tokens[idx]
        check_type(result, Token)
        return self._tokens[idx]

    def build_dep_graph(self):
        self._dep_graph = DependencyGraph(self._idx, len(self._tokens))
        self._dep_graph.build(self._deps)

    @property
    def dep_graph(self):
        return self._dep_graph

    def lookup_label(self, direction, token_idx, dep_label):
        return [self.get_token(idx) for idx in
                self._dep_graph.lookup_label(direction, token_idx, dep_label)]

    # get list of all subjective token indices for a predicate
    def get_subj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('head', pred_idx, 'nsubj'))
        # agent of passive verb
        results.extend(self.lookup_label('head', pred_idx, 'nmod:agent'))
        # controlling subject
        results.extend(self.lookup_label('head', pred_idx, 'nsubj:xsubj'))
        return sorted(results, key=lambda token: token.token_idx)

    # get list of all objective token indices for a predicate
    def get_dobj_list(self, pred_idx):
        results = []
        results.extend(self.lookup_label('head', pred_idx, 'dobj'))
        # nsubjpass of passive verb
        results.extend(self.lookup_label('head', pred_idx, 'nsubjpass'))
        # TODO: include acl relation?
        # results.extend(self.lookup_label('mod', pred_idx, 'acl'))
        return sorted(results, key=lambda token: token.token_idx)

    # get list of all prepositional objective token indices for a predicate
    def get_pobj_list(self, pred_idx):
        results = []
        # look for all nmod dependencies
        for label, indices in self._dep_graph.lookup('head', pred_idx).items():
            if label.startswith('nmod') and ':' in label:
                prep_label = label.split(':')[1]
                # exclude nmod:agent (subject)
                if prep_label != 'agent':
                    results.extend([(prep_label, self.get_token(idx))
                                    for idx in indices])
        return sorted(results, key=lambda pair: pair[1].token_idx)

    def __str__(self):
        return ' '.join([str(token) for token in self._tokens]) + \
               '\t#DEP#\t' + ' '.join([str(dep) for dep in self._deps])

    def pretty_print(self):
        return ' '.join([token.pretty_print() for token in self._tokens]) + \
               '\n\t' + ' '.join([str(dep) for dep in self._deps])

    def plain_text(self):
        return ' '.join([token.word for token in self._tokens])
