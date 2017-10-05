from utils import consts


class CoreArgument(object):
    def __init__(self, word, pos, ner):
        self._word = word
        self._pos = pos
        assert ner in consts.valid_ner_tags or ner == '', \
            'unrecognized NER tag: ' + ner
        self._ner = ner

    @property
    def word(self):
        return self._word

    @property
    def pos(self):
        return self._pos

    @property
    def ner(self):
        return self._ner

    def __eq__(self, other):
        return self.word == other.word and self.pos == other.pos \
               and self.ner == other.ner

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{} // {} // {}'.format(self.word, self.pos, self.ner)
