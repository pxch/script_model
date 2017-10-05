from copy import deepcopy

from lxml import etree


class ImplicitArgumentNode(object):
    def __init__(self, fileid, sentnum, wordnum, height):
        # treebank file name, format: wsj_0000
        assert fileid[:4] == 'wsj_' and fileid[4:].isdigit()
        self.fileid = fileid
        # sentence number, starting from 0
        self.sentnum = sentnum
        # word number, starting from 0
        self.wordnum = wordnum
        # height, min_value = 1
        self.height = height

    def __eq__(self, other):
        return self.fileid == other.fileid and \
               self.sentnum == other.sentnum and \
               self.wordnum == other.wordnum and \
               self.height == other.height

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '{}:{}:{}:{}'.format(
            self.fileid, self.sentnum, self.wordnum, self.height)

    @classmethod
    def parse(cls, text):
        items = text.split(':')
        assert len(items) == 4, \
            'expecting 4 parts separated by ":", {} found.'.format(len(items))
        fileid = items[0]
        sentnum = int(items[1])
        wordnum = int(items[2])
        height = int(items[3])
        return ImplicitArgumentNode(fileid, sentnum, wordnum, height)


class ImplicitArgumentInstance(object):
    def __init__(self, pred_node, arguments):
        self.pred_node = pred_node
        self.arguments = arguments

    def __eq__(self, other):
        if not isinstance(other, ImplicitArgumentInstance):
            return False
        else:
            # only compare predicate node, assuming that any predicate node
            # can have at most one implicit argument instance
            return str(self.pred_node) == str(other.pred_node)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def parse(cls, text):
        root = etree.fromstring(text)
        pred_node = ImplicitArgumentNode.parse(root.get('for_node'))
        arguments = []
        for child in root:
            label = child.get('value')
            arg_node = ImplicitArgumentNode.parse(child.get('node'))
            attribute = ''
            if len(child[0]) > 0:
                attribute = child[0][0].text

            arguments.append((label, deepcopy(arg_node), attribute))
        return cls(pred_node, arguments)

    def __str__(self):
        xml_string = '<annotations for_node="{}">'.format(self.pred_node)
        for label, arg_node, attribute in self.arguments:
            xml_string += \
                '<annotation value="{}" node="{}">'.format(label, arg_node)
            xml_string += '<attributes>{}</attributes>'.format(
                '<attribute>{}</attribute>'.format(attribute)
                if attribute else '')
            xml_string += '</annotation>'
        xml_string += '</annotations>'
        return xml_string
