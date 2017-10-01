from copy import deepcopy

from lxml import etree
from sklearn.model_selection import KFold

from utils import get_console_logger

log = get_console_logger()


class ImplicitArgumentPointer(object):
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

    def to_rich_tree_pointer(self):
        # TODO: write method to convert to RichTreePointer
        raise NotImplementedError()

    @classmethod
    def parse(cls, text):
        items = text.split(':')
        assert len(items) == 4, \
            'expecting 4 parts separated by ":", {} found.'.format(len(items))
        fileid = items[0]
        sentnum = int(items[1])
        wordnum = int(items[2])
        height = int(items[3])
        return ImplicitArgumentPointer(fileid, sentnum, wordnum, height)


class ImplicitArgumentInstance(object):
    def __init__(self, pred_pointer, arguments):
        self.pred_pointer = pred_pointer
        self.arguments = arguments

    def __eq__(self, other):
        # only compare predicate pointer, assuming that any predicate node
        # can have at most one implicit argument instance
        return str(self.pred_pointer) == str(other.pred_pointer)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def parse(cls, text):
        root = etree.fromstring(text)
        pred_pointer = ImplicitArgumentPointer.parse(root.get('for_node'))
        arguments = []
        for child in root:
            label = child.get('value')
            arg_pointer = ImplicitArgumentPointer.parse(child.get('node'))
            attribute = ''
            if len(child[0]) > 0:
                attribute = child[0][0].text

            arguments.append((label, deepcopy(arg_pointer), attribute))
        return cls(pred_pointer, arguments)

    def __str__(self):
        xml_string = '<annotations for_node="{}">'.format(self.pred_pointer)
        for label, arg_pointer, attribute in self.arguments:
            xml_string += \
                '<annotation value="{}" node="{}">'.format(label, arg_pointer)
            xml_string += '<attributes>{}</attributes>'.format(
                '<attribute>{}</attribute>'.format(attribute)
                if attribute else '')
            xml_string += '</annotation>'
        xml_string += '</annotations>'
        return xml_string


class ImplicitArgumentReader(object):
    def __init__(self):
        # list of all implicit argument instances
        self.all_instances = []

        # list of all implicit argument instances, sorted by predicate node
        self.all_instances_sorted = []
        # order of original instances in sorted list
        self.instance_order_list = []

        # number of splits in cross validation
        self.n_splits = 0

        # list of cross validation train/test instance indices in sorted list
        self.train_test_folds = []

    def read_dataset(self, file_path):
        log.info('Reading implicit argument dataset from {}'.format(file_path))
        input_xml = open(file_path, 'r')

        self.all_instances = []
        for line in input_xml.readlines()[1:-1]:
            instance = ImplicitArgumentInstance.parse(line.strip())
            self.all_instances.append(instance)

        input_xml.close()

        log.info('Found {} instances'.format(len(self.all_instances)))

    def sort_dataset(self):
        self.all_instances_sorted = sorted(
            self.all_instances, key=lambda ins: str(ins.pred_pointer))

        self.instance_order_list = [self.all_instances_sorted.index(instance)
                                    for instance in self.all_instances]

    def set_n_splits(self, n_splits):
        self.n_splits = n_splits

    def create_train_test_folds(self):
        assert self.n_splits > 0
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        self.train_test_folds = list(kf.split(self.instance_order_list))

    @classmethod
    def from_dataset(cls, file_path, n_splits=10):
        reader = cls()
        reader.read_dataset(file_path)
        reader.sort_dataset()
        reader.set_n_splits(n_splits)
        reader.create_train_test_folds()
        return reader

    def save_dataset(self, file_path):
        log.info('Printing implicit argument dataset to {}'.format(file_path))
        fout = open(file_path, 'w')

        fout.write('<annotations>\n')

        for instance in self.all_instances:
            fout.write(str(instance) + '\n')

        fout.write('</annotations>\n')

        fout.close()

    def get_all_instances(self, sort=True):
        if sort:
            return self.all_instances_sorted
        else:
            return self.all_instances

    def get_train_fold(self, fold_idx):
        return self.train_test_folds[fold_idx][0]

    def get_test_fold(self, fold_idx):
        return self.train_test_folds[fold_idx][1]
