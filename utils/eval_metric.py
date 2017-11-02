import abc


class BaseEvalMetric(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_metric(self, other):
        pass

    @abc.abstractmethod
    def precision(self):
        return 0.

    @abc.abstractmethod
    def recall(self):
        return 0.

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        if precision + recall > 0:
            return 2. * precision * recall / (precision + recall)
        else:
            return 0.

    def __str__(self):
        return 'precision = {:6.2f}, recall = {:6.2f}, f1 = {:6.2f}'.format(
            self.precision(), self.recall(), self.f1())
