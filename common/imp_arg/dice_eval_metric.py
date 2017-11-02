from utils import BaseEvalMetric


class DiceEvalMetric(BaseEvalMetric):
    def __init__(self, num_gold=0, num_model=0, num_dice=0):
        self.num_gold = num_gold
        self.num_model = num_model
        self.num_dice = num_dice

    def add_metric(self, other):
        assert isinstance(other, DiceEvalMetric)
        self.num_gold += other.num_gold
        self.num_model += other.num_model
        self.num_dice += other.num_dice

    def precision(self):
        if self.num_model > 0:
            return 100. * self.num_dice / self.num_model
        else:
            return 0.

    def recall(self):
        if self.num_gold > 0:
            return 100. * self.num_dice / self.num_gold
        else:
            return 0.

    def to_text(self):
        return 'num_gold = {}, num_model = {}, num_dice = {:.2f}, {}'.format(
            self.num_gold, self.num_model, self.num_dice, str(self))

    @classmethod
    def eval(cls, num_gold, dice_score_dict, score_matrix, thres=0,
             missing_labels=None):
        if missing_labels:
            assert all(label in dice_score_dict.keys() for label
                       in missing_labels)
        else:
            missing_labels = dice_score_dict.keys()

        num_model = 0
        num_dice = 0
        for row_idx, arg_label in enumerate(missing_labels):
            max_score = score_matrix[row_idx, :].max()
            max_candidate_idx = score_matrix[row_idx, :].argmax()
            if max_score >= thres:
                num_model += 1
                num_dice += dice_score_dict[arg_label][max_candidate_idx]

        return cls(
            num_gold=num_gold, num_model=num_model, num_dice=num_dice)
