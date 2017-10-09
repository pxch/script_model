from collections import defaultdict
from operator import itemgetter

from texttable import Texttable

from common.imp_arg import helper
from common.imp_arg.proposition import Proposition
from utils import check_type


def print_stats(all_propositions):
    propositions_by_pred = defaultdict(list)

    for proposition in all_propositions:
        check_type(proposition, Proposition)
        propositions_by_pred[proposition.n_pred].append(proposition)

    num_dict = {}

    for n_pred, propositions in propositions_by_pred.items():
        num_dict[n_pred] = [len(propositions)]
        num_dict[n_pred].append(
            sum([proposition.num_imp_args() for proposition in propositions]))
        num_dict[n_pred].append(
            sum([proposition.num_imp_args_in_range()
                 for proposition in propositions]))
        num_dict[n_pred].append(
            sum([proposition.num_oracles() for proposition in propositions]))

        for arg in helper.core_arg_list:
            num_dict[n_pred].append(
                sum([1 for proposition in propositions if
                     proposition.has_imp_arg(arg)]))

    num_total = [0] * len(num_dict.values()[0])

    table_content = []

    for n_pred, num in num_dict.items():
        table_row = [n_pred] + num[:2]
        table_row.append(float(num[1]) / num[0])
        table_row.extend(num[2:4])
        table_row.append(100. * float(num[3]) / num[1])
        table_row += num[4:]
        table_content.append(table_row)

        for i in range(len(num_total)):
            num_total[i] += num[i]

    table_content.sort(key=itemgetter(2), reverse=True)

    table_row = ['overall'] + num_total[:2]
    table_row.append(float(num_total[1]) / num_total[0])
    table_row.extend(num_total[2:4])
    table_row.append(100. * float(num_total[3]) / num_total[1])
    table_row += num_total[4:]

    table_content.append([''] * len(table_row))
    table_content.append(table_row)

    table_header = ['predicate', '# pred', '# iarg', '# iarg per pred',
                    '# iarg in range', '# oracle', 'oracle recall']
    table_header.extend(['# iarg_{}'.format(i) for i in range(5)])

    table = Texttable()
    table.set_deco(Texttable.BORDER | Texttable.HEADER)
    table.set_cols_align(['c'] * len(table_header))
    table.set_cols_valign(['m'] * len(table_header))
    table.set_cols_width([10] * len(table_header))
    table.set_precision(1)

    table.header(table_header)
    for row in table_content:
        table.add_row(row)

    print table.draw()
